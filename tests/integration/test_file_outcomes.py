"""Exercise file completion through the real streaming adapter and tool wrapper."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.capabilities.file import FileWriteParams, file_write
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import AgentConfig, CapabilityResult
from tests.unit.test_task_blocked import _astream, _delta


@pytest.mark.parametrize("factory", ["asyncio", "uvloop"])
@pytest.mark.parametrize("expires", [False, True])
def test_execution_deadline_across_event_loops(monkeypatch, tmp_path, factory, expires):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNE_REQUIREMENT_GATE", "0")
    monkeypatch.setenv("RUNE_ADVISOR", "0")
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if expires:
            await asyncio.sleep(10)
        return _astream([_delta(content="Hello! How can I help?", finish_reason="stop")])

    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)

    async def run():
        loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=1))
        loop._config.timeout_seconds = 0.1 if expires else 10
        trace = await loop._execute_loop(
            goal="Hello", system_prompt="Answer the greeting.", tools=[], max_iterations=1,
            classification=ClassificationResult(goal_type="chat", confidence=.99, tier=2),
            context={"workspace_root": str(tmp_path)},
        )
        assert trace.reason == ("error: Execution deadline exceeded." if expires else "completed")
        assert len(calls) == 1
        assert calls[0]["num_retries"] == calls[0]["max_retries"] == 0
        assert 0 < calls[0]["timeout"] <= loop._config.timeout_seconds

    loop_factory = asyncio.SelectorEventLoop if factory == "asyncio" else pytest.importorskip("uvloop").new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["success", "transport_recovery", "denied", "recovered", "recovered_shell", "unrelated_file",
                                "unrelated_shell", "failed_shell", "denied_code", "unknown_code"])
async def test_completion_requires_successful_file_outcomes(monkeypatch, tmp_path, case):
    for key, value in {
        "RUNE_HOME": str(tmp_path / "home"), "RUNE_IN_BEST_OF": "1",
        "RUNE_AUTO_SKILL": "0", "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0",
        "RUNE_ADVISOR": "0", "RUNE_BENCH_EVIDENCE_GATE": "0",
        "RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT": "0", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }.items():
        monkeypatch.setenv(key, value)
    attempts = 0
    streams = 0
    filename = "report.py" if case in {"denied_code", "unknown_code"} else "report.txt"

    async def write(params):
        nonlocal attempts
        attempts += 1
        if case not in {"success", "transport_recovery"} and attempts == 1:
            return CapabilityResult(success=False, error="Permission denied",
                                    metadata={} if case == "unknown_code" else {"action_status": "not_executed"})
        return await file_write(params)

    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name="file_write", description="Write file",
                                          parameters_model=FileWriteParams, execute=write))
    from rune.capabilities.bash import BashParams

    async def shell(params):
        if case == "recovered_shell":
            from rune.capabilities.bash import bash_execute

            return await bash_execute(params)
        return CapabilityResult(success=case != "failed_shell", output="done",
                                metadata={"action_status": "not_executed"} if case == "failed_shell" else {})

    registry.register(CapabilityDefinition(name="bash_execute", description="Run command",
                                          parameters_model=BashParams, execute=shell))

    async def completion(**kwargs):
        nonlocal streams
        if not kwargs.get("stream"):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"report.txt": "output", "other.txt": "output"})))], usage=None)
        streams += 1
        assert streams <= 5, "Completion recovery must be bounded"
        if case == "transport_recovery" and streams == 2:
            raise httpx.ConnectError("Connection failed before sending the follow-up request")
        if case == "transport_recovery" and streams == 3:
            assert any(m["role"] == "tool" and "report.txt" in str(m["content"]) for m in kwargs["messages"])
            assert attempts == 1
        if streams == 1:
            name = "bash_execute" if case == "failed_shell" else "file_write"
            args = {"command": "echo done"} if name == "bash_execute" else {"path": filename, "content": "report"}
        elif streams == 2 and case in {"recovered", "recovered_shell", "unrelated_file", "unrelated_shell"}:
            name = "bash_execute" if case in {"unrelated_shell", "recovered_shell"} else "file_write"
            args = ({"command": "printf report > report.txt" if case == "recovered_shell" else "echo done"} if name == "bash_execute" else
                    {"path": "other.txt" if case == "unrelated_file" else "report.txt", "content": "report"})
        else:
            return _astream([_delta(content="Done. The report was saved.", finish_reason="stop")])
        return _astream([_delta(tool_calls=[{"index": 0, "id": f"call{streams}", "name": name,
                                            "arguments": json.dumps(args)}], finish_reason="tool_calls")])

    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda: registry)
    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=4))
    loop._token_budget.total = 300_000
    loop._workspace_root = str(tmp_path)
    trace = await loop._execute_loop(
        goal="Create report.txt", system_prompt="Create the requested file.",
        tools=["file_write", "bash_execute"], max_iterations=4,
        classification=ClassificationResult(goal_type="full", confidence=.99, tier=2, output_expectation="file"),
        context={"workspace_root": str(tmp_path)},
    )
    completed = case in {"success", "transport_recovery", "recovered", "recovered_shell"}
    assert (trace.reason == "completed") is completed
    assert (tmp_path / "report.txt").exists() is completed
    assert (str(tmp_path / filename) in {str(tmp_path / Path(p)) for p in loop.files_written}) is completed
    if completed:
        assert streams == (2 if case == "success" else 3)
    else:
        assert loop._completion_check
    if case in {"denied_code", "unknown_code"}:
        assert (loop._verification.last_write > 0) is (case == "unknown_code")


def test_recovery_requires_the_expected_revision(tmp_path):
    from rune.agent.file_outcomes import FileOutcomes

    path = tmp_path / "main.py"
    path.write_text("value = 1\n")
    params = {"path": "main.py", "search": "1", "replace": "2"}
    outcomes = FileOutcomes(str(tmp_path))
    expected = outcomes.expected_edit("file_edit", params)
    outcomes.observe("file_edit", params, CapabilityResult(success=False, error="Interrupted"), expected)
    assert outcomes.reconcile() == []
    path.write_text("value = 3\n")
    assert outcomes.reconcile() == []
    path.write_text("value = 2\n")
    assert outcomes.reconcile() == [str(path)]
    assert not outcomes.blocker()


def test_failed_delete_requires_absence_and_bad_paths_remain_blocked(tmp_path):
    from rune.agent.file_outcomes import FileOutcomes

    path = tmp_path / "report.txt"
    path.write_text("report")
    outcomes = FileOutcomes(str(tmp_path))
    outcomes.observe("file_delete", {"path": str(path)}, CapabilityResult(success=False, error="Interrupted"))
    assert not outcomes.reconcile()
    path.unlink()
    assert outcomes.reconcile() == [str(path)]
    outcomes.observe("file_write", {"path": "invalid\0path", "content": "report"},
                     CapabilityResult(success=False, error="Invalid path", metadata={"action_status": "not_executed"}))
    assert outcomes.blocker()
    assert not outcomes.reconcile()


def test_unreadable_delete_target_is_not_confirmed_absent(monkeypatch, tmp_path):
    from rune.agent.file_outcomes import FileOutcomes

    path = tmp_path / "report.txt"
    outcomes = FileOutcomes(str(tmp_path))
    outcomes.observe("file_delete", {"path": str(path)}, CapabilityResult(success=False, error="Permission denied"))

    def denied(self):
        raise PermissionError("Cannot inspect directory")

    monkeypatch.setattr(Path, "lstat", denied)
    assert not outcomes.reconcile()
    assert outcomes.blocker()
