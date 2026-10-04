"""Background work shares chat verification, budgets and approval boundaries."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rune.agent.background import BackgroundTask, run_background
from rune.agent.run_control import current_control
from rune.safety.approval_context import approval_granted, was_approved
from rune.types import CompletionTrace


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_WORKSPACE", str(tmp_path))


class Loop:
    def __init__(self, trace, **kwargs):
        self.trace = trace
        self._last_answer_text = "Result from the real output field"
        self._last_run_timings = {"usage": {"calls": 1, "unpriced_calls": 1}}

    def set_approval_callback(self, callback):
        self.approve = callback

    async def run(self, goal, **kwargs):
        assert not was_approved()
        assert current_control() is not None
        assert kwargs["max_steps"] == 30
        assert kwargs["context"]["workspace_root"]
        return self.trace


@pytest.mark.asyncio
@pytest.mark.parametrize("verification,status,verified", [
    (None, "completed", False),
    ({"required": False, "status": "inconclusive"}, "unverified", False),
    ({"required": True, "status": "fail"}, "failed", False),
    ({"required": True, "status": "pass"}, "verified", True),
])
async def test_completion_is_not_verification(verification, status, verified):
    loop = Loop(CompletionTrace(reason="completed", verification=verification))
    with approval_granted():
        result = await run_background(BackgroundTask("Report", "test"), loop_factory=lambda **kw: loop)
        assert was_approved()
    assert result["status"] == status
    assert result["verified"] is verified
    assert result["output"] == loop._last_answer_text
    assert result["timings"]["usage"]["unpriced_calls"] == 1
    assert current_control() is None


@pytest.mark.asyncio
async def test_background_approval_and_timeout_are_not_success():
    loop = Loop(CompletionTrace(reason="completed"))

    async def asks(*args, **kwargs):
        assert await loop.approve("send_mail", "Send message") is False
        return loop.trace

    loop.run = asks
    result = await run_background(BackgroundTask("Report", "test"), loop_factory=lambda **kw: loop)
    assert result["status"] == "needs_approval"
    assert not result["verified"]

    loop.run = AsyncMock(side_effect=lambda *args, **kwargs: None)

    async def slow(*args, **kwargs):
        await asyncio.sleep(1)

    loop.run = slow
    result = await run_background(BackgroundTask("Report", "test", timeout_seconds=.01), loop_factory=lambda **kw: loop)
    assert result["execution_unknown"]
    assert not result["verified"]


@pytest.mark.asyncio
async def test_verification_does_not_inherit_approval(monkeypatch):
    from rune.agent.goal_validate import _default_exec
    from rune.safety.verification import CheckResult

    async def execute(command, cwd, timeout, limit):
        assert not was_approved()
        return CheckResult(error="approval required")

    monkeypatch.setattr("rune.safety.verification._run_check", execute)
    with approval_granted():
        code, output = await _default_exec("echo check", "", 1)
        assert was_approved()
    assert code != 0
    assert output == "approval required"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,ask", [("max_iterations", False), ("completed", True)])
async def test_process_worker_cannot_invent_approval_or_completion(tmp_path, monkeypatch, reason, ask):
    from rune.agent.worker_proc import _run

    class Worker(Loop):
        def __init__(self, config):
            super().__init__(CompletionTrace(reason=reason))

        def on(self, *args):
            return None

        async def run(self, goal, **kwargs):
            assert not was_approved()
            assert kwargs["context"]["workspace_root"] == str(tmp_path)
            assert kwargs["max_steps"] == 3
            if ask:
                assert not await self.approve("send_mail", "Send message")
            return self.trace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNE_WORKER", "1")
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", Worker)
    with approval_granted():
        result = await _run({"root": str(tmp_path), "goal": "Report", "max_iterations": 3})
    assert not result["ok"]
    assert not result["verified"]


def test_denied_worker_does_not_spend_more_tokens_on_escalation():
    from rune.agent.parallel_isolated import Escalation, WorkerOutcome, WorkerSpec, _escalate_spec
    outcome = WorkerOutcome(worker_id="w", result={"blocked_tools": ["send_mail"]})
    assert _escalate_spec(WorkerSpec("w", "Report"), outcome, 1, Escalation(provider="openai")) is None


@pytest.mark.asyncio
async def test_replay_workspace_scopes_do_not_affect_other_runs(tmp_path, monkeypatch):
    from rune.agent.isolation import enforce, isolation_root, isolation_scope

    monkeypatch.delenv("RUNE_ISOLATION_ROOT", raising=False)
    async def check(root):
        with isolation_scope(str(root)):
            await asyncio.sleep(0)
            assert isolation_root() == str(root)
            assert enforce(str(tmp_path / "outside"))
            assert enforce(str(root / "local")) is None
            with pytest.raises(ValueError):
                with isolation_scope(str(tmp_path)):
                    pytest.fail("Expanded isolation")

    await asyncio.gather(check(tmp_path / "one"), check(tmp_path / "two"))
    assert isolation_root() is None


@pytest.mark.parametrize("trace,success,verified", [
    (CompletionTrace(reason="completed"), True, False),
    (CompletionTrace(reason="verified", mech_check="pass"), True, True),
    (CompletionTrace(reason="completed", tool_budget_exhausted=True), False, False),
    (CompletionTrace(reason="completed", verification={"required": True, "status": "unverified"}), False, False),
    (CompletionTrace(reason="completed", verification={"required": True, "status": "fail"}), False, False),
    (CompletionTrace(reason="cancelled", mech_check="pass"), False, False),
    (CompletionTrace(reason="max_iterations"), False, False),
    (CompletionTrace(), False, False),
    (CompletionTrace(reason="completed", mech_check="pass",
                     requirement_acceptance={"required": True, "status": "inconclusive"}), False, False),
    (CompletionTrace(reason="completed", mech_check="pass",
                     requirement_acceptance={"required": True, "status": "fail"}), False, False),
    (CompletionTrace(reason="completed", mech_check="pass",
                     requirement_acceptance={"required": True, "status": "pass"}), True, True),
    (CompletionTrace(reason="completed",
                     requirement_acceptance={"required": True, "status": "pass"}), True, False),
])
async def test_daemon_and_chat_share_completion_verdicts(monkeypatch, trace, success, verified):
    from rune.agent.run_outcome import run_outcome
    from rune.api.trust import build_trust_payload
    from rune.daemon.main import RuneDaemon

    loop = Loop(trace)
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", lambda **kw: loop)
    result = await RuneDaemon._execute_agent(None, {"goal": "Report"})
    assert result["success"] is success
    assert result["verified"] is verified
    assert result["answer"] == loop._last_answer_text
    outcome = run_outcome(trace)
    assert result["outcome"] == outcome.payload()
    assert all(build_trust_payload(trace)[key] == value for key, value in outcome.payload().items())


async def test_background_keeps_workspace_and_backend_through_validation(tmp_path, monkeypatch):
    from rune.config.schema import SandboxConfig
    from rune.safety.execution_environment import (
        environment_scope,
        execution_config,
        execution_workspace,
    )

    workspaces = [tmp_path / "one", tmp_path / "two"]
    for workspace in workspaces:
        workspace.mkdir()
    seen = []

    def validator(*, cwd, timeout_s, auto_root):
        assert not auto_root
        assert execution_workspace() == cwd
        async def check(commands):
            await asyncio.sleep(0)
            assert execution_workspace() == cwd
            assert execution_config().image == Path(cwd).name
            assert not was_approved()
            seen.append(cwd)
            return True, "check passed"
        return check

    monkeypatch.setattr("rune.agent.goal_validate.make_validate_fn", validator)

    async def run(workspace):
        loop = Loop(CompletionTrace(reason="completed"))
        async def execute(*args, **kwargs):
            assert execution_workspace() == str(workspace)
            await asyncio.sleep(0)
            return loop.trace
        loop.run = execute
        with environment_scope(str(workspace), SandboxConfig(backend="container", image=workspace.name)):
            result = await run_background(BackgroundTask("Report", "test", workspace=str(workspace), verification=["check"]),
                                          loop_factory=lambda **kw: loop)
            assert result["verified"]

    await asyncio.gather(*(run(path) for path in workspaces))
    assert set(seen) == {str(p) for p in workspaces}


async def test_background_validation_does_not_follow_a_midrun_config_change(tmp_path, monkeypatch):
    from rune.config import get_config
    from rune.config.schema import SandboxConfig
    from rune.safety.execution_environment import execution_config, execution_workspace

    config = get_config()
    monkeypatch.setattr(config.safety, "sandbox", SandboxConfig(backend="container", image="original"))
    loop = Loop(CompletionTrace(reason="completed"))

    async def execute(*args, **kwargs):
        monkeypatch.setattr(config.safety, "sandbox", SandboxConfig(backend="local"))
        return loop.trace

    def validator(**kwargs):
        async def check(commands):
            assert execution_config().backend == "container"
            assert execution_config().image == "original"
            assert execution_workspace() == str(tmp_path)
            return True, "Verified in original environment"
        return check

    loop.run = execute
    monkeypatch.setattr("rune.agent.goal_validate.make_validate_fn", validator)
    result = await run_background(BackgroundTask("Report", "test", workspace=str(tmp_path), verification=["check"]),
                                  loop_factory=lambda **kw: loop)
    assert result["verified"]
    assert execution_config().backend == "local"


async def test_noninteractive_api_does_not_grant_approval(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rune.api.handlers.agent import AgentRunRequest, _execute_agent
    from rune.api.run_tracker import RunTracker

    class ApiLoop(Loop):
        def __init__(self):
            super().__init__(CompletionTrace(reason="completed"))

        def on(self, *args):
            pass

        def set_ask_user_callback(self, callback):
            pass

        async def run(self, *args, **kwargs):
            assert not was_approved()
            assert current_control().run_id == "api-test"
            assert not await self.approve("send_mail", "Send a message")
            return self.trace

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", ApiLoop)
    monkeypatch.setattr("rune.agent.agent_context.prepare_agent_context", AsyncMock(return_value=SimpleNamespace(
        workspace_root=str(tmp_path), goal="Report", messages=[],
    )))
    learn = AsyncMock()
    monkeypatch.setattr("rune.agent.agent_context.post_process_agent_result", learn)
    tracker = RunTracker()
    tracker.create("api-test", "api", "", "Report")
    with approval_granted():
        await _execute_agent(tracker, "api-test", AgentRunRequest(goal="Report"))
    result = tracker.get("api-test").result
    assert not result.success
    assert result.outcome["reason"] == "approval_required"
    assert not learn.call_args.args[0].success


async def test_api_cancel_stops_the_execution_task(monkeypatch):
    from rune.api.handlers import agent
    from rune.api.run_tracker import RunTracker

    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def execute(tracker, run_id, *args):
        tracker.mark_running(run_id)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    tracker = RunTracker()
    monkeypatch.setattr(agent, "_tracker", tracker)
    monkeypatch.setattr(agent, "_execute_agent", execute)
    response = await agent.agent_run(agent.AgentRunRequest(goal="Report"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await agent.agent_cancel(response.run_id)
        await asyncio.wait_for(stopped.wait(), timeout=1)
        assert tracker.get(response.run_id).status == "aborted"
    finally:
        for task in tuple(agent._background_tasks):
            if task.get_name() == f"api-run:{response.run_id}":
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
