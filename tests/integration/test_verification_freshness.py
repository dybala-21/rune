"""Completion gates with stubbed model responses and tool execution."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.types import AgentConfig, CapabilityResult
from tests.integration.test_fast_lane_upshift import _FakeStream


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["37개 · 점검 완료.", "The review is complete. The document describes two independent issues and their causes."])
@pytest.mark.parametrize("goal_type,tool", [("full", "file_read"), ("web", "web_fetch")])
async def test_read_only_full_task_never_receives_a_mutation_nudge(monkeypatch, tmp_path, answer, goal_type, tool):
    for key, value in {"RUNE_HOME": str(tmp_path), "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0",
                       "RUNE_ADVISOR": "0", "RUNE_IN_BEST_OF": "1"}.items():
        monkeypatch.setenv(key, value)
    options = None
    injected = []

    def build_tools(opts):
        nonlocal options
        options = opts
        return {}

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            await options.on_tool_start(tool, {"path": "report.txt", "url": "https://example.org/status"})
            await options.on_tool_end(tool, CapabilityResult(success=True, output="37개 · 점검 완료."))
            yield _FakeStream(answer)

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop(AgentConfig(model="test", max_iterations=1))
    loop._token_budget.total = 300_000
    original = loop._inject_system_message
    monkeypatch.setattr(loop, "_inject_system_message", lambda messages, text: (injected.append(text), original(messages, text))[1])
    trace = await loop._execute_loop(goal="Review the report and explain the causes", system_prompt="test", tools=[], max_iterations=1,
        classification=ClassificationResult(goal_type=goal_type, confidence=.99, tier=2, output_expectation="text"))
    assert not any("edit/write/execute" in text or "not made any changes" in text or "requested action has no execution" in text for text in injected)
    assert trace.reason == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("goal_type", ["code_modify", "execution", "full"])
@pytest.mark.parametrize("read_source", [False, True])
async def test_action_task_cannot_finish_with_only_a_claim(monkeypatch, tmp_path, goal_type, read_source):
    for key, value in {"RUNE_HOME": str(tmp_path), "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0",
                       "RUNE_ADVISOR": "0", "RUNE_IN_BEST_OF": "1"}.items():
        monkeypatch.setenv(key, value)
    options = None
    rounds = 0

    def build_tools(opts):
        nonlocal options
        options = opts
        return {}

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            nonlocal rounds
            rounds += 1
            if read_source:
                await options.on_tool_start("file_read", {"path": "source.txt"})
                await options.on_tool_end("file_read", CapabilityResult(success=True, output="source"))
            yield _FakeStream("Done. The requested action has been completed successfully.")

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop(AgentConfig(model="test", max_iterations=4))
    loop._token_budget.total = 300_000
    trace = await loop._execute_loop(goal="Perform the requested action", system_prompt="test", tools=[], max_iterations=4,
        classification=ClassificationResult(goal_type=goal_type, confidence=.99, tier=2,
            output_expectation="file" if goal_type == "full" else "text"))
    assert trace.reason not in {"completed", "verified"}
    assert loop._completion_check and rounds <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("output,passed", [("3 passed", True), ("no tests ran", False), ("1 failed, 2 passed", False)])
async def test_predictions_receive_actual_verification_results(monkeypatch, tmp_path, output, passed):
    from rune.proactive.prediction.engine import PredictionEngine

    for key, value in {
        "RUNE_HOME": str(tmp_path / "rune"), "RUNE_IN_BEST_OF": "",
        "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0", "RUNE_ADVISOR": "0",
    }.items():
        monkeypatch.setenv(key, value)
    engine = PredictionEngine()
    monkeypatch.setattr("rune.proactive.prediction.engine.get_prediction_engine", lambda: engine)
    monkeypatch.setattr("rune.memory.store.get_memory_store", lambda: SimpleNamespace(log_tool_call=lambda *a, **kw: None))
    options = None

    def build_tools(opts):
        nonlocal options
        options = opts
        return {}

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            for i in range(4):
                await options.on_tool_start("file_edit", {"path": "main.py", "content": f"x = {i}"})
                await options.on_tool_end("file_edit", CapabilityResult(success=True, output="edited"))
            await options.on_tool_start("bash_execute", {"command": "pytest -q", "cwd": str(tmp_path)})
            await options.on_tool_end("bash_execute", CapabilityResult(success=passed, output=output))
            yield _FakeStream("The test run has finished; see its recorded result.")

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop(AgentConfig(model="test", max_iterations=1))
    loop._session_id = "prediction-test"
    loop._token_budget.total = 300_000
    await loop._execute_loop(
        goal="Fix main.py", system_prompt="test", tools=[], max_iterations=1,
        classification=ClassificationResult(goal_type="code_modify", confidence=.99, tier=2, requires_code=True),
    )
    actions = engine._recent_actions
    assert len(actions) == 5
    assert all(action["code_changed"] for action in actions[:4])
    assert actions[-1]["tests_passed"] is passed
    assert actions[-1]["session_id"] == "prediction-test"
    assert (engine.need_inferer._check_testing_need(actions) is None) is passed


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Done.", "The requested change is complete and ready for use."])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("mutation", ["file_write", "file_delete"])
@pytest.mark.parametrize("goal_type,requires_code", [("code_modify", True), ("full", False)])
async def test_stale_check_requires_another_verification(monkeypatch, tmp_path, answer, repair, mutation, goal_type, requires_code):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "rune"))
    monkeypatch.setenv("RUNE_IN_BEST_OF", "1")
    monkeypatch.setenv("RUNE_REQUIRE_TEST_PASS", "1")
    monkeypatch.setenv("RUNE_AUTO_VERIFY", "0")
    monkeypatch.setenv("RUNE_VERIFY_FRESHNESS", "0")
    monkeypatch.setenv("RUNE_REQUIREMENT_GATE", "0")
    monkeypatch.setenv("RUNE_ADVISOR", "0")
    options = None
    rounds = 0

    def build_tools(opts):
        nonlocal options
        options = opts
        return {}

    async def event(name, params, output):
        await options.on_tool_start(name, params)
        await options.on_tool_end(name, CapabilityResult(success=True, output=output))

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                await event("file_read", {"path": "main.py"}, "x = 1")
                await event("bash_execute", {"command": "pytest -q"}, "3 passed")
                await event(mutation, {"path": "main.py", "content": "x = 2"}, "changed")
                await event("bash_execute", {"command": "echo done"}, "done")
            elif repair and rounds == 2:
                await event("bash_execute", {"command": "pytest -q"}, "3 passed")
            yield _FakeStream(answer)

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop(AgentConfig(model="test", max_iterations=6))
    loop._token_budget.total = 300_000
    loop._requires_execution = True
    trace = await loop._execute_loop(
        goal="Fix main.py", system_prompt="test", tools=[], max_iterations=6,
        classification=ClassificationResult(
            goal_type=goal_type, confidence=.99, tier=2,
            requires_code=requires_code, requires_execution=True, output_expectation="file",
        ),
    )
    assert rounds >= 2
    if repair:
        assert trace.reason == "completed"
        assert loop._verification.tests_passed_after_edit is True
    else:
        assert trace.reason != "completed"
        assert loop._verification.tests_passed_after_edit is False


@pytest.mark.parametrize("later", ["ruff check .", "pytest tests/unit/test_one.py", "pytest || true"])
async def test_failed_suite_blocks_both_completion_paths(monkeypatch, tmp_path, later):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "rune"))
    monkeypatch.setenv("RUNE_IN_BEST_OF", "1")
    monkeypatch.setenv("RUNE_REQUIRE_TEST_PASS", "1")
    monkeypatch.setenv("RUNE_AUTO_VERIFY", "0")
    monkeypatch.setenv("RUNE_REQUIREMENT_GATE", "0")
    monkeypatch.setenv("RUNE_ADVISOR", "0")
    options = None
    rounds = 0

    def build_tools(opts):
        nonlocal options
        options = opts
        return {}

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                for name, params, success, output in [
                    ("file_write", {"path": "main.py", "content": "x = 2"}, True, "written"),
                    ("bash_execute", {"command": "pytest"}, False, "1 failed, 3 passed"),
                    ("bash_execute", {"command": later}, True, "1 passed"),
                ]:
                    await options.on_tool_start(name, params)
                    await options.on_tool_end(name, CapabilityResult(success=success, output=output))
            yield _FakeStream("Done." if rounds % 2 else "The requested change is complete and ready.")

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop(AgentConfig(model="test", max_iterations=6))
    loop._token_budget.total = 300_000
    loop._requires_execution = True
    trace = await loop._execute_loop(
        goal="Fix main.py", system_prompt="test", tools=[], max_iterations=6,
        classification=ClassificationResult(
            goal_type="code_modify", confidence=.99, tier=2,
            requires_code=True, requires_execution=True, output_expectation="file",
        ),
    )
    assert trace.reason not in {"completed", "verified"}
    assert loop._verification.pending


async def test_data_file_delivery_does_not_require_code_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "rune"))
    monkeypatch.setenv("RUNE_IN_BEST_OF", "1")
    monkeypatch.setenv("RUNE_REQUIRE_TEST_PASS", "1")
    monkeypatch.setenv("RUNE_AUTO_VERIFY", "0")
    monkeypatch.setenv("RUNE_REQUIREMENT_GATE", "0")
    monkeypatch.setenv("RUNE_ADVISOR", "0")
    options = None
    rounds = 0

    def build_tools(opts):
        nonlocal options
        options = opts
        return {}

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            nonlocal rounds
            rounds += 1
            await options.on_tool_start("file_write", {"path": "totals.csv", "content": "team,total\nA,100\n"})
            await options.on_tool_end("file_write", CapabilityResult(success=True, output="written"))
            yield _FakeStream("Saved the requested CSV with one team and its total of 100.")

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop(AgentConfig(model="test", max_iterations=3))
    loop._token_budget.total = 300_000
    trace = await loop._execute_loop(
        goal="Save a CSV", system_prompt="test", tools=[], max_iterations=3,
        classification=ClassificationResult(goal_type="full", confidence=.99, tier=2),
    )
    assert rounds == 1 and trace.reason == "completed"
    assert loop._verification.last_write == 0
    assert loop._verification.tests_passed_after_edit is None
