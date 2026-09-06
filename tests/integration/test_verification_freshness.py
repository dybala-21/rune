"""Exercise both completion paths with scripted tool events.

The model and external execution are stubbed; loop callbacks and gates run normally.
"""

from contextlib import asynccontextmanager

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.types import AgentConfig, CapabilityResult
from tests.integration.test_fast_lane_upshift import _FakeStream


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Done.", "The requested change is complete and ready for use."])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("mutation", ["file_write", "file_delete"])
async def test_stale_check_requires_another_verification(monkeypatch, tmp_path, answer, repair, mutation):
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
            goal_type="code_modify", confidence=.99, tier=2,
            requires_code=True, requires_execution=True, output_expectation="file",
        ),
    )
    assert rounds >= 2
    if repair:
        assert trace.reason == "completed"
        assert loop._verification.tests_passed_after_edit is True
    else:
        assert trace.reason != "completed"
        assert loop._verification.tests_passed_after_edit is False
