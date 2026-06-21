"""Tests for the orchestrator worker factory (no real LLM).

The runner must apply the per-worker budget slice and return a WorkerResult
carrying the answer plus execution evidence (iterations and tool-call actions)
so the orchestrator's quality gate can see what the worker did.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rune.agent.worker_factory import make_worker_factory
from rune.types import AgentConfig


@pytest.mark.asyncio
async def test_runner_slices_budget_and_returns_answer(monkeypatch):
    captured = {}

    class _FakeLoop:
        def __init__(self, config=None):
            captured["override"] = getattr(config, "token_budget_override", None)
            self._last_answer_text = "WORKER OUTPUT"

        def on(self, event, handler):
            pass

        async def run(self, goal, context=None):
            captured["goal"] = goal
            return SimpleNamespace(reason="completed", final_step=4)

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", _FakeLoop)

    factory = make_worker_factory(
        AgentConfig(model="m", provider="openai"), total_budget=200_000, n_workers=4)
    runner = await factory("executor")
    out = await runner("do subtask X")

    assert out.answer == "WORKER OUTPUT"
    assert out.iterations == 4
    assert captured["goal"] == "do subtask X"
    assert captured["override"] == 50_000  # 200k / 4 workers


@pytest.mark.asyncio
async def test_runner_counts_tool_calls_as_actions(monkeypatch):
    class _FakeLoop:
        def __init__(self, config=None):
            self._last_answer_text = "done"
            self._handler = None

        def on(self, event, handler):
            if event == "tool_call":
                self._handler = handler

        async def run(self, goal, context=None):
            # Two tool calls during the run.
            await self._handler({"name": "file_read"})
            await self._handler({"name": "bash_execute"})
            return SimpleNamespace(reason="completed", final_step=2)

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", _FakeLoop)
    factory = make_worker_factory(
        AgentConfig(model="m", provider="openai"), total_budget=200_000, n_workers=2)
    runner = await factory("executor")
    out = await runner("subtask")
    assert out.actions == 2


@pytest.mark.asyncio
async def test_runner_reports_when_no_answer(monkeypatch):
    class _FakeLoop:
        def __init__(self, config=None):
            self._last_answer_text = ""

        def on(self, event, handler):
            pass

        async def run(self, goal, context=None):
            return SimpleNamespace(reason="max_gate_blocked", final_step=0)

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", _FakeLoop)
    factory = make_worker_factory(
        AgentConfig(model="m", provider="openai"), total_budget=200_000, n_workers=2)
    runner = await factory("executor")
    out = await runner("subtask")
    assert "no answer" in out.answer and "max_gate_blocked" in out.answer


@pytest.mark.asyncio
async def test_worker_budget_floor(monkeypatch):
    seen = {}

    class _FakeLoop:
        def __init__(self, config=None):
            seen["override"] = getattr(config, "token_budget_override", None)
            self._last_answer_text = "x"

        def on(self, event, handler):
            pass

        async def run(self, goal, context=None):
            return SimpleNamespace(reason="completed", final_step=1)

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", _FakeLoop)
    # 100k / 8 = 12.5k, below the 40k floor, so it is floored.
    factory = make_worker_factory(
        AgentConfig(model="m", provider="openai"), total_budget=100_000, n_workers=8)
    runner = await factory("executor")
    await runner("t")
    assert seen["override"] == 40_000
