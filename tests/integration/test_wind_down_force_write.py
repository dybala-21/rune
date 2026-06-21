"""End-to-end proof that the budget wind-down forces a best-effort write.

Drives the real NativeAgentLoop.run() with a faked litellm stream. The token
budget is restored (via a checkpoint) already at 92%, so the first step starts
in the wind-down 'final' phase, and the loop must inject the write-now directive
(N1) before finalizing. This exercises the real loop wiring
(_update_wind_down_phase -> _maybe_force_wind_down_write), not just the helpers.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune.agent.checkpoint import CheckpointData
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.types import AgentConfig


async def _fake_completion(**_):
    async def _gen():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Here is the answer.", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, iterations=[]),
        )
    return _gen()


@pytest.mark.asyncio
async def test_wind_down_forces_write_through_full_loop(monkeypatch):
    config = AgentConfig(max_iterations=3, timeout_seconds=30,
                         model="test-model", provider="openai")
    config.token_budget_override = 20_000  # tiny budget
    loop = NativeAgentLoop(config=config)

    # Classify without a real LLM call.
    monkeypatch.setattr(
        "rune.agent.loop.classify_goal",
        AsyncMock(return_value=ClassificationResult(
            goal_type="research", confidence=0.9, tier=2, requires_execution=False)),
    )
    # Resume with the budget already at 92% so step 1 starts in the final phase.
    ckpt = CheckpointData(session_id="s", step=0, token_usage=18_400,
                          goal="explain something deeply")
    fake_mgr = MagicMock()
    fake_mgr.load.return_value = ckpt
    monkeypatch.setattr("rune.agent.loop.CheckpointManager", lambda: fake_mgr)

    with patch("litellm.acompletion", side_effect=_fake_completion):
        await loop.run("explain something deeply", resume_session_id="s")

    # N1: the loop injected the write-now directive at the final-phase step start.
    assert loop._wind_down_write_forced is True
