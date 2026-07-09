"""Tests for the per-turn latency changes:

- loop.run() reuses a caller-supplied classification only on a single turn
  (no message_history), and re-classifies when history is present.
- _build_system_prompt is async (repo map scan offloaded off the event loop).
- the dead _classify_intent_llm fallback is gone.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.types import CompletionTrace


def _clf(goal_type: str = "chat") -> ClassificationResult:
    return ClassificationResult(goal_type=goal_type, confidence=0.9, tier=2)


async def _count_classify_calls(**run_kwargs) -> int:
    """Run loop.run() with generation stubbed, counting classify_goal calls."""
    loop = NativeAgentLoop()
    calls = {"n": 0}

    async def fake_classify(goal, *, previous_goal="", previous_goal_type=""):
        calls["n"] += 1
        return _clf()

    dummy = CompletionTrace(reason="test")
    with patch("rune.agent.loop.classify_goal", side_effect=fake_classify), \
         patch.object(NativeAgentLoop, "_execute_loop",
                      new=AsyncMock(return_value=dummy)):
        await loop.run("do something", **run_kwargs)
    return calls["n"]


class TestClassificationReuse:
    async def test_reuses_passed_classification_single_turn(self):
        # classification supplied + no history -> no redundant classify call.
        assert await _count_classify_calls(classification=_clf()) == 0

    async def test_reclassifies_multi_turn_even_when_passed(self):
        # History present -> must re-classify for domain-change detection,
        # regardless of a passed classification.
        calls = await _count_classify_calls(
            classification=_clf(),
            message_history=[{"role": "user", "content": "earlier turn"}],
        )
        assert calls == 1

    async def test_classifies_when_none_passed(self):
        assert await _count_classify_calls() == 1


class TestBuildSystemPromptAsync:
    def test_build_system_prompt_is_coroutine(self):
        assert inspect.iscoroutinefunction(NativeAgentLoop._build_system_prompt)

    async def test_build_system_prompt_returns_str(self):
        loop = NativeAgentLoop()
        prompt = await loop._build_system_prompt("hello", _clf("chat"), None)
        assert isinstance(prompt, str) and prompt


class TestDeadCodeRemoved:
    def test_classify_intent_llm_removed(self):
        assert not hasattr(NativeAgentLoop, "_classify_intent_llm")
