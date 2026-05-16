"""Tests for goal classifier (LLM-only, no Tier1 regex)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from rune.agent.goal_classifier import VALID_GOAL_TYPES, ClassificationResult


class TestClassificationResult:
    def test_all_goal_types_valid(self):
        for gt in VALID_GOAL_TYPES:
            r = ClassificationResult(goal_type=gt, confidence=0.9, tier=2)
            assert r.goal_type == gt

    def test_default_fields(self):
        r = ClassificationResult(goal_type="chat", confidence=0.8, tier=2)
        assert r.is_continuation is False
        assert r.is_complex_coding is False
        assert r.complexity == "simple"
        assert r.output_expectation == "text"
        assert r.intent_categories == frozenset()

    def test_intent_categories_can_be_set(self):
        r = ClassificationResult(
            goal_type="full", confidence=0.9, tier=2,
            intent_categories=frozenset({"email", "document"}),
        )
        assert "email" in r.intent_categories
        assert "document" in r.intent_categories

    def test_valid_goal_types_set(self):
        assert "chat" in VALID_GOAL_TYPES
        assert "code_modify" in VALID_GOAL_TYPES
        assert "full" in VALID_GOAL_TYPES
        assert "invalid" not in VALID_GOAL_TYPES


class _FakeLLMClient:
    """Stub LLM client returning a fixed JSON verdict."""

    def __init__(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict

    async def completion(self, **kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": json.dumps(self.verdict)}}]}


def _patch_llm_client(monkeypatch: pytest.MonkeyPatch, verdict: dict[str, Any]) -> None:
    """Swap rune.llm.client.get_llm_client to return a stub."""
    import rune.llm.client as client_mod

    monkeypatch.setattr(
        client_mod, "get_llm_client", lambda: _FakeLLMClient(verdict),
    )


class TestIntentCategoryParsing:
    """LLM-emitted intent_categories survive into ClassificationResult."""

    async def test_email_intent_extracted_from_llm_response(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        _patch_llm_client(monkeypatch, {
            "goal_type": "full",
            "confidence": 0.9,
            "reason": "user wants email",
            "intent_categories": ["email"],
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("check my gmail inbox")
        assert result.intent_categories == frozenset({"email"})

    async def test_document_intent_japanese(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        _patch_llm_client(monkeypatch, {
            "goal_type": "full",
            "confidence": 0.9,
            "reason": "user wants document",
            "intent_categories": ["document"],
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("報告書を書いて")
        assert "document" in result.intent_categories

    async def test_email_intent_chinese(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        _patch_llm_client(monkeypatch, {
            "goal_type": "full",
            "confidence": 0.85,
            "reason": "email task",
            "intent_categories": ["email"],
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("帮我写邮件")
        assert "email" in result.intent_categories

    async def test_both_intents_at_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        _patch_llm_client(monkeypatch, {
            "goal_type": "full",
            "confidence": 0.9,
            "reason": "send the report by email",
            "intent_categories": ["email", "document"],
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("Send the project report to john@example.com")
        assert result.intent_categories == frozenset({"email", "document"})

    async def test_chat_no_intents(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        _patch_llm_client(monkeypatch, {
            "goal_type": "chat",
            "confidence": 0.95,
            "reason": "greeting",
            "intent_categories": [],
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("hi")
        assert result.intent_categories == frozenset()

    async def test_unknown_intent_filtered_out(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Hallucinated / unknown values must not poison ClassificationResult.
        _patch_llm_client(monkeypatch, {
            "goal_type": "full",
            "confidence": 0.9,
            "reason": "test",
            "intent_categories": ["email", "weather", "stocks"],
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("...")
        assert result.intent_categories == frozenset({"email"})

    async def test_non_list_intent_categories_ignored(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        _patch_llm_client(monkeypatch, {
            "goal_type": "full",
            "confidence": 0.9,
            "reason": "test",
            "intent_categories": "email",  # wrong shape
        })
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("...")
        assert result.intent_categories == frozenset()


class TestClassifyGoalFallback:
    """When LLM is unavailable, classify_goal should fallback gracefully."""

    @pytest.mark.asyncio
    async def test_fallback_returns_valid_type(self):
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("do something complex")
        assert result.goal_type in VALID_GOAL_TYPES
        assert result.tier == 2

    @pytest.mark.asyncio
    async def test_fallback_confidence_range(self):
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("some random input")
        assert 0.0 <= result.confidence <= 1.0

    async def test_fallback_includes_protective_intent_categories(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Force the LLM client to raise so we hit the except branch.
        import rune.llm.client as client_mod

        class _RaisingClient:
            async def completion(self, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("simulated provider down")

        monkeypatch.setattr(
            client_mod, "get_llm_client", lambda: _RaisingClient(),
        )
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("anything")
        # Protective default — both sections will be included downstream.
        assert "email" in result.intent_categories
        assert "document" in result.intent_categories
