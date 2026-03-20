"""Tests for goal classifier (LLM-only, no Tier1 regex)."""

from __future__ import annotations

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

    def test_valid_goal_types_set(self):
        assert "chat" in VALID_GOAL_TYPES
        assert "code_modify" in VALID_GOAL_TYPES
        assert "full" in VALID_GOAL_TYPES
        assert "invalid" not in VALID_GOAL_TYPES


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
