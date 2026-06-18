"""Unit tests for executor-tier-based tool_rounds scaling.

Base budget: complex=24, simple=12 (raised so data-heavy multi-file tasks are not
truncated mid-turn; token budget + max_iterations remain the runaway backstops).
"""

from __future__ import annotations

from types import SimpleNamespace

from rune.agent.loop import _compute_tool_rounds


def _cls(complex_coding: bool) -> SimpleNamespace:
    return SimpleNamespace(is_complex_coding=complex_coding)


class TestStrongExecutor:
    def test_opus_complex_base(self):
        r = _compute_tool_rounds(_cls(True), "anthropic:claude-opus-4-6", advisor_enabled=False)
        assert r == 24

    def test_opus_simple_base(self):
        r = _compute_tool_rounds(_cls(False), "anthropic:claude-opus-4-6", advisor_enabled=False)
        assert r == 12

    def test_gpt4o_complex_no_bonus(self):
        r = _compute_tool_rounds(_cls(True), "openai:gpt-4o", advisor_enabled=False)
        assert r == 24


class TestMediumExecutor:
    def test_gemini_flash_complex(self):
        r = _compute_tool_rounds(_cls(True), "gemini:gemini-2.5-flash", advisor_enabled=False)
        # tier 45 < 50, bonus=2, base=24 → 26
        assert r == 26


class TestWeakExecutor:
    def test_gpt4o_mini_complex(self):
        r = _compute_tool_rounds(_cls(True), "openai:gpt-4o-mini", advisor_enabled=False)
        # tier 45, bonus=2, base=24 → 26
        assert r == 26

    def test_haiku_complex(self):
        r = _compute_tool_rounds(_cls(True), "anthropic:claude-haiku-4-5", advisor_enabled=False)
        # tier 40, bonus=2, base=24 → 26
        assert r == 26

    def test_small_ollama_complex(self):
        r = _compute_tool_rounds(_cls(True), "ollama:qwen2.5:7b", advisor_enabled=False)
        # tier 28, bonus=2, base=24 → 26
        assert r == 26


class TestAdvisorDiscount:
    def test_weak_with_advisor(self):
        r_off = _compute_tool_rounds(_cls(True), "openai:gpt-4o-mini", advisor_enabled=False)
        r_on = _compute_tool_rounds(_cls(True), "openai:gpt-4o-mini", advisor_enabled=True)
        # advisor discount -2 (26 → 24)
        assert r_on == r_off - 2

    def test_never_below_base(self):
        """advisor_discount should never push below base."""
        r = _compute_tool_rounds(_cls(False), "openai:gpt-4o", advisor_enabled=True)
        # base=12, tier bonus=0, advisor=-2 → would be 10, clamped to 12
        assert r == 12


class TestUnknownModel:
    def test_unknown_model_treated_as_medium(self):
        r = _compute_tool_rounds(_cls(True), "unknown:mystery-model", advisor_enabled=False)
        # UNKNOWN_TIER = 50, so bonus=1, base=24 → 25
        assert r == 25
