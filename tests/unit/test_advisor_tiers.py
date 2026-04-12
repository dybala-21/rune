"""Unit tests for advisor tier map and pairing validation."""

from __future__ import annotations

from rune.agent.advisor.tiers import (
    MIN_TIER_GAP,
    UNKNOWN_TIER,
    check_pairing,
    extract_provider_and_model,
    resolve_tier,
)


class TestResolveTier:
    def test_anthropic_opus(self):
        assert resolve_tier("anthropic", "claude-opus-4-6") == 100
        assert resolve_tier("anthropic", "claude-opus-4-1") == 100

    def test_anthropic_sonnet(self):
        assert resolve_tier("anthropic", "claude-sonnet-4-5-20250929") == 70

    def test_anthropic_haiku(self):
        assert resolve_tier("anthropic", "claude-haiku-4-5-20251001") == 40

    def test_openai_o1(self):
        assert resolve_tier("openai", "o1") == 95
        assert resolve_tier("openai", "o1-preview") == 95

    def test_openai_gpt_variants(self):
        assert resolve_tier("openai", "gpt-5.4") == 92
        assert resolve_tier("openai", "gpt-4o") == 75
        assert resolve_tier("openai", "gpt-4o-mini") == 45

    def test_gemini(self):
        assert resolve_tier("gemini", "gemini-2.5-pro") == 85
        assert resolve_tier("gemini", "gemini-2.5-flash") == 45

    def test_deepseek(self):
        assert resolve_tier("deepseek", "deepseek-reasoner") == 90
        assert resolve_tier("deepseek", "deepseek-chat") == 55

    def test_ollama_local(self):
        assert resolve_tier("ollama", "qwen2.5:72b") == 70
        assert resolve_tier("ollama", "llama3.1:8b") == 30

    def test_unknown_falls_back(self):
        assert resolve_tier("anthropic", "unknown-future-model") == UNKNOWN_TIER
        assert resolve_tier("no-such-provider", "anything") == UNKNOWN_TIER
        assert resolve_tier("", "") == UNKNOWN_TIER

    def test_longest_prefix_wins(self):
        # "gpt-5.4" should win over "gpt-5" for a model "gpt-5.4".
        # Both match the startswith, but the longer prefix should take
        # precedence so mini is not treated as gpt-5 or gpt-5.4.
        assert resolve_tier("openai", "gpt-5.4") == 92
        assert resolve_tier("openai", "gpt-5-mini") == 50
        assert resolve_tier("openai", "gpt-5-pro-something") == 90


class TestCheckPairing:
    def test_sonnet_opus_ok(self):
        r = check_pairing(
            "anthropic", "claude-sonnet-4-5-20250929",
            "anthropic", "claude-opus-4-6",
        )
        assert r.ok is True
        assert r.executor_tier == 70
        assert r.advisor_tier == 100

    def test_haiku_opus_ok(self):
        r = check_pairing(
            "anthropic", "claude-haiku-4-5-20251001",
            "anthropic", "claude-opus-4-6",
        )
        assert r.ok is True

    def test_same_tier_fails(self):
        r = check_pairing(
            "anthropic", "claude-opus-4-6",
            "anthropic", "claude-opus-4-6",
        )
        assert r.ok is False
        assert "tier" in r.reason

    def test_weaker_advisor_fails(self):
        r = check_pairing(
            "anthropic", "claude-sonnet-4-5-20250929",
            "anthropic", "claude-haiku-4-5-20251001",
        )
        assert r.ok is False

    def test_cross_provider_opus_advisor(self):
        # Hybrid: local executor + cloud advisor
        r = check_pairing(
            "ollama", "qwen2.5:7b",
            "anthropic", "claude-opus-4-6",
        )
        assert r.ok is True
        assert r.advisor_tier - r.executor_tier >= MIN_TIER_GAP

    def test_deepseek_pair(self):
        r = check_pairing(
            "deepseek", "deepseek-chat",
            "deepseek", "deepseek-reasoner",
        )
        assert r.ok is True

    def test_gemini_pair(self):
        r = check_pairing(
            "gemini", "gemini-2.5-flash",
            "gemini", "gemini-2.5-pro",
        )
        assert r.ok is True

    def test_openai_o1_advisor(self):
        r = check_pairing(
            "openai", "gpt-4o-mini",
            "openai", "o1",
        )
        assert r.ok is True


class TestExtractProviderAndModel:
    def test_slash_format(self):
        assert extract_provider_and_model("anthropic/claude-opus-4-6") == (
            "anthropic", "claude-opus-4-6",
        )

    def test_colon_format(self):
        assert extract_provider_and_model("anthropic:claude-opus-4-6") == (
            "anthropic", "claude-opus-4-6",
        )

    def test_bare_gpt(self):
        assert extract_provider_and_model("gpt-4o-mini") == ("openai", "gpt-4o-mini")

    def test_bare_claude(self):
        assert extract_provider_and_model("claude-opus-4-6") == (
            "anthropic", "claude-opus-4-6",
        )

    def test_bare_gemini(self):
        assert extract_provider_and_model("gemini-2.5-pro") == (
            "gemini", "gemini-2.5-pro",
        )

    def test_bare_grok(self):
        assert extract_provider_and_model("grok-3") == ("xai", "grok-3")

    def test_bare_deepseek(self):
        assert extract_provider_and_model("deepseek-reasoner") == (
            "deepseek", "deepseek-reasoner",
        )

    def test_empty(self):
        assert extract_provider_and_model("") == ("", "")
