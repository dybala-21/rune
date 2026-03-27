"""Tests for provider_capabilities — prompt supplements per provider."""

from __future__ import annotations

from rune.agent.provider_capabilities import _extract_provider, get_prompt_supplement


class TestExtractProvider:

    def test_anthropic_with_prefix(self):
        assert _extract_provider("anthropic/claude-sonnet-4-6") == "anthropic"

    def test_openai_no_prefix(self):
        assert _extract_provider("gpt-4o") == "openai"

    def test_openai_mini(self):
        assert _extract_provider("gpt-4o-mini") == "openai"

    def test_gemini_with_prefix(self):
        assert _extract_provider("gemini/gemini-2.5-pro") == "gemini"

    def test_gemini_no_prefix(self):
        assert _extract_provider("gemini-2.5-pro") == "gemini"

    def test_claude_no_prefix(self):
        assert _extract_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_grok(self):
        assert _extract_provider("grok-3") == "xai"

    def test_ollama_with_prefix(self):
        assert _extract_provider("ollama/llama3") == "ollama"

    def test_unknown_returns_empty(self):
        assert _extract_provider("some-random-model") == ""


class TestGetPromptSupplement:

    def test_claude_no_supplement(self):
        assert get_prompt_supplement("anthropic/claude-sonnet-4-6") == ""
        assert get_prompt_supplement("claude-haiku-4-5-20251001") == ""

    def test_openai_has_supplement(self):
        s = get_prompt_supplement("gpt-4o")
        assert len(s) > 0
        assert "tool" in s.lower() or "edit" in s.lower()

    def test_openai_mini_has_supplement(self):
        s = get_prompt_supplement("gpt-4o-mini")
        assert len(s) > 0

    def test_gemini_has_supplement(self):
        s = get_prompt_supplement("gemini-2.5-pro")
        assert len(s) > 0

    def test_xai_no_supplement(self):
        assert get_prompt_supplement("grok-3") == ""

    def test_ollama_no_supplement(self):
        assert get_prompt_supplement("ollama/llama3") == ""

    def test_unknown_no_supplement(self):
        assert get_prompt_supplement("random-model") == ""
