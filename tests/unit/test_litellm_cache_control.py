"""Anthropic prompt caching: cache_control breakpoint on system message.

Adds cache_control: {type: "ephemeral"} to the system prompt for Anthropic
models, reducing fixed overhead (tools+system ~16K) to 0.1x cost from step 2.
"""

from __future__ import annotations

from rune.agent.litellm_adapter import (
    _apply_anthropic_cache_control,
    _is_anthropic_model,
)


class TestIsAnthropicModel:
    def test_claude_models(self):
        assert _is_anthropic_model("claude-opus-4-6")
        assert _is_anthropic_model("anthropic/claude-sonnet-4-5")
        assert _is_anthropic_model("Claude-3-Haiku")  # case insensitive

    def test_non_anthropic_models(self):
        assert not _is_anthropic_model("gpt-5.4")
        assert not _is_anthropic_model("openai/gpt-5.4")
        assert not _is_anthropic_model("gemini/gemini-2.5-pro")
        assert not _is_anthropic_model("ollama/qwen2.5-coder:7b")


class TestApplyAnthropicCacheControl:
    def test_string_system_gets_cache_control(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        out = _apply_anthropic_cache_control("anthropic/claude-opus-4-6", msgs)

        # System content converted to block list
        assert out[0]["role"] == "system"
        content = out[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "You are a helpful assistant."
        assert content[0]["cache_control"] == {"type": "ephemeral"}

        # Rest of messages unchanged
        assert out[1] == msgs[1]

    def test_block_list_system_gets_cache_control_on_last_text(self):
        msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        out = _apply_anthropic_cache_control("claude-sonnet-4-5", msgs)

        content = out[0]["content"]
        # First block unchanged
        assert "cache_control" not in content[0]
        # Last text block has cache_control
        assert content[1]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_unchanged(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        out = _apply_anthropic_cache_control("openai/gpt-5.4", msgs)
        assert out == msgs
        # Content still a string, not converted
        assert isinstance(out[0]["content"], str)

    def test_no_system_message_unchanged(self):
        msgs = [{"role": "user", "content": "Hello"}]
        out = _apply_anthropic_cache_control("anthropic/claude-opus-4-6", msgs)
        assert out == msgs

    def test_empty_messages_safe(self):
        assert _apply_anthropic_cache_control("anthropic/claude-opus-4-6", []) == []

    def test_empty_system_content_unchanged(self):
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]
        out = _apply_anthropic_cache_control("anthropic/claude-opus-4-6", msgs)
        assert out == msgs  # Empty content not converted
