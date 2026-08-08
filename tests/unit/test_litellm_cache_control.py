"""Anthropic prompt caching: cache_control breakpoint on system message.

Adds cache_control: {type: "ephemeral"} to the system prompt for Anthropic
models, reducing fixed overhead (tools+system ~16K) to 0.1x cost from step 2.
"""

from __future__ import annotations

from rune.agent.litellm_adapter import (
    _apply_anthropic_cache_control,
    _apply_anthropic_message_cache,
)
from rune.agent.model_traits import traits


class TestAnthropicWireTrait:
    def test_claude_models(self):
        assert traits("claude-opus-4-6").anthropic_wire
        assert traits("anthropic/claude-sonnet-4-5").anthropic_wire
        assert traits("Claude-3-Haiku").anthropic_wire  # case insensitive

    def test_non_anthropic_models(self):
        assert not traits("gpt-5.4").anthropic_wire
        assert not traits("openai/gpt-5.4").anthropic_wire
        assert not traits("gemini/gemini-2.5-pro").anthropic_wire
        assert not traits("ollama/qwen2.5-coder:7b").anthropic_wire


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


class TestApplyAnthropicMessageCache:
    """Moving breakpoint on the last message caches the transcript prefix."""

    _M = "anthropic/claude-haiku-4-5"

    def test_tool_tail_gets_message_level_cache_control(self):
        msgs = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "content": "output"},
        ]
        out = _apply_anthropic_message_cache(self._M, msgs)
        assert out[-1]["cache_control"] == {"type": "ephemeral"}
        assert out[-1]["content"] == "output"
        # earlier messages untouched, original list not mutated
        assert out[:-1] == msgs[:-1]
        assert "cache_control" not in msgs[-1]

    def test_user_string_tail_converted_to_block(self):
        msgs = [{"role": "user", "content": "hello"}]
        out = _apply_anthropic_message_cache(self._M, msgs)
        block = out[-1]["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "hello"
        assert block["cache_control"] == {"type": "ephemeral"}
        assert isinstance(msgs[-1]["content"], str)  # original untouched

    def test_block_list_tail_marks_last_text_block(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ]},
        ]
        out = _apply_anthropic_message_cache(self._M, msgs)
        assert "cache_control" not in out[-1]["content"][0]
        assert out[-1]["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _apply_anthropic_message_cache("openai/gpt-5.4", msgs) == msgs
        assert isinstance(msgs[-1]["content"], str)

    def test_env_off_unchanged(self, monkeypatch):
        monkeypatch.setenv("RUNE_MSG_CACHE", "0")
        msgs = [{"role": "user", "content": "hello"}]
        out = _apply_anthropic_message_cache(self._M, msgs)
        assert out is msgs

    def test_empty_and_contentless_tails_safe(self):
        assert _apply_anthropic_message_cache(self._M, []) == []
        msgs = [{"role": "assistant", "content": None, "tool_calls": []}]
        assert _apply_anthropic_message_cache(self._M, msgs) == msgs
