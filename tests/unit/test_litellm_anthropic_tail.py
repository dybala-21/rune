"""Anthropic rejects a request whose final message is an assistant turn
("does not support assistant message prefill"). The wire history must end with
a user turn for Anthropic models; other providers are left unchanged."""

from __future__ import annotations

from rune.agent.litellm_adapter import _ensure_anthropic_user_tail


def test_anthropic_assistant_tail_gets_user_continuation():
    msgs = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "thinking..."},
    ]
    out = _ensure_anthropic_user_tail("anthropic/claude-opus-4-6", msgs)
    assert out[-1]["role"] == "user"
    assert len(out) == 3


def test_anthropic_user_tail_unchanged():
    msgs = [{"role": "user", "content": "do X"}]
    out = _ensure_anthropic_user_tail("anthropic/claude-opus-4-6", msgs)
    assert out == msgs


def test_anthropic_tool_tail_unchanged():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "1"},
    ]
    out = _ensure_anthropic_user_tail("claude-sonnet-4-5", msgs)
    assert out == msgs


def test_non_anthropic_assistant_tail_untouched():
    msgs = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "thinking..."},
    ]
    assert _ensure_anthropic_user_tail("openai/gpt-5.4", msgs) == msgs
    assert _ensure_anthropic_user_tail("gemini/gemini-2.5-pro", msgs) == msgs


def test_empty_messages_safe():
    assert _ensure_anthropic_user_tail("anthropic/claude-opus-4-6", []) == []
