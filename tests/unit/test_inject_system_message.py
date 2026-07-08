"""Test that _inject_system_message uses user role with <system-reminder> tags.

Previously used role="system" which litellm_adapter skipped, so nudges like
wind-down warnings and stall guidance never reached the model.
"""

from __future__ import annotations

from rune.agent.loop import NativeAgentLoop


def test_inject_system_message_uses_user_role():
    """Nudges must use role=user to reach the model (adapter skips system)."""
    messages: list = [{"role": "user", "content": "hello"}]
    result = NativeAgentLoop._inject_system_message(messages, "Budget warning: 5 steps remaining")

    assert len(result) == 2
    injected = result[-1]
    assert injected["role"] == "user"  # NOT "system"
    assert "<system-reminder>" in injected["content"]
    assert "Budget warning" in injected["content"]


def test_inject_system_message_preserves_original():
    """Original messages list should not be mutated."""
    original = [{"role": "user", "content": "hello"}]
    result = NativeAgentLoop._inject_system_message(original, "test")

    assert len(original) == 1  # unchanged
    assert len(result) == 2


def test_inject_system_message_wraps_with_tags():
    """Content should be wrapped in <system-reminder> tags."""
    messages: list = []
    result = NativeAgentLoop._inject_system_message(messages, "Important nudge")

    content = result[0]["content"]
    assert content.startswith("<system-reminder>")
    assert content.endswith("</system-reminder>")
    assert "Important nudge" in content
