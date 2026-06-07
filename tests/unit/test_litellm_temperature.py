"""gpt-5 / o-series reasoning models reject temperature != 1, so the agent
loop must omit the param for them (see _rejects_temperature)."""

from __future__ import annotations

from rune.agent.litellm_adapter import _rejects_temperature


def test_reasoning_models_reject_temperature():
    assert _rejects_temperature("gpt-5")
    assert _rejects_temperature("gpt-5-mini")
    assert _rejects_temperature("openai/gpt-5-codex")
    assert _rejects_temperature("o1")
    assert _rejects_temperature("o3-mini")
    assert _rejects_temperature("o4-mini")


def test_temperature_capable_models_allowed():
    # gpt-5.x point releases allow temperature; non-reasoning models too.
    assert not _rejects_temperature("gpt-5.1")
    assert not _rejects_temperature("gpt-4o")
    assert not _rejects_temperature("gemini-2.5-flash")
    assert not _rejects_temperature("claude-haiku-4-5")
