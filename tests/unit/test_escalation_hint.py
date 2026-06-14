"""Escalation hint on verified-unfixable failures (max_gate_blocked)."""
from __future__ import annotations

from rune.agent.escalation import escalation_hint


def _set(monkeypatch, provider, model=None):
    from rune.config import get_config
    cfg = get_config().llm
    monkeypatch.setattr(cfg, "escalation_provider", provider, raising=False)
    monkeypatch.setattr(cfg, "escalation_model", model, raising=False)


def test_no_hint_on_normal_completion(monkeypatch):
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    assert escalation_hint("completed") is None


def test_no_hint_for_other_failures(monkeypatch):
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    assert escalation_hint("stalled") is None
    assert escalation_hint("token_budget_exhausted") is None


def test_no_hint_when_no_escalation_profile(monkeypatch):
    # Never suggest something the user cannot act on.
    _set(monkeypatch, None, None)
    assert escalation_hint("max_gate_blocked") is None


def test_hint_on_verified_failure_with_profile(monkeypatch):
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    hint = escalation_hint("max_gate_blocked")
    assert hint and "/escalate" in hint and "claude-sonnet-4-6" in hint


def test_hint_falls_back_to_provider_when_no_model(monkeypatch):
    _set(monkeypatch, "openai", None)
    hint = escalation_hint("max_gate_blocked")
    assert hint and "openai's best model" in hint
