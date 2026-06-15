"""Escalation hint on verified-unfixable failures (max_gate_blocked / advisor_abort)."""
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


def test_hint_on_advisor_abort(monkeypatch):
    # The advisor recommending abort is the strongest escalate signal; it must
    # not be silent. Suggest a full /escalate handoff.
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    hint = escalation_hint("advisor_abort")
    assert hint and "/escalate" in hint and "claude-sonnet-4-6" in hint
    assert "advisor" in hint.lower()


def test_advisor_abort_no_hint_without_profile(monkeypatch):
    _set(monkeypatch, None, None)
    assert escalation_hint("advisor_abort") is None


def test_max_gate_suggests_advisor_when_disabled(monkeypatch):
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "0")
    hint = escalation_hint("max_gate_blocked")
    assert hint and "/advisor" in hint  # points at the lighter in-loop rung


def test_max_gate_omits_advisor_when_enabled(monkeypatch):
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "1")
    hint = escalation_hint("max_gate_blocked")
    assert hint and "/advisor" not in hint  # advisor already on; only /escalate


# --- /goal outer-loop escalation hint (gap #4, L1) ---

def test_goal_hint_on_stuck_causes(monkeypatch):
    from rune.agent.escalation import goal_escalation_hint
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    for cause in ("stagnation", "max_iterations", "budget"):
        h = goal_escalation_hint(cause)
        assert h and "/escalate" in h and "claude-sonnet-4-6" in h, cause


def test_goal_hint_none_on_success_cancel_error(monkeypatch):
    from rune.agent.escalation import goal_escalation_hint
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    for cause in ("verified", "cancelled", "error"):
        assert goal_escalation_hint(cause) is None, cause


def test_goal_hint_none_without_profile(monkeypatch):
    from rune.agent.escalation import goal_escalation_hint
    _set(monkeypatch, None, None)
    assert goal_escalation_hint("stagnation") is None
