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


# --- honest-failure surface (independent of escalation config) ----------------


def test_honest_note_for_max_gate_blocked():
    from rune.agent.escalation import honest_failure_note

    note = honest_failure_note("max_gate_blocked")
    assert note and "test" in note.lower()
    # The honest stance: do not claim an unverified result.
    assert "verif" in note.lower() or "claim" in note.lower()


def test_honest_note_is_config_independent(monkeypatch):
    # Unlike escalation_hint, the honest note shows even with no escalation model.
    from rune.agent.escalation import honest_failure_note

    _set(monkeypatch, None, None)
    assert honest_failure_note("max_gate_blocked")
    assert honest_failure_note("advisor_abort")


def test_honest_note_none_for_success_and_unknown():
    from rune.agent.escalation import honest_failure_note

    assert honest_failure_note("completed") is None
    assert honest_failure_note("no_progress") is None
    assert honest_failure_note("") is None


def test_setup_hint_only_when_capability_fail_and_unconfigured(monkeypatch):
    from rune.agent.escalation import escalation_setup_hint

    # No escalation configured + a capability failure -> tell user how to enable.
    _set(monkeypatch, None, None)
    s = escalation_setup_hint("max_gate_blocked")
    assert s and "escalation_provider" in s and "/escalate" in s
    # Non-capability reason -> no setup hint (escalation wouldn't be suggested).
    assert escalation_setup_hint("stalled") is None


def test_setup_hint_silent_when_already_configured(monkeypatch):
    from rune.agent.escalation import escalation_setup_hint

    # When configured, escalation_hint() carries the actionable line instead.
    _set(monkeypatch, "anthropic", "claude-sonnet-4-6")
    assert escalation_setup_hint("max_gate_blocked") is None
