"""honest_failure_note must not claim it 'couldn't verify' a task with nothing to verify."""
from types import SimpleNamespace

from rune.agent.escalation import honest_failure_note, run_was_verifiable


# --- run_was_verifiable ---
def test_verifiable_when_files_changed():
    assert run_was_verifiable(SimpleNamespace(changed_files=3)) is True


def test_verifiable_when_gate_ran():
    assert run_was_verifiable(SimpleNamespace(evidence_gate={"has_check": True})) is True


def test_verifiable_when_tests_ran():
    assert run_was_verifiable(SimpleNamespace(tests_passed_after_edit=True)) is True
    assert run_was_verifiable(SimpleNamespace(tests_passed_after_edit=False)) is True


def test_not_verifiable_for_pure_generation():
    # A table/writing run: no files, no gate check, no tests.
    t = SimpleNamespace(changed_files=0, evidence_gate=None, tests_passed_after_edit=None)
    assert run_was_verifiable(t) is False


def test_unknown_trace_defaults_verifiable():
    assert run_was_verifiable(None) is True


# --- honest_failure_note branching ---
def test_budget_note_generation_drops_verify_wording():
    note = honest_failure_note("token_budget_exhausted", verifiable=False)
    assert note is not None
    assert "verify" not in note.lower()
    assert "incomplete" in note.lower()
    assert "not marking it done" in note.lower()


def test_budget_note_coding_keeps_verify_wording():
    note = honest_failure_note("token_budget_exhausted", verifiable=True)
    assert "verify" in note.lower()


def test_default_is_verifiable_backcompat():
    # No arg → old behaviour (coding framing).
    assert honest_failure_note("token_budget_exhausted") == honest_failure_note(
        "token_budget_exhausted", verifiable=True
    )


def test_other_reasons_unaffected_by_verifiable():
    for reason in ("max_gate_blocked", "checks_failed", "stalled", "task_blocked"):
        assert honest_failure_note(reason, verifiable=False) == honest_failure_note(
            reason, verifiable=True
        )


def test_unknown_reason_returns_none():
    assert honest_failure_note("completed", verifiable=False) is None
    assert honest_failure_note("nonexistent") is None
