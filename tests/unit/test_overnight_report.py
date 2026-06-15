"""Morning report for `rune overnight`: honest success vs not-done."""
from __future__ import annotations

from rune.agent.overnight_report import format_overnight_report


def test_success_lists_changes_and_checks():
    r = format_overnight_report(
        goal="add a helper", success=True, stop_cause="verified",
        iterations=2, validation=["pytest -q"], changed_files=["helper.py"],
    )
    assert "DONE" in r and "pytest -q" in r and "helper.py" in r
    assert "fabricated" not in r


def test_not_done_states_reason_not_success():
    r = format_overnight_report(
        goal="hard task", success=False, stop_cause="max_iterations",
        iterations=3, validation=["pytest -q"], changed_files=[],
        escalation_hint="Run /escalate to retry on a stronger model.",
    )
    assert "NOT done" in r and "ran out of attempts" in r
    assert "no fabricated success" in r
    assert "/escalate" in r


def test_each_stuck_cause_has_a_reason():
    for cause in ("max_iterations", "stagnation", "budget", "advisor_abort"):
        r = format_overnight_report(
            goal="g", success=False, stop_cause=cause, iterations=1,
            validation=[], changed_files=[],
        )
        assert "NOT done" in r and cause not in r.split("NOT done")[1][:10]  # mapped to prose
