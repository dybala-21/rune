"""Stop messages and optional suggestions to retry with another model.

An unresolved completion check or advisor abort can suggest ``/escalate``
when a retry profile is configured. This module does not start the retry.
"""

from __future__ import annotations

_HINT_REASONS = ("max_gate_blocked", "advisor_abort")

# Goal-loop limits that qualify for escalation; cancellation and crashes do not.
_GOAL_STUCK_CAUSES = ("stagnation", "max_iterations", "budget")


def can_escalate(reason: str) -> bool:
    """Whether the stop reason supports offering a stronger-model retry."""
    return reason in _HINT_REASONS


def escalation_hint(reason: str) -> str | None:
    """Suggest a retry when the stop reason and configured profile support it."""
    if not can_escalate(reason):
        return None
    from rune.config import get_config

    cfg = get_config().llm
    provider = cfg.escalation_provider
    if not provider:
        return None
    target = cfg.escalation_model or f"{provider}'s best model"

    if reason == "advisor_abort":
        return (
            "The advisor reviewed this run and recommended stopping. Run "
            f"/escalate to hand the whole task to {target}."
        )

    # This may be an answer-evidence check, even when the tests passed.
    base = (
        "Required completion checks remain unresolved after correction attempts. "
        f"Run /escalate to retry once on {target}"
    )
    return _with_advisor_suffix(base)


# Stop messages are shown even when no escalation profile is configured.
_HONEST_STOP_NOTES = {
    "desktop_blocked": (
        "The desktop task stopped before its result could be confirmed. Review the app or connection error before retrying."
    ),
    "max_gate_blocked": (
        "Required completion checks remain unresolved. Review the specific check before treating this result as complete."
    ),
    "advisor_abort": (
        "Stopping here: a stronger reviewer inspected this run and advised against "
        "shipping an unverified result."
    ),
    "completed_gate_warnings": (
        "The response was returned, but completion checks remain unresolved."
    ),
    "checks_failed": (
        "Not marking this done: the last test run on this work FAILED and no "
        "later run passed. The changes are in place but unverified."
    ),
    "task_blocked": (
        "Stopping: the task cannot be completed correctly as stated — see the "
        "conflict described above. Nothing was forced through."
    ),
    "stalled": (
        "Stopping: I stopped making progress and won't claim a result I didn't "
        "actually reach."
    ),
    "token_budget_exhausted": (
        "Stopping: I ran out of budget before I could verify the result, so I'm "
        "not marking it done."
    ),
}


def run_was_verifiable(trace: object) -> bool:
    """Did the run produce something checkable — files changed, or a
    verification gate that actually ran?

    A pure question/answer or writing task (draw a table, summarise) has
    neither, so telling the user "I couldn't verify the result" is a category
    error: there was nothing to verify. Unknown → True, keeping the stricter
    coding framing for the higher-stakes case.
    """
    if trace is None:
        return True
    gate = getattr(trace, "evidence_gate", None)
    if isinstance(gate, dict) and gate.get("has_check"):
        return True
    if getattr(trace, "tests_passed_after_edit", None) is not None:
        return True
    return bool(getattr(trace, "changed_files", 0))


def honest_failure_note(reason: str, verifiable: bool = True) -> str | None:
    """Why the run stopped, in plain words. None for success or unknown reasons.

    ``verifiable`` (from :func:`run_was_verifiable`) reframes the budget note
    for a task with nothing to verify, so a writing/Q&A run doesn't claim it
    "couldn't verify" a result that never needed verifying.
    """
    if reason == "token_budget_exhausted" and not verifiable:
        return (
            "Stopping: I ran out of budget before finishing this, so the answer "
            "above may be incomplete — I'm not marking it done."
        )
    return _HONEST_STOP_NOTES.get(reason)


def escalation_setup_hint(reason: str) -> str | None:
    """How to enable escalation, when it would help but isn't set up yet.

    None if escalation is already configured (escalation_hint covers it) or the
    reason isn't one a stronger model would fix."""
    if reason not in _HINT_REASONS:
        return None
    from rune.config import get_config

    if get_config().llm.escalation_provider:
        return None
    return (
        "To retry once on a stronger model, set llm.escalation_provider and "
        "llm.escalation_model, then run /escalate."
    )


def goal_escalation_hint(stop_cause: str) -> str | None:
    """Escalation suggestion for a /goal outer-loop run that ended stuck.

    The autonomous loop iterated fresh attempts and could not pass validation.
    Same contract as ``escalation_hint``: shown only on a stuck cause AND when an
    escalation profile is configured, so the suggestion is always actionable.
    """
    if stop_cause not in _GOAL_STUCK_CAUSES:
        return None
    from rune.config import get_config

    cfg = get_config().llm
    provider = cfg.escalation_provider
    if not provider:
        return None
    target = cfg.escalation_model or f"{provider}'s best model"
    base = (
        "The goal loop could not pass validation after repeated fresh attempts. "
        f"Run /escalate to retry on {target}"
    )
    return _with_advisor_suffix(base)


def _with_advisor_suffix(base: str) -> str:
    """Append the in-loop advisor hint only when the advisor is off (else it is
    already the earlier rung of the same ladder)."""
    from rune.agent.advisor.runtime_toggle import is_advisor_enabled

    if is_advisor_enabled():
        return base + "."
    return base + ", or turn on the in-loop advisor with /advisor."
