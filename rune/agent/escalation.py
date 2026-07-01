"""Escalation hinting for verified-unfixable failures.

Two terminal outcomes mean the local model could not finish and a stronger model
is the next step:

- ``max_gate_blocked``: the agent verified its output against the project's tests,
  the tests failed, and repeated self-fix attempts did not converge.
- ``advisor_abort``: the in-loop advisor (a stronger reviewer) inspected the run
  and recommended stopping.

We only suggest the next step; data leaves the machine on an explicit
``/escalate``, never automatically. The message adapts to the advisor's state so
the two rungs of the ladder (in-loop advisor, then full /escalate handoff) stay
coherent.
"""

from __future__ import annotations

_HINT_REASONS = ("max_gate_blocked", "advisor_abort")

# /goal outer-loop terminal causes where the model tried and could not pass
# validation. "cancelled" (user stopped) and "error" (crash) are not capability
# failures, and "verified" is success, so none of those suggest escalation.
_GOAL_STUCK_CAUSES = ("stagnation", "max_iterations", "budget")


def escalation_hint(reason: str) -> str | None:
    """Return a one-line escalation suggestion, or None.

    Shown only on a terminal local failure AND when an escalation profile is
    configured, so the suggestion is always actionable.
    """
    if reason not in _HINT_REASONS:
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

    # max_gate_blocked: the solution still fails the project's tests.
    base = (
        "The solution still fails the project's tests after repeated self-fix "
        f"attempts. Run /escalate to retry once on {target}"
    )
    return _with_advisor_suffix(base)


# Why the run stopped, in plain words. Shown whether or not an escalation model
# is set — RUNE says what it couldn't verify rather than claiming it's done.
_HONEST_STOP_NOTES = {
    "max_gate_blocked": (
        "Not marking this done: the solution still fails its tests after repeated "
        "self-fix attempts, so I won't claim a result I can't verify."
    ),
    "advisor_abort": (
        "Stopping here: a stronger reviewer inspected this run and advised against "
        "shipping an unverified result."
    ),
    "completed_gate_warnings": (
        "Delivered with caveats: the artifact exists but some quality checks did "
        "not fully pass — treat it as unverified."
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


def honest_failure_note(reason: str) -> str | None:
    """Why the run stopped, in plain words. None for success or unknown reasons."""
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
