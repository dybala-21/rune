"""Escalation hinting for verified-unfixable failures.

When a run ends in ``max_gate_blocked`` the agent verified its output against the
project's tests, the tests failed, and repeated self-fix attempts did not
converge. A weak local model often cannot self-correct such an error, so the next
step is a stronger model. We only suggest it: data leaves the machine on an
explicit ``/escalate``, never automatically.
"""

from __future__ import annotations

_VERIFIED_FAILURE_REASON = "max_gate_blocked"


def escalation_hint(reason: str) -> str | None:
    """Return a one-line escalation suggestion, or None.

    Shown only when the run hit the verified-failure cap AND an escalation
    profile is configured — never hint at something the user cannot act on.
    """
    if reason != _VERIFIED_FAILURE_REASON:
        return None
    from rune.config import get_config

    cfg = get_config().llm
    provider = cfg.escalation_provider
    if not provider:
        return None
    target = cfg.escalation_model or f"{provider}'s best model"
    return (
        "Verified the solution still fails the project's tests after repeated "
        "self-fix attempts — a systematic error a weak model will not fix by "
        f"retrying. Run /escalate to retry once on {target}."
    )
