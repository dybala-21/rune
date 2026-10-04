"""Completion and verification shared by interactive and background runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.agent.verification_state import verified_outcome


def verification_status(trace: Any) -> str:
    verification = getattr(trace, "verification", None) or {}
    gate = getattr(trace, "evidence_gate", None) or {}
    mech = getattr(trace, "mech_check", "")
    table = getattr(trace, "table_acceptance", None) or {}
    requirements = getattr(trace, "requirement_acceptance", None) or {}
    if "fail" in (verification.get("status"), gate.get("last_verdict"), mech, requirements.get("status")):
        return "failed"
    if table.get("required") and table.get("status") != "pass":
        return {"fail": "failed", "unverified": "not_checked"}.get(table.get("status"), "inconclusive")
    if requirements.get("required") and requirements.get("status") != "pass":
        return "inconclusive"
    if verification.get("status") == "inconclusive":
        return "inconclusive"

    # Inspect check evidence without treating an interrupted run as a failed check.
    outcome = verified_outcome({
        "verification": verification,
        "evidence_gate": gate,
        "mech_check": mech,
        "tests_passed_after_edit": getattr(trace, "tests_passed_after_edit", None),
        "table_acceptance": table,
    })
    if outcome is True:
        return "passed"
    if outcome is False:
        return "not_checked"  # A required fresh check is still missing.
    if gate.get("has_check") or mech == "skip":
        return "inconclusive"
    return "not_checked"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    completion: str
    verification: str
    verification_required: bool
    reason: str

    @property
    def success(self) -> bool:
        return (self.completion == "completed" and self.verification != "failed"
                and (not self.verification_required or self.verification == "passed"))

    @property
    def verified(self) -> bool:
        return self.completion == "completed" and self.verification == "passed"

    @property
    def status(self) -> str:
        if self.completion != "completed" or self.verification == "failed":
            return "failed"
        if self.verified:
            return "verified"
        if self.verification_required or self.verification == "inconclusive":
            return "unverified"
        return "completed"

    def payload(self) -> dict[str, Any]:
        return {"completionStatus": self.completion, "verificationStatus": self.verification,
                "verificationRequired": self.verification_required, "verified": self.verified,
                "reason": self.reason}


def run_outcome(trace: Any) -> RunOutcome:
    reason = getattr(trace, "reason", "") or ""
    requirements = getattr(trace, "requirement_acceptance", None) or {}
    capped = bool(getattr(trace, "tool_budget_exhausted", False))
    if reason == "cancelled":
        completion = "cancelled"
    elif reason == "error" or reason.startswith("error:"):
        completion = "failed"
    elif reason in ("completed", "verified") and not capped:
        completion = "completed"
    else:
        completion = "incomplete" if reason or capped else "unknown"
    required = (bool((getattr(trace, "verification", None) or {}).get("required"))
                or getattr(trace, "tests_passed_after_edit", None) is False
                or bool((getattr(trace, "table_acceptance", None) or {}).get("required"))
                or bool(requirements.get("required") and requirements.get("status") != "pass"))
    return RunOutcome(completion, verification_status(trace), required, reason)
