"""Completion and verification shared by interactive and background runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.agent.verification_state import verification_status


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
        if self.completion == "unknown" and self.verification != "failed":
            return "unverified"
        if self.completion != "completed" or self.verification == "failed":
            return "failed"
        if self.verified:
            return "verified"
        if self.verification_required or self.verification in {"inconclusive", "unavailable"}:
            return "unverified"
        return "completed"

    def payload(self) -> dict[str, Any]:
        return {"completionStatus": self.completion, "verificationStatus": self.verification,
                "verificationRequired": self.verification_required, "verified": self.verified,
                "reason": self.reason}


def run_outcome(trace: Any) -> RunOutcome:
    if isinstance(trace, dict):
        from types import SimpleNamespace
        trace = SimpleNamespace(**trace)
    reason = getattr(trace, "reason", "") or ""
    requirements = getattr(trace, "requirement_acceptance", None) or {}
    capped = bool(getattr(trace, "tool_budget_exhausted", False))
    if reason == "cancelled":
        completion = "cancelled"
    elif reason == "error" or reason.startswith("error:"):
        completion = "failed"
    elif reason in ("completed", "verified") and not capped:
        completion = "completed"
    elif reason in {"verification_unavailable", "verification_inconclusive"}:
        completion = "unknown"
    else:
        completion = "incomplete" if reason or capped else "unknown"
    required = (bool((getattr(trace, "verification", None) or {}).get("required"))
                or getattr(trace, "tests_passed_after_edit", None) is False
                or bool((getattr(trace, "table_acceptance", None) or {}).get("required"))
                or bool(requirements.get("required") and requirements.get("status") not in {"pass", "acknowledged"}))
    return RunOutcome(completion, verification_status(trace), required, reason)


def merge_requirement_checks(left: dict | None, right: dict | None) -> dict | None:
    if not left or not right:
        return left or right
    required = [check for check in (left, right) if check.get("required")]
    states = {check.get("status") for check in required}
    if states - {"pass", "acknowledged", "fail", "unavailable", "inconclusive", "unverified", "not_checked"}:
        states.add("inconclusive")
    status = next((state for state in ("fail", "unavailable", "inconclusive", "unverified", "not_checked", "acknowledged")
                   if state in states), "pass")
    return {"required": bool(required), "status": status, "method": "combined",
            "requirements": [item for check in required for item in check.get("requirements", [])],
            "detail": "\n".join(check["detail"] for check in required if check.get("detail")),
            "components": [left, right]}


def explanation_warning(trace: Any) -> str:
    review = getattr(trace, "explanation_review", None) or {}
    if run_outcome(trace).success and review.get("status") == "unavailable":
        return "Execution checks passed. The final explanation could not be verified."
    return ""
