"""Summarize run completion separately from the checks that ran."""

import copy
from types import SimpleNamespace
from typing import Any

from rune.agent.escalation import can_escalate, honest_failure_note, run_was_verifiable
from rune.agent.verification_state import verified_outcome
from rune.utils.logger import get_logger

log = get_logger(__name__)


def build_cancelled_trust(trace: Any = None, *, artifact_receipts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out = build_trust_payload(trace or SimpleNamespace(reason="cancelled"))
    out.update(completionStatus="cancelled", reason="cancelled", verified=False,
               canEscalate=False, honestNote="", escalationHint="", completionCheck=None)
    if trace is None:
        out["artifactReceipts"] = artifact_receipts or []
    out["artifactReceipts"] = copy.deepcopy(out["artifactReceipts"])
    return out


def _verification_status(trace: Any) -> str:
    verification = getattr(trace, "verification", None) or {}
    gate = getattr(trace, "evidence_gate", None) or {}
    mech = getattr(trace, "mech_check", "")
    if "fail" in (verification.get("status"), gate.get("last_verdict"), mech):
        return "failed"

    # Inspect check evidence without treating an interrupted run as a failed check.
    outcome = verified_outcome({
        "verification": verification,
        "evidence_gate": gate,
        "mech_check": mech,
        "tests_passed_after_edit": getattr(trace, "tests_passed_after_edit", None),
    })
    if outcome is True:
        return "passed"
    if outcome is False:
        return "not_checked"  # A required fresh check is still missing.
    if gate.get("has_check") or mech == "skip":
        return "inconclusive"
    return "not_checked"


def build_trust_payload(trace: Any) -> dict[str, Any]:
    reason = getattr(trace, "reason", "") or ""
    capped = bool(getattr(trace, "tool_budget_exhausted", False))
    if reason == "cancelled":
        completion = "cancelled"
    elif reason == "error" or reason.startswith("error:"):
        completion = "failed"
    elif reason in ("completed", "verified") and not capped:
        completion = "completed"
    else:
        completion = "incomplete" if reason or capped else "unknown"

    verification = getattr(trace, "verification", None)
    status = _verification_status(trace)
    out: dict[str, Any] = {
        "completionStatus": completion,
        "verificationStatus": status,
        "verificationRequired": bool((verification or {}).get("required"))
        or getattr(trace, "tests_passed_after_edit", None) is False,
        "verified": completion == "completed" and status == "passed",
        "reason": reason,
        "budgetExhausted": capped,
        "testsPassedAfterEdit": getattr(trace, "tests_passed_after_edit", None),
        "verification": verification,
        "completionCheck": getattr(trace, "completion_check", None)
        if reason in ("completed_gate_warnings", "max_gate_blocked") else None,
        "artifactReceipts": getattr(trace, "artifact_receipts", []),
        "canEscalate": can_escalate(reason),
        "honestNote": honest_failure_note(reason, run_was_verifiable(trace)) or "",
        "escalationHint": "",
    }
    gate = getattr(trace, "evidence_gate", None)
    if isinstance(gate, dict):
        out["evidenceGate"] = {
            "hasCheck": gate.get("has_check", False),
            "lastVerdict": gate.get("last_verdict", ""),
            "verdictCounts": gate.get("verdict_counts", {}),
            "lastEvidence": gate.get("last_evidence", ""),
        }
    if out["canEscalate"]:
        try:
            from rune.agent.escalation import escalation_hint

            out["escalationHint"] = escalation_hint(reason) or ""
        except Exception as exc:
            log.debug("trust_payload_hint_failed", error=str(exc)[:100])
    return out
