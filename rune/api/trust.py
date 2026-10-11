"""Summarize run completion separately from the checks that ran."""

import copy
from types import SimpleNamespace
from typing import Any

from rune.agent.escalation import can_escalate, honest_failure_note, run_was_verifiable
from rune.agent.run_outcome import run_outcome
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


def build_trust_payload(trace: Any) -> dict[str, Any]:
    outcome = run_outcome(trace)
    reason = outcome.reason
    out: dict[str, Any] = {
        **outcome.payload(),
        "reason": reason,
        "budgetExhausted": bool(getattr(trace, "tool_budget_exhausted", False)) or reason == "request_budget_exhausted",
        "testsPassedAfterEdit": getattr(trace, "tests_passed_after_edit", None),
        "verification": getattr(trace, "verification", None),
        "completionCheck": getattr(trace, "completion_check", None)
        if reason in ("completed_gate_warnings", "max_gate_blocked", "desktop_blocked", "request_budget_exhausted") else None,
        "artifactReceipts": getattr(trace, "artifact_receipts", []),
        "tableAcceptance": getattr(trace, "table_acceptance", None),
        "requirementAcceptance": getattr(trace, "requirement_acceptance", None),
        "workspaceWarning": getattr(trace, "workspace_warning", "") or "",
        "unsourcedNumbers": list(getattr(trace, "unsourced_numbers", None) or []),
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
