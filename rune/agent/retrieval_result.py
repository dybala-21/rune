"""Completion is tied to the collector's fixed contract and exact rendered result."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from rune.agent.task_evidence import TaskEvidence


def accepted_answer(evidence: TaskEvidence | None) -> str:
    report = getattr(evidence, "retrieval_acceptance", None) or {}
    if report.get("status") != "pass" or not report.get("contract") or not report.get("answer"):
        return ""
    from rune.agent.stay_contract import bound_query
    from rune.lodging.evidence import digest

    query = bound_query(evidence)
    if query is None or report["contract"] != digest(query.model_dump_json()):
        return ""
    try:
        expiry = datetime.fromisoformat(report["expires_at"])
        result = json.loads(report["answer"])
        if expiry <= datetime.now(UTC) or result["limitations"]:
            return ""
        if len(result["offers"]) != query.count or report["progress"]["required"] != query.count:
            return ""
    except (ValueError, KeyError, TypeError):
        return ""
    return cast(str, report["answer"])


def progress_hint(evidence: TaskEvidence) -> str:
    report = evidence.retrieval_acceptance or {}
    if not report:
        return ""
    return ("Saved comparison evidence survives this stream boundary. Keep completed fields and retrieve only missing ones. "
            "Use browser_stay_evidence(status/read) for full saved sources. Progress: "
            + json.dumps(report.get("progress", {}), ensure_ascii=False)[:4000])


def partial_answer(evidence: TaskEvidence | None) -> str:
    """Recover exact already-checked fields when no final model call is available."""
    report = getattr(evidence, "retrieval_acceptance", None) or {}
    try:
        result = json.loads(report["partial_answer"])
        if not isinstance(result["offers"], list) or not isinstance(result["limitations"], list):
            return ""
        expiry = report.get("expires_at")
        if expiry and datetime.fromisoformat(expiry) <= datetime.now(UTC):
            result = {"offers": [], "limitations": ["Saved observations expired; fresh evidence is required"]}
        elif not result["limitations"]:
            result["limitations"] = ["Run stopped before completion was accepted"]
        return json.dumps(result, ensure_ascii=False)
    except (ValueError, KeyError, TypeError):
        return ""


def acceptance(evidence: TaskEvidence) -> dict[str, Any] | None:
    report = evidence.retrieval_acceptance
    if report is None or evidence.browser_effect != "view":
        return None
    ready = bool(accepted_answer(evidence))
    return {"required": True, "status": "pass" if ready else "inconclusive",
            "method": "captured_stay_evidence", "scope": report.get("scope", ""),
            "contract": report.get("contract", ""), "requirements": [],
            "detail": "Captured stay fields and exact quotes verified." if ready else progress_hint(evidence),
            "evidence_path": report.get("evidence_path", "")}
