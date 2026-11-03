"""Bind accommodation collection to the request router, before executor tools run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rune.agent.browser_requirements import request_key
from rune.agent.tool_errors import ToolPreparationError
from rune.lodging.evidence import StayQuery

if TYPE_CHECKING:
    from rune.agent.task_evidence import TaskEvidence

INSTRUCTION = (
    "stay_query: null except a NOL accommodation comparison whose complete requested conditions "
    "fit this schema. Extract city, full check-in/out dates, adults, room count, distinct property count "
    "and the maximum displayed room charge in KRW from the user's request/context, before browsing. "
    "Never reduce the requested count or loosen dates, occupancy or budget. Do not invent missing "
    "dates, count or budget. Use city_labels=[] unless translations are known. If conditions are "
    "ambiguous, missing, or include additional requirements this schema cannot represent, return null; "
    "the executor must clarify rather than claim a verified comparison. This is independent of the "
    "collector's later arguments. It never authorizes booking."
)


def response_schema() -> dict[str, Any]:
    schema = StayQuery.model_json_schema()
    schema["required"] = list(schema["properties"])

    def compact(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if key not in {"title", "default"}}
        return [compact(item) for item in value] if isinstance(value, list) else value

    return {"anyOf": [compact(schema), {"type": "null"}]}


def validated_query(value: Any) -> StayQuery:
    if not isinstance(value, dict) or set(value) != set(StayQuery.model_fields):
        raise ValueError("A routed stay query must explicitly provide every condition")
    return StayQuery.model_validate(value)


def bound_query(evidence: TaskEvidence | None) -> StayQuery | None:
    requests = getattr(evidence, "request_texts", ())
    if not requests or getattr(evidence, "stay_query_key", "") != request_key(requests[-1:]):
        return None
    try:
        return validated_query(getattr(evidence, "stay_query", None))
    except ValueError:
        return None


def prepare(evidence: TaskEvidence, params: dict[str, Any]) -> None:
    if params.get("action") != "begin":
        return
    expected = bound_query(evidence)
    if expected is None:
        raise ToolPreparationError(
            "stay_request_unbound",
            "Accommodation conditions were not fixed from this request before execution. "
            "Clarify missing conditions and start a new comparison; do not invent a smaller contract.",
        )
    if StayQuery.model_validate(params.get("query")) != expected:
        raise ToolPreparationError(
            "stay_request_mismatch",
            "Use the fixed requested conditions without changing the count, dates, guests or budget.",
            query=expected.model_dump(mode="json"),
        )


def prompt(evidence: TaskEvidence) -> str:
    query = bound_query(evidence)
    if query is None:
        return ""
    return ("Accommodation comparison conditions fixed from the user request: " + query.model_dump_json()
            + ". Pass this exact query to browser_stay_evidence begin. The collector cannot relax it.")
