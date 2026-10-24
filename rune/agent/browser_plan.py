"""Bind browser outcomes before the first input."""

from copy import deepcopy

from pydantic import ValidationError

from rune.agent.browser_requirements import EXECUTOR_KINDS, KIND_GUIDE, quote_repair, validated
from rune.agent.tool_errors import ToolPreparationError, validation_details
from rune.utils.logger import get_logger

log = get_logger(__name__)

ENTRY_TOOLS = {"browser_navigate", "browser_open", "browser_observe"}
PLAN_TOOLS = ENTRY_TOOLS | {"browser_act", "browser_batch"}


def executor_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["items"],
        "description": (
            "Supply once before input; omit after Task outcomes are returned. Selectors belong in actions/checks. "
            "For 'Set count to 3, save and confirm': "
            '{"items":[{"requirement":"count to 3","kind":"value","expected":"3"},'
            '{"requirement":"save and confirm","kind":"result","expected":null}]}'
        ),
        "properties": {
            "items": {"type": "array", "maxItems": 16, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["requirement", "kind", "expected"],
                "properties": {
                    "requirement": {"type": "string", "maxLength": 1000, "description": "Copy a continuous phrase from the user, including the value. For a save and its confirmation, copy the whole combined phrase into ONE result."},
                    "kind": {"type": "string", "enum": list(EXECUTOR_KINDS), "description": KIND_GUIDE},
                    "expected": {"anyOf": [{"type": "string", "maxLength": 1000}, {"type": "null"}], "description": "Literal in quote; null for save outcomes or unknown values."},
                    "value_type": {"type": "string", "enum": ["text", "number", "time"],
                                   "description": "Numbers/times apply to field values. Saved results use text."},
                    "canonical": {"type": "string", "maxLength": 1000, "description": "Optional equivalent decimal/ISO value, never a field name."},
                },
            }},
        },
    }


def _normalize_plan(value: dict, requests: tuple[str, ...], bound=None) -> dict:
    result = deepcopy(value)
    items = result.get("items")
    if not isinstance(items, list):
        return result
    fixed = {item.requirement: item for item in bound.items} if bound is not None else {}
    for item in items:
        if not isinstance(item, dict):
            continue
        existing = fixed.get(item.get("requirement")) if isinstance(item.get("requirement"), str) else None
        if existing is not None:
            for key, reference in (("id", existing.id), ("requirement_ids", [existing.id])):
                if item.get(key) == reference:
                    item.pop(key)
        if item.get("kind") != "result":
            continue
        expected, quote = item.get("expected"), item.get("requirement")
        if (isinstance(expected, str) and isinstance(quote, str) and quote
                and any(quote in request for request in requests)
                and not any(expected in request for request in requests)
                and item.get("value_type", "text") == "text" and item.get("canonical") is None):
            # A generated success message is not a user-specified condition.
            item["expected"] = None
            log.debug("browser_unrequested_result_wording")
    return result


def configure_tools(evidence, tools: dict) -> None:
    if not evidence.requirements_unavailable:
        return
    for name in {"browser_observe", "browser_batch"} & tools.keys():
        tool = tools[name]
        tool.json_schema = deepcopy(tool.json_schema)
        tool.json_schema["properties"]["taskRequirements"] = executor_schema()


def accept_plan(evidence, name: str, params: dict) -> None:
    if "taskRequirements" not in params:
        return
    value = params.pop("taskRequirements")
    if name not in PLAN_TOOLS or not evidence.coverage_required:
        raise ToolPreparationError("unexpected_requirements", "Declare browser outcomes before the first input.")
    if isinstance(value, dict):
        value = {"complete": True, **value}
        if evidence.requirements is not None or not evidence.browser_actions:
            value = _normalize_plan(value, evidence.request_texts, evidence.requirements)
    try:
        plan = validated(value, evidence.request_texts)
    except ValueError as exc:
        detail = validation_details(exc) if isinstance(exc, ValidationError) else str(exc)
        if evidence.requirements is not None:
            raise ToolPreparationError(
                "fixed_requirements", "Omit taskRequirements; use the already bound outcomes in actions and checks.",
                detail=detail[:1000],
            ) from exc
        proposal = quote_repair(value, evidence.request_texts) if evidence.requirements is None and not evidence.browser_actions else None
        raise ToolPreparationError(
            "invalid_requirements",
            "Use browser_observe(taskRequirements=...) to correct the quoted conditions before input. "
            "Keep selectors/actions outside taskRequirements; use null for unspecified save wording.",
            detail=detail[:1000], requests=evidence.request_texts,
            **({"suggestedRequirements": proposal} if proposal else {}),
        ) from exc
    if plan is None or not plan.items:
        raise ToolPreparationError("empty_requirements", "List the requested end states before changing data.")
    if evidence.requirements is not None:
        if evidence.requirements == plan:
            return
        raise ToolPreparationError("fixed_requirements", "Omit taskRequirements; use the already bound outcomes in actions and checks.")
    if evidence.browser_actions:
        raise ToolPreparationError("late_requirements", "Outcomes cannot be changed after input.")
    saves = [item.requirement for item in plan.outcomes if item.kind == "result"]
    if len(saves) > 1 and not evidence.multiple_saves_confirmed:
        # Each result needs its own saved record; ask once and accept a resubmission.
        evidence.multiple_saves_confirmed = True
        raise ToolPreparationError(
            "multiple_saves",
            f"This plan lists {len(saves)} separate saved results, and each needs its own saved object. A limit such as "
            "'save only once' or a confirmation of the same save is not another result: merge it into that save's item. "
            "If these really are different objects to save, submit the same items again.",
            results=saves,
        )
    evidence.requirements = plan
    evidence.plan_correction_required = False
    evidence.plan_pending = True
