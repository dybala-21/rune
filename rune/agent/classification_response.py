"""Constrain routing responses and reject incomplete decisions before execution."""

from __future__ import annotations

import asyncio
import json
import math
import random
from copy import deepcopy
from typing import Any

from rune.agent.browser_requirements import response_schema
from rune.utils.logger import get_logger

log = get_logger(__name__)

COMPLEXITY_CRITERIA = {
    "simple": "A bounded lookup, inspection, routine edit or known procedure with few constraints.",
    "moderate": "A local algorithm change, numerical precision, or several interacting requirements. Use this when difficulty is unclear.",
    "complex": "Architecture, repository-wide changes, difficult diagnosis or dependent outcomes spanning several systems.",
}

# Shared routing rules live in the prompt.
_PROPERTIES = {
    "goal_type": {"type": "string", "enum": ["chat", "web", "research", "code_modify", "execution", "browser", "full", "artifact"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "reason": {"type": "string"},
    "complexity": {"type": "string", "enum": list(COMPLEXITY_CRITERIA)},
    "requires_execution": {"type": "boolean"},
    "needs_web_evidence": {"type": "boolean"},
    "needs_current_source": {"type": "boolean"},
    "lookup_only": {"type": "boolean"},
    "source_output": {"type": "string", "enum": ["none", "verbatim", "derived", "summary"]},
    "browser_effect": {"type": "string", "enum": ["view", "edit", "change"]},
    "browser_requirements": response_schema(),
    "calculation_operation": {"type": "string", "enum": ["sum", "count", "mean", "min", "max", "other"]},
    "intent_categories": {"type": "array", "items": {"type": "string", "enum": ["coding", "email", "document", "table", "desktop", "calculation"]}},
    "requires_desktop_input": {"type": "boolean"},
    "is_related_to_previous": {"type": "boolean"},
    "table_output": {"type": "string", "enum": ["none", "csv", "xlsx"]},
    "calculation_expression": {
        "type": "string",
        "description": "Copy the complete numeric arithmetic expression verbatim from the current request. Never solve, rewrite or invent it. Use an empty string for all other requests, including native app tasks, identifiers and code-writing requests.",
    },
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "task_routing", "strict": True, "schema": {
        "type": "object", "properties": _PROPERTIES,
        "required": list(_PROPERTIES), "additionalProperties": False,
    },
}}


class InvalidClassification(ValueError):
    pass


class RoutingUnavailable(RuntimeError):
    """A routing request failed before any task tools were selected."""

    def __init__(self, detail: str, *, fallback_reason: str = "", diagnostics: dict | None = None):
        super().__init__(detail)
        self.fallback_reason = fallback_reason
        self.diagnostics = diagnostics or {}


ROUTING_TIMEOUT = 35.0


def decode_object(response: Any) -> dict[str, Any]:
    def get(value, name, default=None):
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)

    choices = get(response, "choices", [])
    if not choices:
        raise InvalidClassification("missing_choice")
    choice = choices[0]
    if get(choice, "finish_reason") in {"content_filter", "refusal"}:
        raise InvalidClassification("refusal")
    if get(choice, "finish_reason") in {"length", "max_tokens"}:
        raise InvalidClassification("incomplete_response")
    message = get(choice, "message", {})
    if get(message, "refusal"):
        raise InvalidClassification("refusal")
    text = get(message, "content")
    if not isinstance(text, str) or not text.strip():
        raise InvalidClassification("empty_response")
    text = text.strip()
    # Accept a single fenced object from providers without schema support.
    if text.startswith("```") and text.endswith("```") and text.count("```") == 2:
        text = text.split("\n", 1)[-1][:-3].strip()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise InvalidClassification("invalid_json") from exc
    if not isinstance(data, dict):
        raise InvalidClassification("invalid_object")
    return data


def decode_response(response: Any) -> dict[str, Any]:
    return validate_decision(decode_object(response))


def validate_decision(data: dict[str, Any]) -> dict[str, Any]:
    if set(data) - {"resume_task", "task_scope", "browser_requirements", "complexity"} != set(_PROPERTIES) - {"browser_requirements", "complexity"}:
        raise InvalidClassification("invalid_fields")
    if "complexity" in data and (not isinstance(data["complexity"], str) or data["complexity"] not in COMPLEXITY_CRITERIA):
        raise InvalidClassification("invalid_complexity")
    if data["goal_type"] not in _PROPERTIES["goal_type"]["enum"]:
        raise InvalidClassification("invalid_goal_type")
    if type(data["confidence"]) not in (int, float) or not math.isfinite(data["confidence"]) or not 0 <= data["confidence"] <= 1:
        raise InvalidClassification("invalid_confidence")
    if not isinstance(data["reason"], str):
        raise InvalidClassification("invalid_reason")
    for key in ("requires_execution", "needs_web_evidence", "needs_current_source", "lookup_only", "requires_desktop_input", "is_related_to_previous"):
        if type(data[key]) is not bool:
            raise InvalidClassification("invalid_" + key)
    if not isinstance(data["intent_categories"], list) or any(
        item not in _PROPERTIES["intent_categories"]["items"]["enum"] for item in data["intent_categories"]
    ):
        raise InvalidClassification("invalid_intent_categories")
    if data["table_output"] not in ("none", "csv", "xlsx"):
        raise InvalidClassification("invalid_table_output")
    if data["source_output"] not in ("none", "verbatim", "derived", "summary"):
        raise InvalidClassification("invalid_source_output")
    for key in ("browser_effect", "calculation_operation"):
        if data[key] not in _PROPERTIES[key]["enum"]:
            raise InvalidClassification("invalid_" + key)
    if "resume_task" in data and (not isinstance(data["resume_task"], str) or len(data["resume_task"]) > 12):
        raise InvalidClassification("invalid_resume_task")
    if "task_scope" in data and data["task_scope"] not in {"task", "step", "inspect", "amend"}:
        raise InvalidClassification("invalid_task_scope")
    if not isinstance(data["calculation_expression"], str):
        raise InvalidClassification("invalid_calculation_expression")
    if data.get("browser_requirements") is not None:
        from rune.agent.browser_requirements import BrowserRequirements
        try:
            BrowserRequirements.model_validate(data["browser_requirements"])
        except ValueError as exc:
            from pydantic import ValidationError

            from rune.agent.tool_errors import validation_details
            detail = validation_details(exc) if isinstance(exc, ValidationError) else "Invalid requirement structure"
            raise InvalidClassification("invalid_browser_requirements: " + detail) from exc
    desktop = "desktop" in data["intent_categories"]
    if data["needs_current_source"] and not data["needs_web_evidence"]:
        data = {**data, "needs_web_evidence": True}
    if data["lookup_only"] and (not data["needs_web_evidence"] or data["goal_type"] not in {"chat", "web"}
                                or data["requires_execution"] or data["intent_categories"]):
        log.debug("lookup_reasoning_disabled", goal_type=data["goal_type"])
        data = {**data, "lookup_only": False}
    if data["goal_type"] == "artifact" and "coding" in data["intent_categories"]:
        raise InvalidClassification("inconsistent_artifact_route")
    if data["requires_desktop_input"] and not desktop:
        raise InvalidClassification("inconsistent_desktop_input")
    if desktop and data["goal_type"] != "full":
        raise InvalidClassification("inconsistent_desktop_route")
    if data["table_output"] != "none" and data["goal_type"] in {"chat", "web"}:
        raise InvalidClassification("inconsistent_table_route")
    # Only direct answers read an expression; other routes drop it, since small models fill it with filler.
    if data["calculation_expression"] and data["goal_type"] != "chat":
        log.debug("calculation_expression_dropped", goal_type=data["goal_type"])
        data = {**data, "calculation_expression": ""}
    if data["calculation_expression"] and (data["goal_type"] != "chat"
            or data["requires_execution"] or data["table_output"] != "none"):
        raise InvalidClassification("inconsistent_calculation_route")
    return data


async def request_classification(
    client: Any, system: str, content: str, *, selected=None, deadline: float | None = None,
    file_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    from rune.llm.model_selection import get_effective_model_selection
    from rune.llm.reasoning import reasoning_control

    selected = selected or get_effective_model_selection()
    from rune.agent.browser_requirements import INSTRUCTION, validated

    system += "\n" + INSTRUCTION
    response_format = RESPONSE_FORMAT
    try:
        state = json.loads(content)
    except ValueError:
        log.debug("classification_plain_request")
        state = {}
    task_ids = [task["id"] for task in state.get("saved_tasks", [])] if isinstance(state, dict) else []
    if task_ids:
        response_format = deepcopy(response_format)
        schema = response_format["json_schema"]["schema"]
        schema["properties"]["resume_task"] = {"type": "string", "enum": ["", *task_ids]}
        schema["required"].append("resume_task")
        schema["properties"]["task_scope"] = {"type": "string", "enum": ["task", "step", "inspect", "amend"]}
        schema["required"].append("task_scope")
    names = ()
    if file_roles is not None:
        from rune.agent.provenance import role_hint_names
        request = json.loads(content).get("request_to_classify", "")
        names = tuple(name for name in role_hint_names(request) if name not in file_roles)
        if len(names) > 12:
            names = ()
        if names:
            response_format = deepcopy(response_format)
            schema = response_format["json_schema"]["schema"]
            schema["properties"]["file_roles"] = {"type": "array", "items": {
                "type": "object", "properties": {
                    "name": {"type": "string", "enum": list(names)},
                    "role": {"type": "string", "enum": ["input", "output", "preserve"]},
                }, "required": ["name", "role"], "additionalProperties": False,
            }}
            schema["required"].append("file_roles")
            system += (
                "\nAlso classify the supplied file names by their role in the current request: "
                "input=existing source to read or modify; output=new file to create; "
                "preserve=existing file explicitly left unchanged. Omit uncertain roles. "
                "When a file must be read and also left unchanged, use input; reading does not authorize edits. "
                "Use each name once. Names: " + json.dumps(list(names), ensure_ascii=False)
            )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    from rune.llm.failures import request_failure

    deadline = deadline if deadline is not None else asyncio.get_running_loop().time() + ROUTING_TIMEOUT
    async with asyncio.timeout_at(deadline):
        control = await asyncio.to_thread(reasoning_control, f"{selected.provider}/{selected.model}")
        # Routing uses the smallest supported reasoning budget.
        effort = next((level for level in ("none", "minimal", "low") if level in control.efforts), None)
        for attempt in range(2):
            try:
                response = await client.completion(
                    messages=messages, model=selected.model, provider=selected.provider, max_tokens=1024 if attempt == 0 else 2048,
                    timeout=min(ROUTING_TIMEOUT, max(0.01, deadline - asyncio.get_running_loop().time())),
                    response_format=response_format, cache_system=True, max_retries=0,
                    **({"reasoning_effort": effort} if effort is not None else {}),
                )
                data = decode_object(response)
                roles = data.pop("file_roles", []) if names else []
                browser_plan = data.get("browser_requirements")
                values = validate_decision({**data, "browser_requirements": None})
                if browser_plan is not None:
                    try:
                        validated(browser_plan, tuple(
                            str(state.get(key, "")) for key in ("request_to_classify", "previous_request")))
                        values["browser_requirements"] = browser_plan
                    except ValueError as exc:
                        # The executor can bind a new plan before input without repeating routing.
                        log.info("browser_plan_deferred", error=type(exc).__name__)
                if task_ids and values.get("resume_task") not in ["", *task_ids]:
                    raise InvalidClassification("invalid_resume_task")
                if not task_ids and values.get("resume_task"):
                    raise InvalidClassification("unexpected_resume_task")
                if names and isinstance(roles, list):
                    proposed = {}
                    for item in roles:
                        if (not isinstance(item, dict) or set(item) != {"name", "role"}
                                or not isinstance(item["name"], str) or not isinstance(item["role"], str)
                                or item["name"] not in names or item["role"] not in {"input", "output", "preserve"}
                                or item["name"] in proposed):
                            proposed = {}
                            break
                        proposed[item["name"]] = item["role"]
                    file_roles.update(proposed)
                return values
            except InvalidClassification as exc:
                log.warning("classification_response_rejected", attempt=attempt + 1, reason=str(exc))
                if str(exc) == "refusal" or attempt:
                    raise
                messages[0] = {"role": "system", "content": system + (
                    f"\nThe previous routing response was rejected: {exc}. Return only the schema object. "
                    "Do not perform the request, propose code, or write the user's final answer."
                )}
            except Exception as exc:
                failure = request_failure(exc)
                delay = failure.retry_after if failure.retry_after is not None else random.uniform(0.25, 0.75)
                remaining = deadline - asyncio.get_running_loop().time()
                if attempt or not failure.retryable or not math.isfinite(delay) or delay + 1 >= remaining:
                    raise
                log.info("classification_request_retry", **failure.to_dict(), delay_seconds=round(delay, 3))
                await asyncio.sleep(delay)
    raise InvalidClassification("no_decision")
