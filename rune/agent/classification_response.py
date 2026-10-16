"""Constrain routing responses and reject incomplete decisions before execution."""

from __future__ import annotations

import asyncio
import json
import math
import random
from copy import deepcopy
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_PROPERTIES = {
    "goal_type": {
        "type": "string", "enum": ["chat", "web", "research", "code_modify", "execution", "browser", "full", "artifact"],
        "description": "Route by the requested outcome. Reading or summarizing even an already open webpage is web. Browser requires interaction with webpage controls, not merely an open browser or use of a page-reading tool. Native app state/features use full with desktop intent.",
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Confidence as a fraction from 0 to 1, never a percentage."},
    "reason": {"type": "string", "description": "One short phrase explaining the routing decision."},
    "requires_execution": {
        "type": "boolean",
        "description": "True only for required code, script, shell or test execution. False for native app input, including calculations in Calculator, unless the user also requests code/tests to run.",
    },
    "needs_web_evidence": {
        "type": "boolean",
        "description": "External facts must be retrieved to help with this request, even when phrased casually or emotionally. False for stable knowledge, supplied facts, or an explicit request only to listen/not browse.",
    },
    "needs_current_source": {
        "type": "boolean",
        "description": "True when the answer needs current or recent observations, such as prices, weather, service status, availability or schedules. Requires opening a source page/API; indexed search excerpts are insufficient. False for link discovery, stable explanations, supplied facts or no-browsing requests.",
    },
    "lookup_only": {
        "type": "boolean",
        "description": "True when a few sourced observations answer a chat/web request, including a casual worry grounded by checking current conditions. Expressing a worry is not asking for a prediction. False when the user requests causal analysis, predictions, recommendations, professional advice, broad comparisons, or scope is unclear. Controls lighter default reasoning, not verification.",
    },
    "intent_categories": {"type": "array", "items": {"type": "string", "enum": ["coding", "email", "document", "table", "desktop", "calculation"]},
                          "description": "coding requires software work as an outcome, not helper code for a document. desktop requires native app features or unsaved state. Office/PDF files alone use document/file tools."},
    "requires_desktop_input": {"type": "boolean", "description": "True only for input in a native app required by the request. Saving or checking a file through document tools is false."},
    "is_related_to_previous": {"type": "boolean"},
    "table_output": {
        "type": "string", "enum": ["none", "csv", "xlsx"],
        "description": "CSV/XLSX deliverable aggregating existing source data. Use none for Markdown tables, code/test summaries, software that processes data, or blank templates.",
    },
    "calculation_expression": {
        "type": "string",
        "description": "When asked to evaluate explicit numeric arithmetic, copy the complete expression verbatim from the current request. Never solve, rewrite or invent it. Otherwise use an empty string, including native app tasks, identifiers and code-writing requests.",
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
    if set(data) != set(_PROPERTIES):
        raise InvalidClassification("invalid_fields")
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
    if not isinstance(data["calculation_expression"], str):
        raise InvalidClassification("invalid_calculation_expression")
    desktop = "desktop" in data["intent_categories"]
    if data["needs_current_source"] and not data["needs_web_evidence"]:
        raise InvalidClassification("inconsistent_source_requirement")
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
    # Deliverables may need calculation; only direct answers carry an expression.
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
    response_format = RESPONSE_FORMAT
    names = ()
    if file_roles is not None:
        from rune.agent.provenance import role_hint_names
        request = json.loads(content).get("request_to_classify", "")
        names = tuple(name for name in role_hint_names(request) if name not in file_roles)
        if len(names) > 12:
            names = ()
        if names:
            response_format = deepcopy(RESPONSE_FORMAT)
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
                values = validate_decision(data)
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
