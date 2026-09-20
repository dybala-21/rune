"""Constrain routing responses and reject incomplete decisions before execution."""

from __future__ import annotations

import asyncio
import json
import math
import random
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_PROPERTIES = {
    "goal_type": {"type": "string", "enum": ["chat", "web", "research", "code_modify", "execution", "browser", "full"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Confidence as a fraction from 0 to 1, never a percentage."},
    "reason": {"type": "string", "description": "One short phrase explaining the routing decision."},
    "requires_execution": {
        "type": "boolean",
        "description": "True only for required code, script, shell or test execution. False for native app input, including calculations in Calculator, unless the user also requests code/tests to run.",
    },
    "intent_categories": {"type": "array", "items": {"type": "string", "enum": ["email", "document", "table", "desktop"]}},
    "requires_desktop_input": {"type": "boolean"},
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
    data = decode_object(response)
    if set(data) != set(_PROPERTIES):
        raise InvalidClassification("invalid_fields")
    if data["goal_type"] not in _PROPERTIES["goal_type"]["enum"]:
        raise InvalidClassification("invalid_goal_type")
    if type(data["confidence"]) not in (int, float) or not math.isfinite(data["confidence"]) or not 0 <= data["confidence"] <= 1:
        raise InvalidClassification("invalid_confidence")
    if not isinstance(data["reason"], str):
        raise InvalidClassification("invalid_reason")
    for key in ("requires_execution", "requires_desktop_input", "is_related_to_previous"):
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
    return data


async def request_classification(client: Any, system: str, content: str) -> dict[str, Any]:
    from rune.llm.model_selection import get_effective_model_selection
    from rune.llm.reasoning import reasoning_control

    selected = get_effective_model_selection()
    control = reasoning_control(f"{selected.provider}/{selected.model}")
    # Routing uses the smallest supported reasoning budget.
    effort = next((level for level in ("none", "minimal", "low") if level in control.efforts), None)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    from rune.llm.failures import request_failure

    deadline = asyncio.get_running_loop().time() + ROUTING_TIMEOUT
    async with asyncio.timeout(ROUTING_TIMEOUT):
        for attempt in range(2):
            try:
                response = await client.completion(
                    messages=messages, model=selected.model, provider=selected.provider, max_tokens=1024 if attempt == 0 else 2048,
                    timeout=max(0.01, deadline - asyncio.get_running_loop().time()),
                    response_format=RESPONSE_FORMAT, cache_system=True, max_retries=0,
                    **({"reasoning_effort": effort} if effort is not None else {}),
                )
                return decode_response(response)
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
