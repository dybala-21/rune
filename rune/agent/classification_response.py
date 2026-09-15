"""Constrain routing responses and reject incomplete decisions before execution."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_PROPERTIES = {
    "goal_type": {"type": "string", "enum": ["chat", "web", "research", "code_modify", "execution", "browser", "full"]},
    "confidence": {"type": "number"},
    "reason": {"type": "string", "description": "One short phrase explaining the routing decision."},
    "requires_execution": {"type": "boolean"},
    "intent_categories": {"type": "array", "items": {"type": "string", "enum": ["email", "document", "table", "desktop"]}},
    "requires_desktop_input": {"type": "boolean"},
    "is_related_to_previous": {"type": "boolean"},
    "table_output": {
        "type": "string", "enum": ["none", "csv", "xlsx"],
        "description": "CSV/XLSX deliverable aggregating existing source data. Use none for Markdown tables, code/test summaries, software that processes data, or blank templates.",
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
    return data


async def request_classification(client: Any, system: str, content: str) -> dict[str, Any]:
    from rune.llm.model_selection import get_effective_model_selection

    selected = get_effective_model_selection()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    async with asyncio.timeout(35):
        for attempt in range(2):
            try:
                response = await client.completion(
                    messages=messages, model=selected.model, provider=selected.provider, max_tokens=1024 if attempt == 0 else 2048,
                    timeout=15.0 if attempt == 0 else 20.0, response_format=RESPONSE_FORMAT, cache_system=True,
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
    raise InvalidClassification("no_decision")
