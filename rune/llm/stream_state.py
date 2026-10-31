"""Keep model continuation state attached to the response that produced it."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


def field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def provider_fields(source: Any, target: dict) -> None:
    values = field(source, "provider_specific_fields")
    if isinstance(values, dict) and values:
        target.setdefault("provider_specific_fields", {}).update(deepcopy(values))


class StreamState:
    def __init__(self, model: str = "") -> None:
        self.model = model
        self.fields: dict[str, Any] = {}
        self._reasoning: list[str] = []
        self._thinking: list[dict] = []
        self._pending: dict | None = None

    def observe(self, delta: Any) -> None:
        provider_fields(delta, self.fields)
        if isinstance(text := field(delta, "reasoning_content"), str):
            self._reasoning.append(text)
        # Responses sends complete reasoning items at the end.
        if items := field(delta, "reasoning_items"):
            self.fields["reasoning_items"] = deepcopy(items)
        for block in field(delta, "thinking_blocks") or []:
            block = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)
            if block.get("type") != "thinking":
                self._flush()
                self._thinking.append(deepcopy(block))
                continue
            if self._pending and self._pending.get("signature") and block.get("thinking"):
                self._flush()
            if self._pending is None:
                self._pending = {"type": "thinking", "thinking": "", "signature": ""}
            self._pending["thinking"] += block.get("thinking") or ""
            self._pending["signature"] += block.get("signature") or ""

    def _flush(self) -> None:
        if self._pending is not None:
            if self._pending.get("signature"):
                self._thinking.append(self._pending)
            else:
                log.debug("unsigned_thinking_not_replayed")
            self._pending = None

    def message(self, content: str = "") -> dict:
        self._flush()
        result = {"role": "assistant", **deepcopy(self.fields)}
        if self.model:
            result["_rune_state_model"] = self.model
        if content:
            result["content"] = content
        if self._thinking:
            result["thinking_blocks"] = deepcopy(self._thinking)
        elif self._reasoning and not result.get("reasoning_items"):
            result["reasoning_content"] = "".join(self._reasoning)
        metadata = result.get("provider_specific_fields", {})
        for key in ("thinking_blocks", "reasoning_content", "reasoning_items"):
            if key in result and metadata.get(key) == result[key]:
                metadata.pop(key)
        if not metadata:
            result.pop("provider_specific_fields", None)
        return result


def continuation_messages(messages: list[dict], model: str) -> list[dict]:
    result, replaced_ids = [], {}
    for message in messages:
        source = message.get("_rune_state_model")
        if source is None:
            if (call_id := message.get("tool_call_id")) in replaced_ids:
                message = {**message, "tool_call_id": replaced_ids[call_id]}
            result.append(message)
            continue
        message = {key: value for key, value in message.items() if key != "_rune_state_model"}
        if source != model:
            # Provider signatures cannot be transferred to another model.
            for key in ("reasoning_items", "thinking_blocks", "reasoning_content", "provider_specific_fields"):
                message.pop(key, None)
            if calls := message.get("tool_calls"):
                replaced_ids.update({call["id"]: call["id"].split("__thought__", 1)[0] for call in calls})
                message["tool_calls"] = [
                    {**{key: value for key, value in call.items() if key != "provider_specific_fields"},
                     "id": call["id"].split("__thought__", 1)[0],
                     "function": {key: value for key, value in call["function"].items() if key != "provider_specific_fields"}}
                    for call in calls
                ]
        result.append(message)
    return result
