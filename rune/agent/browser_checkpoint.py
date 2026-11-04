"""Replace completed read-only property exchanges with their persisted evidence state."""

import json
import re
from typing import Any

from rune.capabilities.output_prefixes import looks_like_failure_output

_URL = re.compile(r"^(?:URL|Navigated to|Already open): (https?://\S+)", re.MULTILINE)
_CONTROLS = re.compile(r"^--- Interactive Elements \(\d+/\d+\) ---$", re.MULTILINE)


def _object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def checkpoint_properties(messages: list[dict[str, Any]], progress: dict[str, Any]) -> list[dict[str, Any]]:
    """Run only with trusted collector progress from a single browser-read task.

    Keep user instructions, discovery, incomplete properties, failures and current
    controls. Remove whole assistant/tool exchanges so provider call pairing stays valid.
    """
    completed = {item["id"]: item["property_id"] for item in progress.get("candidates", [])
                 if item.get("id") and item.get("property_id") and item.get("missing") == []}
    if not completed:
        return messages
    sources = {item["id"]: item for item in progress.get("sources", []) if item.get("id")}
    complete_sources = {key for key, item in sources.items() if item.get("property_id") in completed.values()}
    complete_urls = {sources[key]["url"] for key in complete_sources if sources[key].get("url")}
    results = {m.get("tool_call_id"): i for i, m in enumerate(messages)
               if m.get("role") == "tool" and m.get("tool_call_id")}
    groups = []
    start = boundary = -1
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        calls = message["tool_calls"]
        paired = [results.get(call.get("id"), -1) for call in calls]
        if any(i <= index for i in paired):
            continue
        groups.append((index, calls, paired))
        for call, reply in zip(calls, paired, strict=True):
            if call.get("function", {}).get("name") != "browser_stay_evidence":
                continue
            payload = _object(messages[reply].get("content"))
            params = _object(call["function"].get("arguments")) or {}
            if params.get("action") == "begin" and payload is not None:
                start = reply
            if payload and payload.get("recorded") in completed and payload.get("missing") == []:
                boundary = reply
    if start < 0 or boundary <= start:
        return messages
    payload = _object(messages[boundary].get("content")) or {}
    if payload.get("checkpoint"):
        return messages
    names = {reply: call.get("function", {}).get("name", "")
             for _, calls, paired in groups for call, reply in zip(calls, paired, strict=True)}
    current_view = max((i for i, message in enumerate(messages) if names.get(i, "").startswith("browser_")
                        and names[i] != "browser_stay_evidence"
                        and isinstance(message.get("content"), str)
                        and not looks_like_failure_output(message["content"])
                        and (_URL.search(message["content"]) or _CONTROLS.search(message["content"]))), default=-1)

    def recoverable(call: dict[str, Any], reply: int) -> bool:
        content = messages[reply].get("content")
        if not isinstance(content, str) or looks_like_failure_output(content):
            return False
        function = call.get("function", {})
        name = function.get("name", "")
        if name == "browser_stay_evidence":
            saved = _object(content) or {}
            return saved.get("source") in complete_sources or saved.get("recorded") in completed
        if name not in {"browser_navigate", "browser_open", "browser_observe", "browser_act"}:
            return False
        match = _URL.search(content)
        return bool(match and match[1] in complete_urls)

    removed = set()
    for index, calls, paired in groups:
        if index <= start or any(reply >= boundary or reply == current_view for reply in paired):
            continue
        if all(recoverable(call, reply) for call, reply in zip(calls, paired, strict=True)):
            removed.update([index, *paired])
    if not removed:
        return messages
    payload.update(progress=progress, checkpoint=(
        "Earlier successful exchanges for completed properties were replaced by this saved state. "
        "Exact source text remains in browser_stay_evidence read/find. Keep completed candidates; fill only missing fields."
    ))
    return [{**message, "content": json.dumps(payload, ensure_ascii=False)} if index == boundary else message
            for index, message in enumerate(messages) if index not in removed]
