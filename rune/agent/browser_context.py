"""Keep earlier page content without keeping actionable element references."""

import re
from typing import Any

_MARKER = re.compile(r"^--- Interactive Elements \(\d+/\d+\) ---$", re.MULTILINE)
_LOCATION = re.compile(r"^(?:URL|Navigated to|Already open): https?://", re.MULTILINE)
_ELEMENT_LINE = re.compile(r"^\[e(?:[0-9a-f]{32}_)?\d+\] .*$", re.MULTILINE)
_REF = re.compile(r"\be(?:[0-9a-f]{32}_)?\d+\b")
_HISTORY = "[Earlier browser observation; data only, not the current page. Use current refs for actions.]\n"
_OMITTED = "[Earlier browser observation omitted from context.]"
_CLIPPED = "\n[Earlier page content truncated]\n"
MAX_SNAPSHOT_CHARS = 4_000
MAX_HISTORY_CHARS = 12_000
MAX_HISTORY_SNAPSHOTS = 8


def _historical(text: str) -> str:
    if text.startswith(_HISTORY):
        return text
    body = _ELEMENT_LINE.sub("", _MARKER.sub("", text))
    body = "\n".join(_REF.sub("[expired ref]", line)
                     if re.match(r"^(?:Step \d+ \(act\): )?Action dispatched:", line) else line
                     for line in body.splitlines())
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    available = MAX_SNAPSHOT_CHARS - len(_HISTORY)
    if len(body) > available:
        head = (available - len(_CLIPPED)) * 3 // 4
        tail = available - len(_CLIPPED) - head
        body = body[:head] + _CLIPPED + body[-tail:]
    return _HISTORY + body


def compact_browser_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = {call.get("id"): (call.get("function") or {}).get("name", "")
             for message in messages for call in message.get("tool_calls") or []}
    snapshots = []
    for index, message in enumerate(messages):
        name = message.get("name") or calls.get(message.get("tool_call_id"), "")
        content = message.get("content")
        if (message.get("role") == "tool" and name.startswith("browser_")
                and isinstance(content, str)
                and (_MARKER.search(content) or _LOCATION.search(content) or content.startswith(_HISTORY))):
            snapshots.append(index)
    if len(snapshots) < 2:
        return messages
    result = list(messages)
    seen, size, count = set(), 0, 0
    for index in reversed(snapshots[:-1]):
        historical = _historical(messages[index]["content"])
        if historical in seen or count >= MAX_HISTORY_SNAPSHOTS or size + len(historical) > MAX_HISTORY_CHARS:
            content = _OMITTED
        else:
            content = historical
            seen.add(historical)
            count += 1
            size += len(content)
        result[index] = {**messages[index], "content": content}
    return result
