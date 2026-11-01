"""Keep earlier page content without keeping actionable element references."""

import json
import re
from typing import Any
from uuid import uuid4

from rune.capabilities.output_prefixes import looks_like_failure_output

_MARKER = re.compile(r"^--- Interactive Elements \(\d+/\d+\) ---$", re.MULTILINE)
_LOCATION = re.compile(r"^(?:URL|Navigated to|Already open): https?://", re.MULTILINE)
_URL = re.compile(r"^(?:URL|Navigated to|Already open): (https?://\S+)", re.MULTILINE)
_ELEMENT_LINE = re.compile(r"^\[e(?:[0-9a-f]{32}_)?\d+\] .*$", re.MULTILINE)
_REF = re.compile(r"\be(?:[0-9a-f]{32}_)?\d+\b")
_HISTORY = "[Earlier browser observation; data only, not the current page. Use current refs for actions.]\n"
_OMITTED = "[Earlier browser observation omitted from context.]"
_CLIPPED = "\n[Earlier page content truncated]\n"
MAX_SNAPSHOT_CHARS = 4_000
MAX_HISTORY_CHARS = 12_000
MAX_HISTORY_SNAPSHOTS = 8


async def observe_followup(messages: list[dict], goal: str, classification, state: dict | None, tool) -> list[dict]:
    """Supply a fresh view before planning a related browser interaction."""
    if (classification.goal_type != "browser" or not classification.is_continuation
            or classification.is_domain_change or not classification.available
            or not state or state.get("status") != "open" or tool is None):
        return messages
    from rune.agent.attachments import content_text
    from rune.agent.tool_output import tool_content

    # Use the regular adapter so ownership, approvals, evidence and tracing still apply.
    result = await tool.function()
    seeded = list(messages)
    previous = next((m.get("content", "") for m in reversed(seeded) if m.get("role") == "user"), "")
    if content_text(previous) != goal:
        seeded.append({"role": "user", "content": goal})
    call_id = "observe_" + uuid4().hex[:12]
    seeded.extend([
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "browser_observe", "arguments": "{}",
        }}]},
        {"role": "tool", "tool_call_id": call_id, "content": tool_content(result)},
    ])
    return compact_browser_history(seeded)


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


def compact_browser_history(messages: list[dict[str, Any]], *, retrieval_progress: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if retrieval_progress:
        from rune.agent.browser_checkpoint import checkpoint_properties

        messages = checkpoint_properties(messages, retrieval_progress)
    calls = {call.get("id"): (call.get("function") or {}).get("name", "")
             for message in messages for call in message.get("tool_calls") or []}
    # Keep excerpts until a successful record makes them recoverable from the ledger.
    messages = list(messages)
    evidence_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"
                        and (m.get("name") or calls.get(m.get("tool_call_id"))) == "browser_stay_evidence"]
    parsed = {}
    for index in evidence_indexes:
        content = messages[index].get("content")
        if not isinstance(content, str):
            continue
        try:
            saved = json.loads(content)
        except ValueError:
            continue
        if not isinstance(saved, dict):
            continue
        parsed[index] = saved
    boundary = max((index for index, saved in parsed.items() if saved.get("recorded")), default=-1)
    latest_evidence = max(parsed, default=-1)
    for index, saved in parsed.items():
        if index < boundary:
            summary = {key: saved[key] for key in ("source", "url", "recorded", "missing", "captured") if key in saved}
            summary["stored"] = "Use browser_stay_evidence status/read; original evidence remains in the ledger."
        elif index < latest_evidence:
            # Superseded inventories add no evidence; keep all unrecorded excerpts.
            summary = {key: value for key, value in saved.items() if key not in {"progress", "guidance"}}
        else:
            continue
        messages[index] = {**messages[index], "content": json.dumps(summary, ensure_ascii=False)}
    # After recording, retain the latest controls and retire older UI bodies,
    # including extract/open outputs without snapshot markers.
    browser_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"
                       and (m.get("name") or calls.get(m.get("tool_call_id"), "")).startswith("browser_")
                       and i not in parsed]
    current_view = max((i for i in browser_indexes if isinstance(messages[i].get("content"), str)
                        and not looks_like_failure_output(messages[i]["content"])
                        and (_MARKER.search(messages[i]["content"]) or _LOCATION.search(messages[i]["content"]))),
                       default=-1)
    captured_indexes = {}
    for index, saved in parsed.items():
        if not saved.get("captured") or not saved.get("source"):
            continue
        previous = next((i for i in reversed(browser_indexes) if i < index), None)
        if previous is not None:
            content = messages[previous].get("content")
            match = (_URL.search(content) if isinstance(content, str)
                     and not looks_like_failure_output(content) else None)
            if match and match[1] == saved.get("url"):
                captured_indexes[previous] = saved["source"]
    if current_view in captured_indexes:
        content = messages[current_view]["content"]
        marker, location = _MARKER.search(content), _URL.search(content)
        name = messages[current_view].get("name") or calls.get(messages[current_view].get("tool_call_id"))
        if marker and location and name in {"browser_observe", "browser_navigate", "browser_open"}:
            # Preserve current controls verbatim; the preceding page body is now recoverable.
            messages[current_view] = {**messages[current_view], "content": content[:location.end()]
                + f"\n[Page body saved as source {captured_indexes[current_view]}; use read/find for exact text.]\n"
                + content[marker.start():]}
    for index in browser_indexes[:-1]:
        if index == current_view:
            continue
        content = messages[index].get("content")
        if (isinstance(content, str) and len(content) > 700 and not looks_like_failure_output(content)
                and (index < boundary or index in captured_indexes)):
            messages[index] = {**messages[index], "content": _HISTORY + content.splitlines()[0][:500]
                               + "\n[Earlier UI body stored in the evidence ledger; use status/read."
                               + (f" Source: {captured_indexes[index]}" if index in captured_indexes else "") + "]"}
    snapshots = []
    for index, message in enumerate(messages):
        name = message.get("name") or calls.get(message.get("tool_call_id"), "")
        content = message.get("content")
        if (message.get("role") == "tool" and name.startswith("browser_")
                and isinstance(content, str)
                and not looks_like_failure_output(content)
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
