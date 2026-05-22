"""Shared message utilities for turn-atomic grouping and pair validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Turn:
    """An atomic conversation turn (assistant message + tool results)."""

    messages: list[Any]
    role: str
    token_estimate: int = 0


def msg_role(msg: Any) -> str:
    """Extract the role string from a message (dict or object)."""
    if isinstance(msg, dict):
        role = msg.get("role", "unknown")
    else:
        role = getattr(msg, "role", "unknown")
    return role if isinstance(role, str) else "unknown"


def group_into_turns(messages: list[Any]) -> list[Turn]:
    """Group messages into atomic turns.

    Each assistant message is paired with all immediately following
    tool-result messages. Other roles are their own turn.
    """
    turns: list[Turn] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg_role(msg)

        if role == "assistant":
            turn_msgs: list[Any] = [msg]
            j = i + 1
            while j < len(messages) and msg_role(messages[j]) == "tool":
                turn_msgs.append(messages[j])
                j += 1
            turns.append(Turn(messages=turn_msgs, role="assistant"))
            i = j
        else:
            turns.append(Turn(messages=[msg], role=role))
            i += 1

    return turns


def _assistant_tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, str):
        return tool_call
    if isinstance(tool_call, dict):
        value = tool_call.get("id") or tool_call.get("tool_call_id")
        if isinstance(value, str) and value:
            return value
        function = tool_call.get("function")
        if isinstance(function, dict):
            value = function.get("id")
            if isinstance(value, str) and value:
                return value
    value = getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", None)
    return value if isinstance(value, str) and value else None


def _message_tool_call_id(msg: Any) -> str | None:
    if isinstance(msg, dict):
        value = msg.get("tool_call_id")
    else:
        value = getattr(msg, "tool_call_id", None)
    return value if isinstance(value, str) and value else None


def _assistant_tool_call_ids(msg: Any) -> list[str]:
    if isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    else:
        tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return []
    ids: list[str] = []
    for tool_call in tool_calls:
        tool_call_id = _assistant_tool_call_id(tool_call)
        if tool_call_id:
            ids.append(tool_call_id)
    return ids


def validate_tool_pairs(messages: list[Any]) -> list[Any]:
    """Return messages with OpenAI-valid assistant/tool pair structure.

    Tool messages must immediately follow the assistant message that declared
    their ``tool_call_id``. When context trimming or policy nudges split a
    multi-tool assistant turn, synthesize an explicit omitted-result message so
    providers reject neither the payload nor the later recovery turn.
    """
    result: list[Any] = []
    synthetic = 0
    dropped = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg_role(msg)

        if role == "assistant":
            expected = _assistant_tool_call_ids(msg)
            result.append(msg)
            if not expected:
                i += 1
                continue

            expected_set = set(expected)
            seen: set[str] = set()
            j = i + 1
            while j < len(messages) and msg_role(messages[j]) == "tool":
                tool_call_id = _message_tool_call_id(messages[j])
                if tool_call_id in expected_set and tool_call_id not in seen:
                    result.append(messages[j])
                    seen.add(tool_call_id)
                else:
                    dropped += 1
                j += 1

            for tool_call_id in expected:
                if tool_call_id in seen:
                    continue
                result.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": (
                        "[Tool result omitted during context compaction; "
                        "rerun the tool if the result is still needed.]"
                    ),
                })
                synthetic += 1

            i = j
            continue

        if role == "tool":
            dropped += 1
        else:
            result.append(msg)
        i += 1

    if synthetic:
        log.warning("missing_tool_messages_synthesized", count=synthetic)
    if dropped:
        log.warning("orphaned_tool_messages_dropped", count=dropped)
    return result
