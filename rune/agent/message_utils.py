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
        return msg.get("role", "unknown")
    return getattr(msg, "role", "unknown")


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


def validate_tool_pairs(messages: list[Any]) -> list[Any]:
    """Drop orphaned tool messages that lack a preceding assistant with tool_calls."""
    result: list[Any] = []
    has_pending_tool_calls = False
    dropped = 0
    for msg in messages:
        role = msg_role(msg)
        if role == "assistant":
            if isinstance(msg, dict):
                tc = msg.get("tool_calls")
            else:
                tc = getattr(msg, "tool_calls", None)
            has_pending_tool_calls = bool(tc)
        elif role == "tool":
            if not has_pending_tool_calls:
                dropped += 1
                continue
        else:
            has_pending_tool_calls = False
        result.append(msg)
    if dropped:
        log.warning("orphaned_tool_messages_dropped", count=dropped)
    return result
