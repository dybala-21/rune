"""Intra-step observation masking for the LiteLLM adapter.

Within one agent step the tool-call loop appends each tool result to the
message list and re-sends the whole list on every round. A few large tool
outputs (for example a full compiler run) make this grow super-linearly and
can exhaust the per-iteration token budget before the step finishes.

This returns a masked copy for the wire: the most recent tool results stay
full so the agent still acts on current data; older tool results are
head+tail truncated. Non-tool messages are untouched, and no message is
dropped or reordered, so tool_call/tool_result pairing stays valid. Masking
only activates once the assembled content is large, so normal runs are
returned unchanged.
"""

from __future__ import annotations

from typing import Any


def head_tail(text: str, limit: int, *, note: str = "") -> str:
    """Bound *text* to ~*limit* chars, keeping the head (2/3) and the tail
    (1/3) - the tail holds error summaries. *note* appends context-specific
    guidance to the elision marker. Returns *text* unchanged when it already
    fits. Shared by obs_cap / goal_loop / goal_runtime so the head:tail ratio
    and marker stay consistent."""
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    extra = f" - {note}" if note else ""
    return (
        f"{text[:head]}\n[... {len(text)} chars elided{extra} ...]\n{text[-tail:]}"
    )


def _content_len(m: dict[str, Any]) -> int:
    c = m.get("content")
    return len(c) if isinstance(c, str) else 0


def mask_stale_tool_messages(
    messages: list[dict[str, Any]],
    *,
    keep_last: int = 2,
    trunc: int = 3072,
    activate_over: int = 80_000,
) -> list[dict[str, Any]]:
    """Return ``messages`` unchanged when total content is small; otherwise a
    copy where older ``role == "tool"`` contents are head+tail truncated. The
    last ``keep_last`` tool results are kept full. Order, count and message
    ids are preserved so tool_call/tool_result pairing stays valid.
    """
    if keep_last < 0 or trunc <= 0:
        return messages
    total = sum(_content_len(m) for m in messages)
    if total <= activate_over:
        return messages

    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_positions) <= keep_last:
        return messages
    keep = set(tool_positions[-keep_last:]) if keep_last else set()

    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if (
            m.get("role") == "tool"
            and i not in keep
            and isinstance(content, str)
            and len(content) > trunc
        ):
            masked = dict(m)
            masked["content"] = head_tail(
                content,
                trunc,
                note="re-run a narrower command if you still need the rest",
            )
            out.append(masked)
        else:
            out.append(m)
    return out
