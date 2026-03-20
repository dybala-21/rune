"""Conversation export for RUNE TUI.

Ported from src/ui/export.ts - export conversations as
Markdown, JSON, or HTML files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rune.utils.paths import rune_data_dir

EXPORTS_DIR = rune_data_dir() / "exports"


@dataclass(slots=True)
class ExportMessage:
    role: str
    content: str
    timestamp: str = ""


@dataclass(slots=True)
class ExportToolCall:
    action: str
    success: bool
    capability: str = ""
    params: dict[str, Any] = field(default_factory=dict)


# Markdown export

def _build_markdown(
    messages: list[ExportMessage],
    tool_calls: list[ExportToolCall],
) -> str:
    lines: list[str] = []
    now_str = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("# RUNE Conversation Export")
    lines.append(f"> Exported: {now_str}")
    lines.append("")

    user_count = sum(1 for m in messages if m.role == "user")
    success_count = sum(1 for t in tool_calls if t.success)
    file_paths = {
        str(t.params.get("path") or t.params.get("filePath") or "")
        for t in tool_calls
        if t.capability.startswith("file.") and t.success
    }
    file_paths.discard("")

    lines.append("## Summary")
    lines.append(f"- **Turns**: {user_count}")
    lines.append(f"- **Tool calls**: {len(tool_calls)} ({success_count} succeeded)")
    lines.append(f"- **Files changed**: {len(file_paths)}")
    lines.append("")

    lines.append("## Conversation")
    lines.append("")
    for msg in messages:
        if msg.role == "user":
            lines.append("### User")
            lines.append(msg.content)
            lines.append("")
        elif msg.role == "assistant":
            lines.append("### Rune")
            lines.append(msg.content)
            lines.append("")
        elif msg.role == "system":
            lines.append(f"> *System: {msg.content}*")
            lines.append("")

    if tool_calls:
        lines.append("## Tool Calls")
        lines.append("")
        lines.append("| # | Action | Status |")
        lines.append("|---|--------|--------|")
        for i, tc in enumerate(tool_calls, 1):
            status = "OK" if tc.success else "FAIL"
            action = tc.action.replace("|", "\\|")[:60]
            lines.append(f"| {i} | {action} | {status} |")
        lines.append("")

    return "\n".join(lines)


# JSON export

def _build_json(
    messages: list[ExportMessage],
    tool_calls: list[ExportToolCall],
) -> str:
    data = {
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "messages": [asdict(m) for m in messages],
        "tool_calls": [asdict(t) for t in tool_calls],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# HTML export

def _build_html(
    messages: list[ExportMessage],
    tool_calls: list[ExportToolCall],
) -> str:
    now_str = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>RUNE Conversation Export</title>",
        "<style>body{font-family:sans-serif;max-width:720px;margin:auto;padding:1em}"
        ".user{color:#7C3AED}.assistant{color:#06B6D4}.system{color:#6B7280}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:4px}</style>",
        "</head><body>",
        f"<h1>RUNE Conversation Export</h1><p>Exported: {now_str}</p>",
    ]
    for msg in messages:
        css = msg.role if msg.role in ("user", "assistant", "system") else ""
        label = msg.role.capitalize()
        parts.append(f"<div class='{css}'><strong>{label}:</strong><p>{msg.content}</p></div>")
    if tool_calls:
        parts.append("<h2>Tool Calls</h2><table><tr><th>#</th><th>Action</th><th>Status</th></tr>")
        for i, tc in enumerate(tool_calls, 1):
            status = "OK" if tc.success else "FAIL"
            parts.append(f"<tr><td>{i}</td><td>{tc.action[:60]}</td><td>{status}</td></tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


# Public API

_BUILDERS = {
    "markdown": (_build_markdown, ".md"),
    "json": (_build_json, ".json"),
    "html": (_build_html, ".html"),
}


async def export_conversation(
    messages: list[ExportMessage],
    tool_calls: list[ExportToolCall],
    *,
    fmt: str = "markdown",
) -> Path:
    """Export the conversation and return the path to the written file.

    *fmt* must be one of ``"markdown"``, ``"json"``, or ``"html"``.
    """
    builder, ext = _BUILDERS.get(fmt, (_build_markdown, ".md"))

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%S")
    file_path = EXPORTS_DIR / f"rune-{timestamp}{ext}"

    content = builder(messages, tool_calls)
    file_path.write_text(content, encoding="utf-8")
    return file_path
