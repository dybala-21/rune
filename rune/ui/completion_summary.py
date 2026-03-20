"""Render completion summaries for RUNE TUI.

Ported from src/ui/completion-summary.ts - build narrative text
describing what the agent accomplished (files changed, commands
run, failures, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallBlock:
    """Minimal representation of a tool call used for summarisation."""

    id: str = ""
    action: str = ""
    observation: str = ""
    success: bool = True
    timestamp: str = ""
    capability: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FileChangeDetail:
    path: str
    operation: str  # "created" | "edited" | "deleted"


# Helpers

def _pluralize(count: int) -> str:
    return "" if count == 1 else "s"


def _short_path(full_path: str) -> str:
    """Return the last two path components for display."""
    parts = full_path.rsplit("/", 2)
    return "/".join(parts[-2:]) if len(parts) >= 2 else full_path


# Public API

def get_touched_files(tool_call_blocks: list[ToolCallBlock]) -> list[str]:
    """Return deduplicated short paths of files modified by successful tool calls."""
    seen: set[str] = set()
    result: list[str] = []

    for b in tool_call_blocks:
        if not b.capability or not b.success:
            continue
        if b.capability not in ("file.edit", "file.write", "file.delete"):
            continue
        raw = str(b.params.get("path") or b.params.get("filePath") or "")
        if not raw:
            continue
        short = _short_path(raw)
        if short not in seen:
            seen.add(short)
            result.append(short)

    return result


def get_file_change_details(tool_call_blocks: list[ToolCallBlock]) -> list[FileChangeDetail]:
    """Return per-file change details (created / edited / deleted)."""
    seen: dict[str, FileChangeDetail] = {}

    for b in tool_call_blocks:
        if not b.capability or not b.success:
            continue
        raw = str(b.params.get("path") or b.params.get("filePath") or "")
        if not raw:
            continue
        short = _short_path(raw)

        if b.capability == "file.write":
            if short not in seen:
                seen[short] = FileChangeDetail(path=short, operation="created")
        elif b.capability == "file.edit":
            seen[short] = FileChangeDetail(path=short, operation="edited")
        elif b.capability == "file.delete":
            seen[short] = FileChangeDetail(path=short, operation="deleted")

    return list(seen.values())


def build_completion_narrative(
    *,
    success: bool,
    iterations: int,
    tool_call_blocks: list[ToolCallBlock],
) -> str:
    """Build a one-line narrative summarising the completed run."""
    tool_count = len(tool_call_blocks)
    fail_count = sum(1 for b in tool_call_blocks if not b.success)
    touched_files = len(get_touched_files(tool_call_blocks))

    if not success:
        if tool_count > 0 and touched_files > 0:
            return (
                f"I hit a stopping point after {tool_count} tool call{_pluralize(tool_count)} "
                f"and {touched_files} file update{_pluralize(touched_files)}."
            )
        if tool_count > 0:
            fail_note = (
                f", with {fail_count} failure{_pluralize(fail_count)}" if fail_count > 0 else ""
            )
            return f"I hit a stopping point after {tool_count} tool call{_pluralize(tool_count)}{fail_note}."
        if iterations > 0:
            return f"I hit a stopping point after {iterations} step{_pluralize(iterations)}."
        return "I hit a stopping point before the run could finish cleanly."

    if tool_count > 0 and touched_files > 0:
        return (
            f"I wrapped up after {tool_count} tool call{_pluralize(tool_count)} "
            f"and {touched_files} file update{_pluralize(touched_files)}."
        )
    if tool_count > 0:
        fail_note = (
            f", recovering from {fail_count} failure{_pluralize(fail_count)}"
            if fail_count > 0
            else ""
        )
        return f"I wrapped up after {tool_count} tool call{_pluralize(tool_count)}{fail_note}."
    if iterations > 0:
        return f"I wrapped up after {iterations} step{_pluralize(iterations)}."
    return "I wrapped up the run."
