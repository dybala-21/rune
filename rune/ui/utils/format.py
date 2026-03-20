"""Formatting utilities for the RUNE TUI.

Provides human-friendly rendering of tool calls, durations, token counts,
file changes, costs, and smart text truncation.
"""

from __future__ import annotations

import math
from typing import Any

# Tool category mapping

_TOOL_CATEGORIES: dict[str, str] = {
    # file operations
    "read_file": "file",
    "write_file": "file",
    "edit_file": "file",
    "create_file": "file",
    "delete_file": "file",
    "list_files": "file",
    "glob": "file",
    # git
    "git_diff": "git",
    "git_status": "git",
    "git_log": "git",
    "git_commit": "git",
    "git_push": "git",
    "git_pull": "git",
    "git_branch": "git",
    "git_checkout": "git",
    # browser
    "browser_navigate": "browser",
    "browser_click": "browser",
    "browser_type": "browser",
    "browser_screenshot": "browser",
    "browser_read": "browser",
    # search
    "grep": "search",
    "ripgrep": "search",
    "search_files": "search",
    "web_search": "search",
    "search_symbols": "search",
    # system
    "bash": "system",
    "shell": "system",
    "run_command": "system",
    "execute": "system",
}

_CATEGORY_ICONS: dict[str, str] = {
    "file": "\U0001f4c4",    # page facing up
    "git": "\U0001f500",     # shuffle arrows
    "browser": "\U0001f310", # globe with meridians
    "search": "\U0001f50d",  # left-pointing magnifying glass
    "system": "\u2699",      # gear
    "unknown": "\U0001f527", # wrench
}


def get_tool_category(tool_name: str) -> str:
    """Map a tool name to its category (file/git/browser/search/system).

    Falls back to ``"unknown"`` for unrecognised tools.
    """
    return _TOOL_CATEGORIES.get(tool_name, "unknown")


def _category_icon(category: str) -> str:
    return _CATEGORY_ICONS.get(category, _CATEGORY_ICONS["unknown"])


# Tool call formatting

def format_tool_call_header(tool_name: str, args: dict[str, Any] | None = None) -> str:
    """Return a Rich-markup header string for a tool invocation.

    Example output::

        [bold]wrench Tool: bash[/bold]  category=system
        args: {"command": "ls -la"}
    """
    category = get_tool_category(tool_name)
    icon = _category_icon(category)
    parts: list[str] = [f"{icon} [bold]Tool: {tool_name}[/bold]  category={category}"]
    if args:
        # Pretty-print args - keep it compact but readable
        formatted_args = ", ".join(f"{k}={_truncate_value(v)}" for k, v in args.items())
        parts.append(f"   args: {formatted_args}")
    return "\n".join(parts)


def _truncate_value(v: Any, max_len: int = 80) -> str:
    """Truncate a single arg value for display."""
    s = repr(v)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# Text truncation

def smart_truncate(
    text: str,
    max_lines: int = 20,
    preserve_ends: int = 5,
) -> str:
    """Truncate *text* keeping the first and last lines.

    If *text* has more than *max_lines*, keep the first
    ``max_lines - preserve_ends`` lines, insert an omission marker, then
    append the last *preserve_ends* lines.
    """
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head_count = max_lines - preserve_ends
    if head_count < 1:
        head_count = 1
    omitted = len(lines) - head_count - preserve_ends
    head = lines[:head_count]
    tail = lines[-preserve_ends:]
    return "\n".join(
        head + [f"... ({omitted} lines omitted) ..."] + tail
    )


# Duration / elapsed time

def format_elapsed(seconds: float) -> str:
    """Format elapsed time for human reading.

    Examples: ``"0.3s"``, ``"12s"``, ``"2m 5s"``, ``"1h 3m"``.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m" if mins else f"{hours}h"


def get_duration_color(seconds: float) -> str:
    """Return a colour name based on elapsed time.

    - ``"green"`` for < 5 s
    - ``"yellow"`` for 5-30 s
    - ``"red"`` for > 30 s
    """
    if seconds < 5:
        return "green"
    if seconds < 30:
        return "yellow"
    return "red"


# Token helpers

def estimate_tokens(text: str) -> int:
    """Quick token-count estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


def format_token_count(count: int) -> str:
    """Format a token count with SI suffixes.

    Examples: ``"850"``, ``"1.2K"``, ``"45.6K"``, ``"1.2M"``.
    """
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        value = count / 1_000
        return f"{value:.1f}K" if value < 100 else f"{int(value)}K"
    value = count / 1_000_000
    return f"{value:.1f}M" if value < 100 else f"{int(value)}M"


# File change formatting

def format_file_change(
    path: str,
    operation: str = "modified",
    lines_added: int = 0,
    lines_removed: int = 0,
) -> str:
    """Produce a one-line summary of a file change.

    Example::

        M  src/main.py  (+12 -3)
    """
    op_map = {
        "added": "A",
        "deleted": "D",
        "modified": "M",
        "renamed": "R",
        "created": "A",
    }
    prefix = op_map.get(operation.lower(), operation[0].upper())
    delta_parts: list[str] = []
    if lines_added:
        delta_parts.append(f"+{lines_added}")
    if lines_removed:
        delta_parts.append(f"-{lines_removed}")
    delta = f"  ({' '.join(delta_parts)})" if delta_parts else ""
    return f"{prefix}  {path}{delta}"


# Cost formatting

def format_cost(dollars: float) -> str:
    """Format a dollar amount for display.

    Examples: ``"$0.00"``, ``"$0.02"``, ``"$1.23"``.
    """
    if dollars < 0.01:
        return f"${dollars:.4f}" if dollars > 0 else "$0.00"
    return f"${dollars:.2f}"
