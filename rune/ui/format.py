"""UI formatting utilities for RUNE TUI.

Helpers for rendering tool calls, tool results, file paths,
and truncated output in Rich markup.
"""

from __future__ import annotations

from pathlib import Path

from rune.ui.theme import COLORS, format_duration, truncate_text


def format_tool_call(name: str, params: dict[str, object]) -> str:
    """Format a tool call for display in the message list.

    Returns a Rich-markup string showing the tool name and parameters.
    """
    header = f"[bold {COLORS['warning']}]\u25b6 {name}[/bold {COLORS['warning']}]"
    if not params:
        return header

    lines = [header]
    for key, value in params.items():
        val_str = truncate_text(str(value), 120)
        lines.append(f"  [dim]{key}:[/dim] {val_str}")
    return "\n".join(lines)


def format_tool_result(
    name: str,
    result: str,
    duration_ms: float = 0.0,
    *,
    success: bool = True,
) -> str:
    """Format a tool result for display.

    Returns a Rich-markup string with success/failure indicator,
    duration, and a truncated result preview.
    """
    icon = "[bold green]\u2713[/bold green]" if success else "[bold red]\u2717[/bold red]"
    dur = f" ({format_duration(duration_ms)})" if duration_ms > 0 else ""
    header = f"{icon} [bold]{name}[/bold]{dur}"

    preview = truncate_output(result)
    if not preview.strip():
        return header

    color = COLORS["success"] if success else COLORS["error"]
    return f"{header}\n[{color}]{preview}[/{color}]"


def truncate_output(text: str, max_lines: int = 20) -> str:
    """Truncate multi-line output to at most *max_lines* lines.

    If the text exceeds the limit, the middle is replaced with an
    indicator showing how many lines were omitted.
    """
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    keep_top = max_lines // 2
    keep_bottom = max_lines - keep_top - 1
    omitted = len(lines) - keep_top - keep_bottom
    top = "\n".join(lines[:keep_top])
    bottom = "\n".join(lines[-keep_bottom:]) if keep_bottom > 0 else ""
    indicator = f"[dim]... ({omitted} lines omitted) ...[/dim]"
    parts = [top, indicator]
    if bottom:
        parts.append(bottom)
    return "\n".join(parts)


def format_file_path(path: str | Path) -> str:
    """Format a file path as a relative path with color.

    Attempts to make the path relative to the current working
    directory for brevity.
    """
    p = Path(path)
    try:
        rel = p.relative_to(Path.cwd())
    except ValueError:
        rel = p

    parent = str(rel.parent)
    name = rel.name

    if parent == ".":
        return f"[{COLORS['secondary']}]{name}[/{COLORS['secondary']}]"
    return f"[dim]{parent}/[/dim][{COLORS['secondary']}]{name}[/{COLORS['secondary']}]"
