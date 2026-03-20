"""Message formatting for RUNE terminal output.

Modern Rich renderables with panels, padding, and styled elements:
- user:      Styled prompt with gradient accent
- assistant: Panel-wrapped Markdown response
- system:    Subtle info bar
- tool:      Compact inline badges with category colors
- thinking:  Dim italic block
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from rune.ui.theme import format_duration

# Design tokens

_USER_COLOR = "#C0C0C0"
_USER_ACCENT = "#61AFEF"
_ASSISTANT_COLOR = "#B8860B"
_ASSISTANT_TEXT = "#D0D0D0"
_ASSISTANT_BORDER = "#333333"
_SYSTEM_COLOR = "#666666"
_THINKING_COLOR = "#7C6F9B"
_THINKING_BORDER = "#2A2A2A"
_DIM = "#333333"
_SUCCESS = "#98C379"
_ERROR = "#E06C75"
_GOLDEN = "#B8860B"
_PROACTIVE_ACCENT = "#56B6C2"  # Soft teal — distinct from golden assistant
_PROACTIVE_DIM = "#3D6B73"     # Dimmed teal for nudges

# Tool call category icons + colors
TOOL_ICONS: dict[str, tuple[str, str]] = {
    "read":     ("◇", "#00CED1"),
    "write":    ("◆", "#D4A017"),
    "exec":     ("▸", "#C678DD"),
    "browse":   ("◎", "#61AFEF"),
    "web":      ("⊕", "#56B6C2"),
    "think":    ("○", "#888888"),
    "interact": ("◈", "#56B6C2"),
    "memory":   ("▪", "#61AFEF"),
    "delegate": ("▷", "#C678DD"),
    "default":  ("▶", "#888888"),
}


# Formatters

def format_separator() -> RenderableType:
    """Render a subtle thin separator between user turns."""
    line = Text("─" * 60, style=_DIM)
    return Padding(line, (1, 0, 0, 1))


_PASTE_THRESHOLD = 4        # Lines before collapsing
_PASTE_PREVIEW_LINES = 2    # Lines to show in preview
_paste_counter = 0


def _collapse_pasted_text(content: str) -> tuple[str, list[tuple[int, str]]]:
    """Detect and collapse long pasted text blocks.

    Returns (display_text, [(paste_number, full_text), ...]).
    A pasted block is any contiguous run of lines exceeding _PASTE_THRESHOLD.
    """
    global _paste_counter  # noqa: PLW0603

    lines = content.split("\n")
    if len(lines) <= _PASTE_THRESHOLD:
        return content, []

    # Whole message is a paste
    _paste_counter += 1
    n = _paste_counter
    preview = "\n".join(lines[:_PASTE_PREVIEW_LINES])
    extra = len(lines) - _PASTE_PREVIEW_LINES
    placeholder = f"{preview}\n[Pasted text #{n} +{extra} lines]"
    return placeholder, [(n, content)]


def format_user_message(content: str) -> RenderableType:
    """Format user message with styled prompt indicator.

    Long pasted text (>4 lines) is collapsed to a preview + placeholder,
    similar to Claude Code's ``[Pasted text #1 +32 lines]`` pattern.
    """
    display, _pastes = _collapse_pasted_text(content)
    lines = display.split("\n")

    text = Text()
    text.append("❯ ", style=f"bold {_GOLDEN}")
    text.append(lines[0], style=f"{_USER_COLOR}")

    for line in lines[1:]:
        text.append("\n")
        if line.startswith("[Pasted text #"):
            text.append(f"  {line}", style=f"dim {_DIM}")
        else:
            text.append(f"  {line}", style=f"{_USER_COLOR}")

    return Padding(text, (0, 0, 0, 1))


def format_assistant_response(content: str) -> RenderableType:
    """Format assistant response in a bordered panel with Markdown."""
    body = Markdown(content)
    return Panel(
        body,
        title=f"[bold {_ASSISTANT_COLOR}]rune[/bold {_ASSISTANT_COLOR}]",
        title_align="left",
        border_style=_ASSISTANT_BORDER,
        padding=(0, 1),
        expand=True,
    )


def format_assistant_plain(content: str) -> RenderableType:
    """Format assistant text without markdown rendering."""
    text = Text(content, style=_ASSISTANT_TEXT)
    return Panel(
        text,
        title=f"[bold {_ASSISTANT_COLOR}]rune[/bold {_ASSISTANT_COLOR}]",
        title_align="left",
        border_style=_ASSISTANT_BORDER,
        padding=(0, 1),
        expand=True,
    )


def format_system_message(content: str) -> RenderableType:
    """Format system info message with subtle styling."""
    text = Text()
    text.append(" ⚠ ", style=f"bold {_GOLDEN}")
    text.append(content, style=_SYSTEM_COLOR)
    return Padding(text, (0, 0, 0, 1))


def format_proactive_suggestion(
    headline: str,
    body: str,
    intensity: str = "suggest",
) -> RenderableType:
    """Format a proactive suggestion as inline conversational text.

    Designed to feel like RUNE is talking, not issuing a system alert:
    - 💬 prefix (not ❯ or ⚠)
    - Teal color (not golden)
    - Inline text (not Panel)
    - No buttons - user responds naturally in the next prompt
    """
    text = Text()

    if intensity == "nudge":
        text.append("  💬 rune: ", style=f"dim {_PROACTIVE_DIM}")
        text.append(body, style=f"dim {_PROACTIVE_DIM}")
        return Padding(text, (0, 0, 0, 1))

    accent = _PROACTIVE_ACCENT
    if intensity == "intervene":
        accent = "#61AFEF"

    text.append("💬 rune: ", style=f"bold {accent}")
    text.append(body, style="#B0B0B0")

    return Padding(text, (1, 0, 0, 1))


def format_thinking(content: str) -> RenderableType:
    """Format thinking block with dim italic styling."""
    text = Text()
    text.append("  💭 ", style=_THINKING_COLOR)
    text.append(content, style=f"italic {_THINKING_COLOR}")
    return Panel(
        text,
        border_style=_THINKING_BORDER,
        padding=(0, 1),
        expand=True,
    )


def _extract_tool_target(name: str, params: dict[str, object] | None = None) -> str:
    """Extract a short human-readable target from tool params.

    Examples:
        file_read {file_path: "/foo/bar.py"} → "bar.py"
        bash_execute {command: "go test ./..."} → "go test ./..."
        file_search {query: "config"} → "config"
    """
    if not params:
        return ""
    # File-oriented tools: show basename
    for key in ("file_path", "path", "filepath"):
        val = params.get(key)
        if val and isinstance(val, str):
            import os
            return os.path.basename(val)
    # Command tools: show command (truncated)
    cmd = params.get("command") or params.get("cmd")
    if cmd and isinstance(cmd, str):
        short = cmd.strip().split("\n")[0]
        return short[:60] + ("…" if len(short) > 60 else "")
    # Search/query tools
    for key in ("query", "pattern", "glob", "keyword"):
        val = params.get(key)
        if val and isinstance(val, str):
            return val[:40]
    # URL tools
    url = params.get("url")
    if url and isinstance(url, str):
        return url[:50]
    return ""


def format_tool_call(
    name: str, category: str = "default", *, target: str = "",
) -> Text:
    """Format tool call with inline badge styling and target info."""
    icon, color = TOOL_ICONS.get(category, TOOL_ICONS["default"])
    text = Text()
    text.append("  ┃ ", style=_DIM)
    text.append(f" {icon} ", style=f"bold {color}")
    text.append(name, style="bold #8BADC8")
    if target:
        text.append(f" {target}", style="#888888")
    text.append(" …", style=_DIM)
    return text


def format_tool_result(
    name: str,
    category: str = "default",
    *,
    success: bool = True,
    duration_ms: float = 0.0,
    target: str = "",
) -> Text:
    """Format tool result with status indicator, target, and timing."""
    icon, color = TOOL_ICONS.get(category, TOOL_ICONS["default"])
    text = Text()
    text.append("  ┃ ", style=_DIM)
    text.append(f" {icon} ", style=f"bold {color}")
    text.append(name, style="#8BADC8")
    if target:
        text.append(f" {target}", style="#888888")
    text.append("  ")
    if success:
        text.append("✓", style=f"bold {_SUCCESS}")
    else:
        text.append("✗", style=f"bold {_ERROR}")
    if duration_ms > 0:
        text.append(f"  {format_duration(duration_ms)}", style=_DIM)
    return text


def format_completion_summary(summary: str) -> RenderableType:
    """Format completion summary with success panel."""
    try:
        inner = Text.from_markup(summary)
    except Exception:
        inner = Text(summary)
    return Padding(inner, (0, 0, 0, 1))
