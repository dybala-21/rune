"""Renderer for RUNE terminal output.

Bridges agent loop events to terminal display using Rich Console (scrollback)
and Rich Live (dynamic footer during agent runs). Matches the TS RUNE two-zone
rendering model: scrollback messages + dynamic streaming/status footer.
"""

from __future__ import annotations

import fcntl
import io
import os
import sys
import threading
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# BlockingIO wrapper - protects ALL Rich writes (including Live's thread)

class _BlockingWriteWrapper(io.TextIOWrapper if False else object):  # noqa: E712
    """Wraps stdout to ensure writes happen in blocking mode.

    prompt_toolkit sets O_NONBLOCK on stdout. Rich Live's internal
    _RefreshThread writes directly to the console's file, bypassing
    _safe_print(). This wrapper intercepts write/flush at the file
    level so every Rich write is protected.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._lock = threading.Lock()

    def _ensure_blocking(self) -> int | None:
        """If O_NONBLOCK is set, clear it and return original flags."""
        try:
            fd = self._wrapped.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            if flags & os.O_NONBLOCK:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                return flags
        except (OSError, ValueError):
            pass
        return None

    def _restore_flags(self, flags: int) -> None:
        try:
            fd = self._wrapped.fileno()
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except (OSError, ValueError):
            pass

    def write(self, s: str) -> int:
        with self._lock:
            saved = self._ensure_blocking()
            try:
                return self._wrapped.write(s)
            except BlockingIOError:
                # Retry once after ensuring blocking
                saved2 = self._ensure_blocking()
                try:
                    return self._wrapped.write(s)
                finally:
                    if saved2 is not None:
                        self._restore_flags(saved2)
            finally:
                if saved is not None:
                    self._restore_flags(saved)

    def flush(self) -> None:
        with self._lock:
            saved = self._ensure_blocking()
            try:
                self._wrapped.flush()
            finally:
                if saved is not None:
                    self._restore_flags(saved)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

from rune.ui.message_format import (
    format_assistant_response,
    format_separator,
    format_thinking,
    format_tool_call,
    format_tool_result,
    format_user_message,
)
from rune.ui.theme import format_duration
from rune.ui.utils.output_styles import OutputStyleName, get_style

# Constants

_STREAMING_CURSOR = "\u2588"  # █ block cursor
_MAX_STREAMING_LINES = 24
# ASCII spinner for predictable rendering across terminal fonts.
_SPINNER_CHARS = "|/-\\"
_SPINNER_INTERVAL = 0.25

_GOLDEN = "#D4A017"
_MUTED = "#555555"
_GREEN = "#98C379"
_YELLOW = "#E5C07B"
_ORANGE = "#E67E22"
_RED = "#E06C75"
_ASSISTANT_COLOR = "#00CED1"
_ASSISTANT_TEXT = "#E0E0E0"
_DIM = "#444444"

_CTX_BAR_WIDTH = 12


# Live display renderable

class _LiveDisplay:
    """Rich renderable for the dynamic footer during agent runs.

    Shows streaming text (up to 24 lines) + status bar in a single renderable
    that Rich Live redraws in place.
    """

    def __init__(self) -> None:
        self.streaming_text: str = ""
        self.status_text: str = "Streaming"
        self.spinner_index: int = 0
        self.run_start: float = 0.0
        self.current_step: int = 0
        self.tokens_used: int = 0
        self.token_budget: int = 100_000

    def __rich_console__(self, console: Any, options: Any) -> Any:
        # Streaming text area - bordered panel
        if self.streaming_text:
            lines = self.streaming_text.split("\n")
            if len(lines) > _MAX_STREAMING_LINES:
                truncated = len(lines) - _MAX_STREAMING_LINES
                visible = "\n".join(lines[-_MAX_STREAMING_LINES:])
                prefix_text = Text(f"  ⋯ {truncated} lines above", style=_DIM)
                yield prefix_text
            else:
                visible = self.streaming_text

            streaming = Text()
            streaming.append(visible, style=_ASSISTANT_TEXT)
            streaming.append(_STREAMING_CURSOR, style=f"bold {_GOLDEN}")
            yield Panel(
                streaming,
                title="[bold #00CED1]rune[/bold #00CED1]",
                title_align="left",
                border_style="#2A6B6E",
                padding=(0, 1),
                expand=True,
            )

        # Status bar
        yield self._status_line()

    def _status_line(self) -> Text:
        """Build the running status bar with visual context gauge."""
        spinner_char = _SPINNER_CHARS[self.spinner_index % len(_SPINNER_CHARS)]
        elapsed_ms = (time.monotonic() - self.run_start) * 1000 if self.run_start else 0
        elapsed = format_duration(elapsed_ms)

        text = Text()
        text.append(f" {spinner_char} ", style=f"bold {_GOLDEN}")
        text.append(self.status_text, style=_GOLDEN)

        # Visual context gauge ████░░░░ 42%
        if self.token_budget > 0 and self.tokens_used > 0:
            pct = min(100, int(self.tokens_used * 100 / self.token_budget))
            color = _ctx_bar_color(pct)
            filled = pct * _CTX_BAR_WIDTH // 100
            empty = _CTX_BAR_WIDTH - filled
            text.append("  ", style=_MUTED)
            text.append("█" * filled, style=color)
            text.append("░" * empty, style=_DIM)
            text.append(f" {pct}%", style=color)

        # Step counter
        if self.current_step > 0:
            text.append(f"  ⟐ {self.current_step}", style=f"bold {_MUTED}")

        text.append(f"  ⏱ {elapsed}", style=_MUTED)
        text.append("  ", style=_MUTED)
        text.append("^C", style=f"bold {_GOLDEN}")
        text.append(" abort", style=_MUTED)

        return text


def _ctx_bar_color(percent: int) -> str:
    if percent >= 90:
        return _RED
    if percent >= 80:
        return _ORANGE
    if percent >= 70:
        return _YELLOW
    return _GREEN


# Idle status bar (for prompt_toolkit bottom_toolbar)

def format_idle_status(
    *,
    model: str = "gpt-5.4",
    provider: str = "openai",
    undo_count: int = 0,
    multiline: bool = False,
) -> str:
    """Build idle status bar text for prompt_toolkit bottom_toolbar."""
    left = f" {provider}:{model}"
    if multiline:
        left += " · multiline"
    if undo_count > 0:
        left += f" · ↩ {undo_count}"
    right = "^C exit · ^S style · ^J multi · /help "
    return f"{left}     {right}"


def make_safe_console(**kwargs: Any) -> Console:
    """Create a Console with a blocking-IO-safe file wrapper.

    This protects ALL Rich output (including Live's refresh thread)
    from BlockingIOError when prompt_toolkit sets O_NONBLOCK on stdout.
    """
    wrapped_file = _BlockingWriteWrapper(sys.stdout)
    return Console(file=wrapped_file, **kwargs)  # type: ignore[arg-type]


# Renderer

class Renderer:
    """Bridges agent events to terminal output.

    - Scrollback: Rich Console.print() - messages persist in terminal buffer.
    - Live footer: Rich Live - streaming text + status bar during agent runs.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._live: Live | None = None
        self._live_display = _LiveDisplay()
        self._streaming_raw_text: str = ""
        self._has_user_message: bool = False
        self._output_style: OutputStyleName = "normal"

    def _safe_print(self, *args: Any, **kwargs: Any) -> None:
        """Console.print() - safe when using make_safe_console().

        The _BlockingWriteWrapper on the console's file handles O_NONBLOCK
        at the write level, protecting both direct prints and Live thread
        refreshes. This method is kept as the single call site for prints.
        """
        self.console.print(*args, **kwargs)

    def set_output_style(self, style: OutputStyleName) -> None:
        """Set the output style for filtering display elements."""
        self._output_style = style

    # -- Scrollback printing --------------------------------------------------

    def print_user_message(self, content: str) -> None:
        """Print user message to scrollback with separator."""
        if self._has_user_message:
            self._safe_print(format_separator())
        self._has_user_message = True
        self._safe_print(format_user_message(content))

    def print_assistant_response(self, content: str) -> None:
        """Print final assistant response as rendered Markdown."""
        if content.strip():
            self._safe_print(format_assistant_response(content))

    def print_system_message(self, content: str) -> None:
        """Print system message (may contain Rich markup)."""
        # System messages from controller use Rich markup tags
        try:
            self._safe_print(Text.from_markup(f"[#666666]\u26a0[/#666666] {content}"))
        except Exception:
            self._safe_print(f"\u26a0 {content}")

    def print_proactive(
        self,
        body: str,
        intensity: str = "suggest",
    ) -> None:
        """Print a proactive suggestion as inline conversational text."""
        from rune.ui.message_format import format_proactive_suggestion
        self._safe_print(format_proactive_suggestion(
            headline="",
            body=body,
            intensity=intensity,
        ))

    def print_completion_summary(self, summary: str) -> None:
        """Print completion summary (Rich markup from controller)."""
        try:
            self._safe_print(Text.from_markup(summary))
        except Exception:
            self._safe_print(summary)

    def print_thinking(self, content: str) -> None:
        """Print thinking message (only in verbose mode)."""
        style = get_style(self._output_style)
        if style["show_thinking"]:
            self._safe_print(format_thinking(content))

    def print_tool_call(
        self, name: str, category: str = "default", *, target: str = "",
    ) -> None:
        """Print tool call start to scrollback."""
        style = get_style(self._output_style)
        if style["show_tool_details"]:
            self._safe_print(format_tool_call(name, category, target=target))

    def print_tool_result(
        self,
        name: str,
        category: str = "default",
        *,
        success: bool = True,
        duration_ms: float = 0.0,
        target: str = "",
    ) -> None:
        """Print tool result to scrollback."""
        style = get_style(self._output_style)
        if style["show_tool_details"]:
            self._safe_print(
                format_tool_result(name, category, success=success, duration_ms=duration_ms, target=target)
            )
        elif not success:
            # Always show failures even in compact mode
            self._safe_print(
                format_tool_result(name, category, success=success, duration_ms=duration_ms, target=target)
            )

    def clear_messages(self) -> None:
        """Reset state (messages are in terminal scrollback, can't truly clear)."""
        self._has_user_message = False
        self._streaming_raw_text = ""
        # Print a visual separator
        self._safe_print()

    # -- Streaming (Live display) ---------------------------------------------

    def start_live(self) -> Live:
        """Start the Rich Live display for agent runs. Returns the Live context."""
        self._live_display = _LiveDisplay()
        self._live_display.run_start = time.monotonic()
        self._streaming_raw_text = ""
        self._live = Live(
            self._live_display,
            console=self.console,
            refresh_per_second=4,
            transient=True,  # clear live display when done
        )
        return self._live

    def stop_live(self) -> None:
        """Stop the live display."""
        self._live = None

    def update_streaming(self, text: str) -> None:
        """Update the streaming text in the live display."""
        self._streaming_raw_text = text
        self._live_display.streaming_text = text
        if self._live:
            self._live.update(self._live_display)

    def finish_streaming(self) -> None:
        """Flush streaming text to scrollback as rendered Markdown."""
        if self._streaming_raw_text.strip():
            self.print_assistant_response(self._streaming_raw_text)
        self._streaming_raw_text = ""
        self._live_display.streaming_text = ""
        if self._live:
            self._live.update(self._live_display)

    def get_streaming_text(self) -> str:
        """Get the current raw streaming text."""
        return self._streaming_raw_text

    # -- Status updates -------------------------------------------------------

    def update_status(
        self,
        *,
        status_text: str | None = None,
        current_step: int | None = None,
        tokens_used: int | None = None,
        token_budget: int | None = None,
    ) -> None:
        """Update the live display status bar."""
        if status_text is not None:
            self._live_display.status_text = status_text
        if current_step is not None:
            self._live_display.current_step = current_step
        if tokens_used is not None:
            self._live_display.tokens_used = tokens_used
        if token_budget is not None:
            self._live_display.token_budget = token_budget
        if self._live:
            self._live.update(self._live_display)

    def tick_spinner(self) -> None:
        """Advance the spinner animation."""
        self._live_display.spinner_index += 1
        if self._live:
            self._live.update(self._live_display)

    # -- Orchestration progress -----------------------------------------------

    _ROLE_COLORS: dict[str, str] = {
        "researcher": "cyan",
        "planner": "yellow",
        "executor": "green",
        "communicator": "magenta",
    }

    def _build_orchestration_table(
        self, title: str,
    ) -> Any:
        """Create a new orchestration progress table."""
        from rich.box import ROUNDED
        from rich.table import Table

        tbl = Table(
            title=title,
            title_style="bold",
            box=ROUNDED,
            show_header=False,
            padding=(0, 1),
            expand=False,
        )
        tbl.add_column(width=4, justify="center")   # status icon
        tbl.add_column(min_width=32)                 # task description
        tbl.add_column(width=14, justify="right")    # role (color-coded)
        return tbl

    def print_orchestration_started(self, task_count: int, description: str = "") -> None:
        """Store plan metadata. The table is printed on completion."""
        self._orch_title = description or "multi-agent plan"
        self._orch_rows: list[tuple[Text, str | Text, Text | str]] = []

    def print_orchestration_task_progress(
        self,
        task_id: str,
        completed: int,
        total: int,
        *,
        success: bool,
        description: str = "",
        role: str = "",
    ) -> None:
        """Buffer a task row for the orchestration table."""
        status = Text.from_markup("[green]\u2713[/green]" if success else "[red]\u2717[/red]")
        label = description[:60] if description else task_id
        if not success:
            label = Text.from_markup(f"[red]{label}[/red]")
        color = self._ROLE_COLORS.get(role, "dim")
        role_text = Text.from_markup(f"[{color}]{role}[/{color}]") if role else Text("")
        self._orch_rows.append((status, label, role_text))

    def print_orchestration_task_retry(
        self, task_id: str, failure_type: str, attempt: int, error: str,
        description: str = "",
    ) -> None:
        """Buffer a retry row for the orchestration table."""
        label = description[:40] if description else task_id
        status = Text.from_markup("[yellow]\u21bb[/yellow]")
        desc = Text.from_markup(f"[dim]{label} (retry #{attempt})[/dim]")
        self._orch_rows.append((status, desc, Text("")))

    def print_orchestration_completed(
        self, *, success: bool, duration_ms: float,
        completed_count: int, failed_count: int,
    ) -> None:
        """Print the full orchestration table with summary."""
        tbl = self._build_orchestration_table(self._orch_title)
        for status, desc, role in self._orch_rows:
            tbl.add_row(status, desc, role)
        self._safe_print(tbl)

        total = completed_count + failed_count
        elapsed = f"{duration_ms / 1000:.1f}s"
        if success:
            summary = f"[green]\u2713 {completed_count}/{total}[/green] [dim]\u00b7 {elapsed}[/dim]"
        elif failed_count > 0 and completed_count > 0:
            summary = f"[yellow]\u26a0 {completed_count}/{total}[/yellow] [dim]\u00b7 {elapsed}[/dim]"
        else:
            summary = f"[red]\u2717 {completed_count}/{total}[/red] [dim]\u00b7 {elapsed}[/dim]"
        self._safe_print(Text.from_markup(f"  {summary}"))

        # Cleanup
        self._orch_rows = []
        self._orch_title = ""
