"""Inline arrow-key selector for RUNE TUI.

Pure ANSI escape rendering - no Rich - to avoid cursor positioning conflicts.
↑↓ or j/k to navigate, Enter to confirm, Esc/q/Ctrl+C to cancel.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import sys
import termios
import tty

# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GOLD = "\033[38;2;212;160;23m"       # #D4A017
_CYAN = "\033[38;2;0;206;209m"        # #00CED1
_GREEN = "\033[38;2;152;195;121m"     # #98C379
_WHITE = "\033[38;2;224;224;224m"     # #E0E0E0
_MUTED = "\033[38;2;85;85;85m"       # #555555
_BORDER = "\033[38;2;60;60;60m"       # #3C3C3C
_SEL_BG = "\033[48;2;31;61;63m"       # #1F3D3F background
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE = "\033[2K\r"
_MOVE_UP = "\033[A"


class _BlockingStdout:
    """Context manager that ensures stdout stays in blocking mode.

    prompt_toolkit sets O_NONBLOCK on stdout which causes BlockingIOError
    for raw ANSI writes. This clears it on entry and restores on exit.
    """

    def __init__(self) -> None:
        self._saved_flags: int | None = None
        self._fd: int | None = None

    def __enter__(self) -> _BlockingStdout:
        try:
            self._fd = sys.stdout.fileno()
            self._saved_flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
            if self._saved_flags & os.O_NONBLOCK:
                fcntl.fcntl(self._fd, fcntl.F_SETFL, self._saved_flags & ~os.O_NONBLOCK)
        except (OSError, ValueError):
            self._fd = None
            self._saved_flags = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None and self._saved_flags is not None:
            with contextlib.suppress(OSError, ValueError):
                fcntl.fcntl(self._fd, fcntl.F_SETFL, self._saved_flags)


def inline_select(
    items: list[tuple[str, str]],
    *,
    title: str = "",
    default_index: int = 0,
) -> int | None:
    """Show an inline arrow-key selector. Returns selected index or None.

    *items* is a list of (value, display_label) tuples.
    All rendering uses raw stdout + ANSI escapes (no Rich).
    """
    if not items:
        return None

    with _BlockingStdout():
        return _inline_select_impl(items, title=title, default_index=default_index)


_MAX_VISIBLE = 15  # Max items visible at once (scrollable window)


def _inline_select_impl(
    items: list[tuple[str, str]],
    *,
    title: str,
    default_index: int,
) -> int | None:
    selected = max(0, min(default_index, len(items) - 1))
    count = len(items)
    visible = min(count, _MAX_VISIBLE)
    out = sys.stdout

    # Strip Rich markup from title for plain output
    plain_title = title
    for tag in ("[bold]", "[/bold]", "[dim]", "[/dim]", "[bold #D4A017]", "[/bold #D4A017]"):
        plain_title = plain_title.replace(tag, "")

    # Calculate max label width for clean alignment
    max_label = max(len(label) for _, label in items)
    box_width = max(max_label + 6, 48)

    # Total lines = visible items + 2 (borders)
    total_lines = visible + 2

    def _render(first_draw: bool = False) -> None:
        # Build entire frame in a single buffer, then write once.
        # Batching all ANSI escapes into one write() prevents partial
        # cursor movement processing in some terminal emulators.
        buf: list[str] = []

        if not first_draw:
            buf.append(_MOVE_UP * total_lines)

        # Calculate scroll window
        half = visible // 2
        if count <= visible or selected <= half:
            start = 0
        elif selected >= count - (visible - half):
            start = count - visible
        else:
            start = selected - half
        end = start + visible

        # Top border with scroll indicator
        scroll_hint = f" {selected + 1}/{count}" if count > visible else ""
        border_fill = box_width - len(scroll_hint)
        buf.append(f"{_CLEAR_LINE}  {_BORDER}╭{'─' * border_fill}{_MUTED}{scroll_hint}{_BORDER}╮{_RESET}\n")

        for i in range(start, end):
            _, label = items[i]
            pad = box_width - len(label) - 4
            if i == selected:
                buf.append(
                    f"{_CLEAR_LINE}  {_BORDER}│{_RESET}"
                    f"{_SEL_BG}{_WHITE}{_BOLD} ❯ {label}{' ' * max(0, pad)}  {_RESET}"
                    f"{_BORDER}│{_RESET}\n"
                )
            else:
                buf.append(
                    f"{_CLEAR_LINE}  {_BORDER}│{_RESET}"
                    f"   {_MUTED}{label}{_RESET}{' ' * max(0, pad + 1)}"
                    f"{_BORDER}│{_RESET}\n"
                )

        # Bottom border
        buf.append(f"{_CLEAR_LINE}  {_BORDER}╰{'─' * box_width}╯{_RESET}\n")

        out.write("".join(buf))
        out.flush()

    # Title line
    if plain_title:
        out.write(f"\n  {_MUTED}{plain_title}{_RESET}\n")
        out.flush()

    # Initial render
    out.write(_HIDE_CURSOR)
    _render(first_draw=True)

    # Save terminal state and enter raw mode for key reading
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    result_idx: int | None = None

    try:
        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)

            if ch == "\r" or ch == "\n":
                result_idx = selected
                break

            if ch == "\x03":  # Ctrl+C
                break

            if ch == "\x1b":  # Escape sequence
                next1 = sys.stdin.read(1)
                if next1 == "[":
                    next2 = sys.stdin.read(1)
                    if next2 == "A":  # Up
                        selected = (selected - 1) % count
                        _render()
                    elif next2 == "B":  # Down
                        selected = (selected + 1) % count
                        _render()
                else:
                    # Plain Escape - cancel
                    break
            elif ch == "k":
                selected = (selected - 1) % count
                _render()
            elif ch == "j":
                selected = (selected + 1) % count
                _render()
            elif ch == "q":
                break

    finally:
        # Always restore terminal state
        with contextlib.suppress(OSError, termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Build cleanup as a single write to prevent partial rendering
        cleanup = [_SHOW_CURSOR]
        cleanup.append(_MOVE_UP * total_lines)
        for _ in range(total_lines):
            cleanup.append(_CLEAR_LINE + "\n")
        cleanup.append(_MOVE_UP * total_lines)

        if result_idx is not None:
            _, label = items[result_idx]
            cleanup.append(f"  {_GREEN}{_BOLD}✓{_RESET} {_WHITE}{label}{_RESET}\n")
        else:
            cleanup.append(f"  {_MUTED}Cancelled.{_RESET}\n")

        out.write("".join(cleanup))
        out.flush()

    return result_idx
