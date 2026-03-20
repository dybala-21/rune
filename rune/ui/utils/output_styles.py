"""Output style definitions for RUNE TUI.

Three output styles matching TS theme.ts: compact, normal, verbose.
Controls visibility of tool details, thinking blocks, timestamps,
tool output line limits, and collapsible tool block state.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# Style types

OutputStyleName = Literal["compact", "normal", "verbose"]


class OutputStyle(TypedDict):
    """Configuration dict for an output style."""

    show_tool_details: bool
    show_thinking: bool
    show_timestamps: bool
    max_tool_output_lines: int
    collapse_tool_blocks: bool


# Style definitions

COMPACT: OutputStyle = {
    "show_tool_details": False,
    "show_thinking": False,
    "show_timestamps": False,
    "max_tool_output_lines": 3,
    "collapse_tool_blocks": True,
}

NORMAL: OutputStyle = {
    "show_tool_details": True,
    "show_thinking": False,
    "show_timestamps": True,
    "max_tool_output_lines": 20,
    "collapse_tool_blocks": False,
}

VERBOSE: OutputStyle = {
    "show_tool_details": True,
    "show_thinking": True,
    "show_timestamps": True,
    "max_tool_output_lines": 100,
    "collapse_tool_blocks": False,
}


# Lookup and cycling

_STYLES: dict[OutputStyleName, OutputStyle] = {
    "compact": COMPACT,
    "normal": NORMAL,
    "verbose": VERBOSE,
}

_CYCLE_ORDER: list[OutputStyleName] = ["compact", "normal", "verbose"]


def get_style(name: OutputStyleName) -> OutputStyle:
    """Return the output style dict for the given name."""
    return _STYLES[name]


def cycle_style(current: OutputStyleName) -> OutputStyleName:
    """Return the next style name in the cycle: compact -> normal -> verbose -> compact."""
    idx = _CYCLE_ORDER.index(current)
    return _CYCLE_ORDER[(idx + 1) % len(_CYCLE_ORDER)]
