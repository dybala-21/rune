"""Text sanitization for terminal safety.

Ported from src/utils/text-sanitize.ts - strips invisible control chars,
terminal mouse sequences, and CSI escapes.
"""

from __future__ import annotations

import re

# Control chars 0x00-0x1f except TAB (0x09) and LF (0x0a)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f]")

# ANSI CSI escape sequences (e.g. colors, cursor movement)
_CSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Terminal mouse sequences (SGR, X10, urxvt)
_MOUSE_SEQ = re.compile(
    r"\x1b\[<[\d;]+[mM]"   # SGR
    r"|\x1b\[M..."          # X10
    r"|\x1b\[\d+;\d+;\d+M"  # urxvt
)

# OSC (Operating System Command) sequences
_OSC_SEQ = re.compile(r"\x1b\].*?\x07|\x1b\].*?\x1b\\")


def sanitize(text: str) -> str:
    """Remove all potentially dangerous terminal sequences from *text*."""
    text = _OSC_SEQ.sub("", text)
    text = _MOUSE_SEQ.sub("", text)
    text = _CSI_ESCAPE.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    return text


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences only (preserving other content)."""
    return _CSI_ESCAPE.sub("", text)
