"""Cross-platform clipboard copy for RUNE TUI.

Detects the available clipboard command (pbcopy on macOS,
xclip/xsel on Linux, clip.exe on WSL/Windows) and provides
a simple ``copy_to_clipboard`` helper.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _detect_clipboard_cmd() -> list[str] | None:
    """Return the clipboard command as an argv list, or *None*."""
    if sys.platform == "darwin":
        if shutil.which("pbcopy"):
            return ["pbcopy"]
        return None

    if sys.platform.startswith("linux"):
        # Prefer xclip, then xsel, then clip.exe (WSL)
        if shutil.which("xclip"):
            return ["xclip", "-selection", "clipboard"]
        if shutil.which("xsel"):
            return ["xsel", "--clipboard", "--input"]
        if shutil.which("clip.exe"):
            return ["clip.exe"]
        return None

    if sys.platform == "win32":
        if shutil.which("clip"):
            return ["clip"]
        return None

    return None


def clipboard_available() -> bool:
    """Return *True* when a clipboard provider is detected."""
    return _detect_clipboard_cmd() is not None


def copy_to_clipboard(text: str, *, timeout: float = 5.0) -> bool:
    """Copy *text* to the system clipboard.

    Returns *True* on success, *False* when no clipboard tool is
    available or the command fails.
    """
    cmd = _detect_clipboard_cmd()
    if cmd is None:
        return False

    try:
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            timeout=timeout,
            check=True,
            capture_output=True,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
