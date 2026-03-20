"""Graceful process exit with cleanup callbacks.

Ported from src/utils/process-exit.ts - registers exit handlers
that run on normal exit and common signals (SIGINT, SIGTERM).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import sys
from collections.abc import Callable
from typing import Any

_callbacks: list[Callable[[], None]] = []
_installed: bool = False


def _run_callbacks() -> None:
    """Execute all registered exit callbacks (best-effort)."""
    for cb in reversed(_callbacks):
        try:
            cb()
        except Exception:  # noqa: BLE001
            pass


def _signal_handler(signum: int, _frame: Any) -> None:
    _run_callbacks()
    sys.exit(128 + signum)


def _ensure_installed() -> None:
    global _installed
    if _installed:
        return

    atexit.register(_run_callbacks)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            # Signal registration can fail in non-main threads
            pass

    _installed = True


def register_exit_handler(callback: Callable[[], None]) -> Callable[[], None]:
    """Register *callback* to run when the process exits.

    Returns an unregister function that removes the callback.
    """
    _ensure_installed()
    _callbacks.append(callback)

    def unregister() -> None:
        with contextlib.suppress(ValueError):
            _callbacks.remove(callback)

    return unregister


def force_exit(code: int = 1) -> None:
    """Run all exit callbacks then terminate immediately with *code*."""
    _run_callbacks()
    os._exit(code)
