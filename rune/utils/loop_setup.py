"""Event-loop setup for RUNE.

Phase 9 optimization - installs ``uvloop`` as the default asyncio event
loop policy when available, providing a significant throughput boost on
Linux and macOS.

Call :func:`setup_event_loop` **before** any ``asyncio.run()`` or event
loop creation to ensure uvloop is active.
"""

from __future__ import annotations

import sys

_installed: bool = False


def setup_event_loop() -> bool:
    """Install uvloop as the default event loop policy if available.

    Returns ``True`` if uvloop was successfully installed, ``False``
    otherwise (missing package, unsupported platform, etc.).

    Safe to call multiple times; only the first call has an effect.
    """
    global _installed

    if _installed:
        return True

    # uvloop does not support Windows.
    if sys.platform == "win32":
        return False

    try:
        import uvloop
        uvloop.install()
        _installed = True
        return True
    except (ImportError, Exception):
        # ImportError  - package not installed
        # Exception    - any runtime failure during install()
        return False
