"""Backward-compat shim — FileTracker moved to rune.agent.file_tracker (it is
shared by the TUI and the web/API server's /files and /undo actions).
"""

from rune.agent.file_tracker import FileTracker

__all__ = ["FileTracker"]
