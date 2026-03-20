"""Session history management for RUNE TUI.

Ported from src/ui/history.ts - persistent command history with
load, save, append, and navigation support.
"""

from __future__ import annotations

import json

from rune.utils.fast_serde import json_decode
from rune.utils.paths import rune_data_dir

HISTORY_FILE = rune_data_dir() / "history.json"
MAX_ENTRIES = 200


async def load_history() -> list[str]:
    """Load command history from disk.

    Returns an empty list when the file is missing or malformed.
    """
    try:
        content = HISTORY_FILE.read_text(encoding="utf-8")
        data = json_decode(content)
        if isinstance(data, list):
            return [str(e) for e in data[:MAX_ENTRIES]]
        return []
    except (OSError, json.JSONDecodeError, ValueError):
        return []


async def save_history(history: list[str]) -> None:
    """Persist command history to disk, silently ignoring write errors."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        trimmed = history[:MAX_ENTRIES]
        HISTORY_FILE.write_text(
            json.dumps(trimmed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


async def append_history(entry: str) -> None:
    """Prepend *entry* to history (deduplicates the top entry)."""
    history = await load_history()
    if not history or history[0] != entry:
        history.insert(0, entry)
        if len(history) > MAX_ENTRIES:
            history.pop()
    await save_history(history)


class HistoryNavigator:
    """Stateful cursor over command history for up/down arrow navigation."""

    def __init__(self, history: list[str] | None = None) -> None:
        self._history: list[str] = list(history) if history else []
        self._index: int = -1

    def reset(self, history: list[str] | None = None) -> None:
        if history is not None:
            self._history = list(history)
        self._index = -1

    def previous(self) -> str | None:
        """Move to the older entry. Returns *None* if at the end."""
        if not self._history:
            return None
        if self._index < len(self._history) - 1:
            self._index += 1
        return self._history[self._index]

    def next(self) -> str | None:
        """Move to the newer entry. Returns *None* if back at the start."""
        if self._index <= 0:
            self._index = -1
            return None
        self._index -= 1
        return self._history[self._index]

    @property
    def current(self) -> str | None:
        if 0 <= self._index < len(self._history):
            return self._history[self._index]
        return None
