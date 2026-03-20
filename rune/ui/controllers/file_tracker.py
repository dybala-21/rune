"""FileTracker for RUNE TUI.

Watches a workspace directory for file changes using watchfiles,
maintains snapshots of file contents before changes, and exposes
a list of changed files.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


class FileTracker:
    """Watches a workspace directory and tracks file changes.

    Uses the ``watchfiles`` library for efficient cross-platform
    filesystem watching.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        *,
        on_change: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._workspace = Path(workspace_path).resolve()
        self._on_change = on_change
        self._task: asyncio.Task[None] | None = None
        self._changed_files: set[str] = set()
        self._snapshots: dict[str, str] = {}
        self._running = False

    # Public API ------------------------------------------------------------

    async def start(self) -> None:
        """Begin watching the workspace directory for changes."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        log.info("file_tracker_start", workspace=str(self._workspace))

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        log.info("file_tracker_stop")

    def get_changed_files(self) -> list[str]:
        """Return a list of files that have changed since tracking started."""
        return sorted(self._changed_files)

    def get_snapshot(self, path: str) -> str | None:
        """Return the content snapshot of *path* taken before the last change.

        Returns None if no snapshot exists for the given path.
        """
        return self._snapshots.get(path)

    def clear(self) -> None:
        """Clear tracked changes and snapshots."""
        self._changed_files.clear()
        self._snapshots.clear()

    # Internal --------------------------------------------------------------

    def _take_snapshot(self, path: Path) -> None:
        """Read current file content and store as a pre-change snapshot."""
        rel = str(path.relative_to(self._workspace))
        if rel in self._snapshots:
            return  # Keep the earliest snapshot
        with contextlib.suppress(OSError):
            self._snapshots[rel] = path.read_text(encoding="utf-8", errors="replace")

    async def _watch_loop(self) -> None:
        """Main watch loop using watchfiles."""
        try:
            import watchfiles
        except ImportError:
            log.warning("watchfiles_not_installed", msg="pip install watchfiles")
            return

        _IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".rune"}

        def _watch_filter(change: Any, path: str) -> bool:
            """Filter out noise directories."""
            parts = Path(path).parts
            return not any(part in _IGNORE_DIRS for part in parts)

        try:
            async for changes in watchfiles.awatch(
                self._workspace,
                watch_filter=_watch_filter,
                stop_event=asyncio.Event() if not self._running else None,
            ):
                if not self._running:
                    break

                changed_paths: list[str] = []
                for _change_type, raw_path in changes:
                    full_path = Path(raw_path)
                    try:
                        rel = str(full_path.relative_to(self._workspace))
                    except ValueError:
                        continue

                    # Take a snapshot before recording the change
                    if full_path.is_file():
                        self._take_snapshot(full_path)

                    self._changed_files.add(rel)
                    changed_paths.append(rel)

                if changed_paths and self._on_change is not None:
                    try:
                        self._on_change(changed_paths)
                    except Exception:
                        log.exception("file_tracker_callback_error")

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("file_tracker_watch_error")
