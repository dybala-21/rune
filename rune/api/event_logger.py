"""SSE event persistence to JSONL files.

Ported from src/api/event-logger.ts - async buffered writes to
per-conversation/run JSONL files under ~/.rune/data/events/.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


EVENTS_DIR = rune_home() / "data" / "events"
_SENSITIVE_DIR_MODE = 0o700
_SENSITIVE_FILE_MODE = 0o600
_FLUSH_INTERVAL_SEC = 0.2
_READ_YIELD_EVERY_LINES = 500

# Async write buffer

_write_buffer: dict[str, list[str]] = {}
_flush_task: asyncio.Task[None] | None = None
_flushing = False
_ensured_dirs: set[str] = set()


def _harden_permissions(path: str | Path, mode: int) -> None:
    """Best-effort permission hardening."""
    with contextlib.suppress(OSError):
        os.chmod(path, mode)


async def _ensure_dir(directory: str | Path) -> None:
    dir_str = str(directory)
    if dir_str in _ensured_dirs:
        return
    d = Path(dir_str)
    d.mkdir(parents=True, exist_ok=True)
    _harden_permissions(d, _SENSITIVE_DIR_MODE)
    _ensured_dirs.add(dir_str)


def _ensure_dir_sync(directory: str | Path) -> None:
    dir_str = str(directory)
    if dir_str in _ensured_dirs:
        return
    d = Path(dir_str)
    d.mkdir(parents=True, exist_ok=True)
    _harden_permissions(d, _SENSITIVE_DIR_MODE)
    _ensured_dirs.add(dir_str)


async def _flush_all() -> None:
    global _flushing
    if _flushing or not _write_buffer:
        return
    _flushing = True
    try:
        entries = list(_write_buffer.items())
        _write_buffer.clear()

        for file_path, lines in entries:
            directory = os.path.dirname(file_path)
            await _ensure_dir(directory)
            content = "".join(lines)
            # Use asyncio.to_thread for non-blocking file writes
            await asyncio.to_thread(_sync_write, file_path, content)
    except Exception:
        pass  # Logging failures must not affect agent execution
    finally:
        _flushing = False


def _sync_write(file_path: str, content: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)
    _harden_permissions(file_path, _SENSITIVE_FILE_MODE)


async def _flush_loop() -> None:
    """Periodic flush task."""
    while True:
        await asyncio.sleep(_FLUSH_INTERVAL_SEC)
        await _flush_all()


def _ensure_flush_task() -> None:
    global _flush_task
    if _flush_task is not None and not _flush_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
        _flush_task = loop.create_task(_flush_loop())
    except RuntimeError:
        pass  # No running event loop


def _flush_sync() -> None:
    """Synchronous flush for process exit."""
    if not _write_buffer:
        return
    for file_path, lines in _write_buffer.items():
        try:
            _ensure_dir_sync(os.path.dirname(file_path))
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("".join(lines))
            _harden_permissions(file_path, _SENSITIVE_FILE_MODE)
        except Exception:
            pass
    _write_buffer.clear()


atexit.register(_flush_sync)


# Public API

def append_event(
    conversation_id: str,
    run_id: str,
    entry: dict[str, Any],
) -> None:
    """Append an event to the async write buffer.

    Does not block the event loop. Events are flushed periodically.
    """
    dir_path = EVENTS_DIR / conversation_id
    file_path = str(dir_path / f"{run_id}.jsonl")
    line = json_encode(entry) + "\n"

    if file_path in _write_buffer:
        _write_buffer[file_path].append(line)
    else:
        _write_buffer[file_path] = [line]

    _ensure_flush_task()


async def flush_events() -> None:
    """Immediately flush all buffered events."""
    await _flush_all()


async def read_events(
    conversation_id: str,
    *,
    run_id: str | None = None,
    include_tools: bool = True,
    include_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Read events for a conversation from JSONL files.

    Args:
        conversation_id: The conversation/session ID.
        run_id: Optional run ID to filter by specific run file.
        include_tools: Whether to include tool-related events.
        include_thinking: Whether to include thinking events.

    Returns:
        List of event log entries.
    """
    # Flush pending writes first
    await _flush_all()

    conv_dir = EVENTS_DIR / conversation_id
    if not conv_dir.exists():
        return []

    events: list[dict[str, Any]] = []

    if run_id:
        file_path = conv_dir / f"{run_id}.jsonl"
        if file_path.exists():
            events = await asyncio.to_thread(_read_jsonl, file_path)
    else:
        # Read all .jsonl files in the conversation directory
        files = sorted(conv_dir.glob("*.jsonl"))
        for f in files:
            events.extend(await asyncio.to_thread(_read_jsonl, f))

    # Filter events
    if not include_tools:
        events = [e for e in events if not e.get("event", "").startswith("tool")]
    if not include_thinking:
        events = [e for e in events if e.get("event") != "thinking"]

    return events


def _read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json_decode(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return entries


async def list_runs(conversation_id: str) -> list[str]:
    """List run IDs that have event logs for a conversation."""
    conv_dir = EVENTS_DIR / conversation_id
    if not conv_dir.exists():
        return []
    files = sorted(conv_dir.glob("*.jsonl"))
    return [f.stem for f in files]
