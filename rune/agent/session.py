"""Session management for RUNE.

Ported from src/agent/session.ts - JSONL transcript with header, message,
action, observation, and compaction entries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from rune.types import TranscriptEntry, TranscriptEntryType
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)


class SessionManager:
    """Manages agent session transcripts persisted as JSONL files."""

    def __init__(self, session_dir: Path | None = None) -> None:
        self._dir = session_dir or (rune_data() / "sessions")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session_id: str | None = None
        self._file: Any = None
        self._entry_count = 0

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def start(self, goal: str = "", model: str = "") -> str:
        """Start a new session. Returns the session ID."""
        self._session_id = uuid4().hex[:16]
        self._entry_count = 0

        session_file = self._dir / f"{self._session_id}.jsonl"
        self._file = open(session_file, "a", encoding="utf-8")

        self._write_entry(TranscriptEntry(
            type=TranscriptEntryType.HEADER,
            content=json_encode({
                "session_id": self._session_id,
                "goal": goal,
                "model": model,
                "started_at": datetime.now(UTC).isoformat(),
            }),
        ))

        log.info("session_started", session_id=self._session_id)
        return self._session_id

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Record a message (user or assistant)."""
        self._write_entry(TranscriptEntry(
            type=TranscriptEntryType.MESSAGE,
            content=content,
            metadata={"role": role, **metadata},
        ))

    def add_action(self, tool: str, params: dict, **metadata: Any) -> None:
        """Record a tool call action."""
        self._write_entry(TranscriptEntry(
            type=TranscriptEntryType.ACTION,
            content=json_encode({"tool": tool, "params": params}),
            metadata=metadata,
        ))

    def add_observation(self, tool: str, result: str, **metadata: Any) -> None:
        """Record a tool call result."""
        self._write_entry(TranscriptEntry(
            type=TranscriptEntryType.OBSERVATION,
            content=result[:10_000],  # Truncate large outputs
            metadata={"tool": tool, **metadata},
        ))

    def add_compaction(self, summary: str) -> None:
        """Record a compaction point (context window management)."""
        self._write_entry(TranscriptEntry(
            type=TranscriptEntryType.COMPACTION,
            content=summary,
        ))

    def end(self) -> None:
        """End the current session."""
        if self._file:
            self._file.close()
            self._file = None
        log.info("session_ended", session_id=self._session_id, entries=self._entry_count)

    def _write_entry(self, entry: TranscriptEntry) -> None:
        if self._file is None:
            return
        record = {
            "type": entry.type,
            "content": entry.content,
            "timestamp": entry.timestamp.isoformat(),
            "metadata": entry.metadata,
        }
        self._file.write(json_encode(record) + "\n")
        self._file.flush()
        self._entry_count += 1

    def load_session(self, session_id: str) -> list[TranscriptEntry]:
        """Load a session transcript from disk."""
        session_file = self._dir / f"{session_id}.jsonl"
        if not session_file.is_file():
            return []

        entries: list[TranscriptEntry] = []
        for line in session_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                data = json_decode(line)
                entries.append(TranscriptEntry(
                    type=TranscriptEntryType(data["type"]),
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                ))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        return entries

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions."""
        sessions: list[dict[str, Any]] = []
        for f in sorted(self._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            if len(sessions) >= limit:
                break
            first_line = f.read_text().split("\n", 1)[0]
            try:
                header = json_decode(first_line)
                content = json_decode(header.get("content", "{}"))
                sessions.append({
                    "session_id": f.stem,
                    "goal": content.get("goal", ""),
                    "started_at": content.get("started_at", ""),
                    "size": f.stat().st_size,
                })
            except (json.JSONDecodeError, KeyError):
                sessions.append({"session_id": f.stem, "goal": "", "started_at": ""})

        return sessions
