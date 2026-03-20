"""Session list and switching UI logic for RUNE TUI.

Ported from src/ui/sessions.ts - save, load, list, restore, and
prune persisted sessions.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rune.utils.fast_serde import json_decode
from rune.utils.paths import rune_data_dir

SESSIONS_DIR = rune_data_dir() / "sessions"
MAX_SESSIONS = 20


# Data types

@dataclass(slots=True)
class SerializedMessage:
    id: str
    role: str
    content: str
    timestamp: str


@dataclass(slots=True)
class SerializedToolCallBlock:
    id: str
    action: str
    observation: str
    success: bool
    timestamp: str
    capability: str = ""


@dataclass(slots=True)
class SavedSession:
    id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int
    tool_call_count: int
    messages: list[SerializedMessage] = field(default_factory=list)
    tool_call_blocks: list[SerializedToolCallBlock] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SessionListItem:
    id: str
    name: str
    updated_at: str
    message_count: int
    preview: str


# Internal helpers

def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


async def _prune_old_sessions() -> None:
    """Remove oldest sessions when count exceeds *MAX_SESSIONS*."""
    _ensure_dir()
    session_files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(session_files) > MAX_SESSIONS:
        oldest = session_files.pop(0)
        with contextlib.suppress(OSError):
            oldest.unlink()


# Public API

async def save_session(
    session_id: str,
    name: str,
    messages: list[SerializedMessage],
    tool_call_blocks: list[SerializedToolCallBlock],
) -> Path:
    """Save a session to disk and prune old ones. Returns the file path."""
    _ensure_dir()
    now = datetime.now(tz=UTC).isoformat()

    session = SavedSession(
        id=session_id,
        name=name,
        created_at=now,
        updated_at=now,
        message_count=sum(1 for m in messages if m.role in ("user", "assistant")),
        tool_call_count=len(tool_call_blocks),
        messages=messages,
        tool_call_blocks=tool_call_blocks,
    )

    fp = _session_path(session_id)
    fp.write_text(
        json.dumps(asdict(session), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    await _prune_old_sessions()
    return fp


async def list_sessions() -> list[SessionListItem]:
    """List saved sessions, most-recently-updated first."""
    _ensure_dir()
    items: list[SessionListItem] = []

    for fp in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json_decode(fp.read_text(encoding="utf-8"))
            first_user = next(
                (m["content"] for m in data.get("messages", []) if m.get("role") == "user"),
                "",
            )
            preview = first_user[:80] + ("..." if len(first_user) > 80 else "")
            items.append(
                SessionListItem(
                    id=data["id"],
                    name=data.get("name", ""),
                    updated_at=data.get("updated_at", ""),
                    message_count=data.get("message_count", 0),
                    preview=preview,
                )
            )
        except (OSError, json.JSONDecodeError, KeyError):
            continue

    return items


async def load_session(session_id: str) -> SavedSession | None:
    """Load a single session by id. Returns *None* when not found."""
    fp = _session_path(session_id)
    if not fp.exists():
        return None
    try:
        data = json_decode(fp.read_text(encoding="utf-8"))
        return SavedSession(
            id=data["id"],
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            message_count=data.get("message_count", 0),
            tool_call_count=data.get("tool_call_count", 0),
            messages=[
                SerializedMessage(**m) for m in data.get("messages", [])
            ],
            tool_call_blocks=[
                SerializedToolCallBlock(**b) for b in data.get("tool_call_blocks", [])
            ],
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


async def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns *True* on success."""
    fp = _session_path(session_id)
    try:
        fp.unlink()
        return True
    except OSError:
        return False
