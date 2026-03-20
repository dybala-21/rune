"""Conversation storage for RUNE.

SQLite-backed persistence for conversations with full CRUD
and digest generation.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from rune.conversation.types import Conversation, ConversationTurn
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _ensure_db(db_path: Path) -> sqlite3.Connection:
    """Create the DB and tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            execution_context TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            digest TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT '',
            episode_id TEXT NOT NULL DEFAULT '',
            execution_context TEXT NOT NULL DEFAULT '',
            archived INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL,
            tool_calls TEXT NOT NULL DEFAULT '[]',
            created_order INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_turns_conv
        ON turns(conversation_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_user
        ON conversations(user_id, updated_at DESC)
    """)
    conn.commit()
    return conn


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class ConversationStore:
    """SQLite-backed conversation storage."""

    __slots__ = ("_db_path", "_conn")

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn = _ensure_db(self._db_path)

    async def save(self, conversation: Conversation) -> None:
        """Save or upsert a conversation and all its turns."""
        conn = self._conn

        conn.execute(
            """
            INSERT OR REPLACE INTO conversations
                (id, user_id, title, status, execution_context, created_at, updated_at, digest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation.id,
                conversation.user_id,
                conversation.title,
                conversation.status,
                conversation.execution_context,
                _dt_to_str(conversation.created_at),
                _dt_to_str(conversation.updated_at),
                conversation.digest,
            ),
        )

        # Delete existing turns and re-insert (simple upsert strategy)
        conn.execute(
            "DELETE FROM turns WHERE conversation_id = ?",
            (conversation.id,),
        )
        for order, turn in enumerate(conversation.turns):
            conn.execute(
                """
                INSERT INTO turns (id, conversation_id, role, content, channel, episode_id,
                    execution_context, archived, timestamp, tool_calls, created_order)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex[:16],
                    conversation.id,
                    turn.role,
                    turn.content,
                    turn.channel,
                    turn.episode_id,
                    turn.execution_context,
                    int(turn.archived),
                    _dt_to_str(turn.timestamp),
                    json_encode(turn.tool_calls),
                    order,
                ),
            )

        conn.commit()
        log.debug("conversation_saved", id=conversation.id)

    async def load(self, conversation_id: str) -> Conversation | None:
        """Load a conversation by ID."""
        row = self._conn.execute(
            "SELECT id, user_id, title, status, execution_context, created_at, updated_at, digest "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if row is None:
            return None

        turns = self._load_turns(conversation_id)

        return Conversation(
            id=row[0],
            user_id=row[1],
            title=row[2],
            turns=turns,
            created_at=_str_to_dt(row[5]),
            updated_at=_str_to_dt(row[6]),
            digest=row[7],
            status=row[3],
            execution_context=row[4],
        )

    async def list(self, user_id: str, limit: int = 20, status: str | None = None) -> list[Conversation]:
        """List conversations for a user, most recent first (without turns)."""
        if status:
            rows = self._conn.execute(
                "SELECT id, user_id, title, status, execution_context, created_at, updated_at, digest "
                "FROM conversations WHERE user_id = ? AND status = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, user_id, title, status, execution_context, created_at, updated_at, digest "
                "FROM conversations WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()

        return [
            Conversation(
                id=r[0],
                user_id=r[1],
                title=r[2],
                created_at=_str_to_dt(r[5]),
                updated_at=_str_to_dt(r[6]),
                digest=r[7],
                status=r[3],
                execution_context=r[4],
            )
            for r in rows
        ]

    async def delete(self, conversation_id: str) -> None:
        """Delete a conversation and its turns."""
        self._conn.execute(
            "DELETE FROM turns WHERE conversation_id = ?",
            (conversation_id,),
        )
        self._conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        self._conn.commit()
        log.debug("conversation_deleted", id=conversation_id)

    async def get_recent_digests(
        self, user_id: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get recent conversation digests for context priming."""
        rows = self._conn.execute(
            "SELECT id, title, digest, updated_at "
            "FROM conversations WHERE user_id = ? AND digest != '' "
            "ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

        return [
            {
                "id": r[0],
                "title": r[1],
                "digest": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]

    def _generate_digest(self, conversation: Conversation) -> str:
        """Generate a short digest summarizing the conversation.

        Uses a simple heuristic: extracts the first user message as topic
        and counts turns/tool calls.
        """
        if not conversation.turns:
            return ""

        first_user = next(
            (t.content[:200] for t in conversation.turns if t.role == "user"),
            "",
        )
        total_turns = len(conversation.turns)
        tool_calls = sum(len(t.tool_calls) for t in conversation.turns)

        parts = []
        if first_user:
            parts.append(f"Topic: {first_user}")
        parts.append(f"{total_turns} turns")
        if tool_calls:
            parts.append(f"{tool_calls} tool calls")

        return " | ".join(parts)

    def _load_turns(self, conversation_id: str) -> list[ConversationTurn]:
        rows = self._conn.execute(
            "SELECT role, content, channel, episode_id, execution_context, "
            "archived, timestamp, tool_calls "
            "FROM turns WHERE conversation_id = ? ORDER BY created_order",
            (conversation_id,),
        ).fetchall()

        return [
            ConversationTurn(
                role=r[0],
                content=r[1],
                timestamp=_str_to_dt(r[6]),
                tool_calls=json_decode(r[7]),
                channel=r[2],
                episode_id=r[3],
                execution_context=r[4],
                archived=bool(r[5]),
            )
            for r in rows
        ]

    # ----- New methods (ported from TS) ------------------------------------

    async def get_turn_count(self, conversation_id: str) -> int:
        """Return the number of turns in a conversation."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return row[0] if row else 0

    async def get_last_turn(self, conversation_id: str) -> ConversationTurn | None:
        """Return the most recent turn in a conversation."""
        row = self._conn.execute(
            "SELECT role, content, channel, episode_id, execution_context, "
            "archived, timestamp, tool_calls "
            "FROM turns WHERE conversation_id = ? ORDER BY created_order DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return ConversationTurn(
            role=row[0],
            content=row[1],
            timestamp=_str_to_dt(row[6]),
            tool_calls=json_decode(row[7]),
            channel=row[2],
            episode_id=row[3],
            execution_context=row[4],
            archived=bool(row[5]),
        )

    async def replace_turns_with_summary(
        self, conversation_id: str, summary: str, *, keep_latest: int = 5,
    ) -> int:
        """Replace older turns with a summary turn, keeping the latest N turns.

        Returns the number of turns replaced.
        """
        # Get IDs of all turns
        all_ids = self._conn.execute(
            "SELECT id FROM turns WHERE conversation_id = ? ORDER BY created_order",
            (conversation_id,),
        ).fetchall()

        if len(all_ids) <= keep_latest:
            return 0

        # IDs to remove (older turns)
        ids_to_remove = [r[0] for r in all_ids[:-keep_latest]]
        placeholders = ",".join("?" * len(ids_to_remove))

        self._conn.execute(
            f"DELETE FROM turns WHERE id IN ({placeholders})",
            tuple(ids_to_remove),
        )

        # Insert summary turn at the beginning (created_order = -1 to sort before kept turns)
        now = datetime.now()
        self._conn.execute(
            """
            INSERT INTO turns (id, conversation_id, role, content, channel, episode_id,
                execution_context, archived, timestamp, tool_calls, created_order)
            VALUES (?, ?, 'system', ?, '', '', '', 0, ?, '[]', -1)
            """,
            (uuid4().hex[:16], conversation_id, f"[Summary of {len(ids_to_remove)} earlier turns]\n{summary}", _dt_to_str(now)),
        )
        self._conn.commit()
        return len(ids_to_remove)

    async def archive_stale(
        self, idle_minutes: int = 60, user_id: str | None = None,
    ) -> int:
        """Archive conversations idle for more than *idle_minutes*.

        Returns the number of conversations archived.
        """
        cutoff = datetime.now()
        # Subtract idle_minutes
        from datetime import timedelta
        cutoff_str = _dt_to_str(cutoff - timedelta(minutes=idle_minutes))

        params: list[Any] = [cutoff_str]
        user_clause = ""
        if user_id:
            user_clause = "AND user_id = ?"
            params.append(user_id)

        cursor = self._conn.execute(
            f"UPDATE conversations SET status = 'archived' "
            f"WHERE status = 'active' AND updated_at < ? {user_clause}",
            tuple(params),
        )
        affected = cursor.rowcount if hasattr(cursor, "rowcount") else 0
        if affected:
            self._conn.commit()
            log.debug("conversations_archived", count=affected)
        return affected

    async def archive_conversation(self, conversation_id: str) -> None:
        """Archive a single conversation."""
        self._conn.execute(
            "UPDATE conversations SET status = 'archived' WHERE id = ?",
            (conversation_id,),
        )
        self._conn.commit()

    async def find_active_conversation(self, user_id: str) -> Conversation | None:
        """Find the most recent active conversation for a user."""
        row = self._conn.execute(
            "SELECT id, user_id, title, status, execution_context, created_at, updated_at, digest "
            "FROM conversations WHERE user_id = ? AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return Conversation(
            id=row[0],
            user_id=row[1],
            title=row[2],
            created_at=_str_to_dt(row[5]),
            updated_at=_str_to_dt(row[6]),
            digest=row[7],
            status=row[3],
            execution_context=row[4],
        )

    async def update_conversation(
        self, conversation_id: str, **fields: Any,
    ) -> None:
        """Update specific fields on a conversation."""
        allowed = {"title", "status", "execution_context", "digest"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return

        updates["updated_at"] = _dt_to_str(datetime.now())
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [conversation_id]

        self._conn.execute(
            f"UPDATE conversations SET {set_clause} WHERE id = ?",
            tuple(values),
        )
        self._conn.commit()
