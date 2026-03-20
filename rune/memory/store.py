"""Memory store using APSW (SQLite with WAL).

Ported from src/memory/store.ts - episodic memory, semantic facts,
safety rules, command history, proactive tables.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import apsw
import apsw.bestpractice

from rune.memory.types import Episode, Fact, SafetyRule  # re-export for compat
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

# Apply APSW best practices (WAL, foreign keys, etc.)
apsw.bestpractice.apply(apsw.bestpractice.recommended)

__all__ = ["Episode", "Fact", "SafetyRule", "MemoryStore", "get_memory_store"]  # Fact/SafetyRule re-exported from types.py for compat


# Schema DDL

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    task_summary TEXT,
    intent TEXT,
    plan TEXT,
    result TEXT,
    lessons TEXT,
    embedding BLOB,
    conversation_id TEXT,
    importance REAL DEFAULT 0.5
);

-- Phase 1: Episode Memory extensions (added columns are nullable for migration)
CREATE TABLE IF NOT EXISTS episode_commitments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    commitment_text TEXT NOT NULL,
    deadline TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_episode_commitments_status
ON episode_commitments(status, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_episode_commitments_episode
ON episode_commitments(episode_id);

CREATE TABLE IF NOT EXISTS command_history (
    id TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    success INTEGER,
    task_id TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    status TEXT DEFAULT 'active',
    execution_context TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    channel TEXT,
    timestamp TEXT NOT NULL,
    created_order INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS learned_patterns (
    id TEXT PRIMARY KEY,
    time_slot TEXT NOT NULL,
    day_type TEXT NOT NULL DEFAULT '',
    activity TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    avg_duration_minutes REAL DEFAULT 0,
    frequent_commands TEXT,
    frequent_file_patterns TEXT,
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS behavior_ngrams (
    id TEXT PRIMARY KEY,
    ngram TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS channel_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    preference TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(user_id, channel)
);

CREATE TABLE IF NOT EXISTS proactive_feedback (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    suggestion_type TEXT NOT NULL DEFAULT '',
    context_summary TEXT,
    response TEXT NOT NULL DEFAULT '',
    confidence REAL
);

CREATE TABLE IF NOT EXISTS sequence_patterns (
    id TEXT PRIMARY KEY,
    from_activity TEXT NOT NULL,
    to_activity TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    avg_transition_minutes REAL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE(from_activity, to_activity)
);

CREATE TABLE IF NOT EXISTS reflexion_strategy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    strategy TEXT NOT NULL,
    success_rate REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rejection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id TEXT NOT NULL,
    reason TEXT DEFAULT '',
    domain TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS engagement_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,
    value REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS proactive_suggestions_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    metadata TEXT DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS proactive_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    channel TEXT DEFAULT '',
    topic TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_engagement_metrics (
    user_id TEXT PRIMARY KEY,
    data TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_channel_preferences (
    user_id TEXT PRIMARY KEY,
    data TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS proactive_conversation_records (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    topic TEXT DEFAULT '',
    message TEXT DEFAULT '',
    channel TEXT DEFAULT '',
    response TEXT DEFAULT '',
    response_time_ms REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,
    command TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    run_count INTEGER NOT NULL DEFAULT 0,
    max_runs INTEGER,
    last_run_at TEXT,
    next_run_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tool_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    params TEXT DEFAULT '{}',
    result_success INTEGER NOT NULL DEFAULT 1,
    duration_ms REAL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    goal TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    steps INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT DEFAULT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_conversation ON episodes(conversation_id);
CREATE INDEX IF NOT EXISTS idx_command_history_timestamp ON command_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_conv ON conversation_turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_behavior_ngrams_count ON behavior_ngrams(count DESC);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_slot ON learned_patterns(time_slot, day_type);
CREATE INDEX IF NOT EXISTS idx_cron_jobs_enabled ON cron_jobs(enabled);
CREATE INDEX IF NOT EXISTS idx_proactive_conv_records_user ON proactive_conversation_records(user_id);
"""


# MemoryStore

_CURRENT_SCHEMA_VERSION = 5


class MemoryStore:
    """SQLite-backed persistent memory store using APSW with WAL mode."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = rune_data() / "memory.db"
        self._db_path = str(db_path)
        self._conn: apsw.Connection | None = None

    def _migrate(self) -> None:
        """Run schema migrations based on PRAGMA user_version."""
        conn = self._conn
        assert conn is not None
        (current_version,) = conn.execute("PRAGMA user_version").fetchone()

        if current_version < 1:
            # Initial schema (tables created by _SCHEMA_SQL via CREATE TABLE IF NOT EXISTS)
            conn.execute(_SCHEMA_SQL)

        if current_version < 2:
            # Add columns that may be missing in older DBs
            # Use try/except since column may already exist
            for stmt in [
                "ALTER TABLE episodes ADD COLUMN importance REAL DEFAULT 0.5",
                "ALTER TABLE learned_patterns ADD COLUMN day_type TEXT NOT NULL DEFAULT ''",
            ]:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # Column already exists

        if current_version < 3:
            # Add cron_jobs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    command TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    max_runs INTEGER,
                    last_run_at TEXT,
                    next_run_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cron_jobs_enabled ON cron_jobs(enabled)"
            )

        if current_version < 4:
            # Rebuild learned_patterns and sequence_patterns with correct schema.
            # The old tables had incompatible columns, so drop and recreate.
            conn.execute("DROP TABLE IF EXISTS learned_patterns")
            conn.execute("DROP TABLE IF EXISTS sequence_patterns")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id TEXT PRIMARY KEY,
                    time_slot TEXT NOT NULL,
                    day_type TEXT NOT NULL DEFAULT '',
                    activity TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    avg_duration_minutes REAL DEFAULT 0,
                    frequent_commands TEXT,
                    frequent_file_patterns TEXT,
                    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sequence_patterns (
                    id TEXT PRIMARY KEY,
                    from_activity TEXT NOT NULL,
                    to_activity TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    avg_transition_minutes REAL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(from_activity, to_activity)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learned_patterns_slot "
                "ON learned_patterns(time_slot, day_type)"
            )

        if current_version < 5:
            # Phase 1: Episode Memory - add extended fields + commitments table
            for stmt in [
                "ALTER TABLE episodes ADD COLUMN entities TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN files_touched TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN commitments TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN duration_ms REAL DEFAULT 0.0",
            ]:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # Column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS episode_commitments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    commitment_text TEXT NOT NULL,
                    deadline TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
                    resolved_at TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_commitments_status "
                "ON episode_commitments(status, detected_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_commitments_episode "
                "ON episode_commitments(episode_id)"
            )

        conn.execute(f"PRAGMA user_version = {_CURRENT_SCHEMA_VERSION}")
        log.info(
            "schema_migrated",
            from_version=current_version,
            to_version=_CURRENT_SCHEMA_VERSION,
        )

    def _get_conn(self) -> apsw.Connection:
        if self._conn is None:
            self._conn = apsw.Connection(self._db_path)
            self._conn.pragma("journal_mode", "wal")
            self._conn.pragma("foreign_keys", True)
            self._conn.pragma("busy_timeout", 5000)
            # Run migrations first (may drop/recreate tables)
            self._migrate()
            # Then ensure all tables/indexes exist
            self._conn.execute(_SCHEMA_SQL)
            log.info("memory_store_opened", path=self._db_path)
        return self._conn

    @property
    def conn(self) -> apsw.Connection:
        return self._get_conn()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----- Episodes ----------------------------------------------------------

    def save_episode(self, episode: Episode) -> None:
        now = datetime.now(UTC).isoformat()
        if not episode.timestamp:
            episode.timestamp = now

        self.conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, timestamp, task_summary, intent, plan, result, lessons,
                embedding, conversation_id, importance,
                entities, files_touched, commitments, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.id, episode.timestamp, episode.task_summary,
                episode.intent, episode.plan, episode.result,
                episode.lessons, episode.embedding,
                episode.conversation_id, episode.importance,
                episode.entities, episode.files_touched,
                episode.commitments, episode.duration_ms,
            ),
        )

    def _row_to_episode(self, r: tuple) -> Episode:
        """Convert a DB row tuple to an Episode, handling both old and new schema."""
        ep = Episode(
            id=r[0], timestamp=r[1], task_summary=r[2], intent=r[3],
            plan=r[4], result=r[5], lessons=r[6], embedding=r[7],
            conversation_id=r[8], importance=r[9],
        )
        # Phase 1 extended fields (may not exist in old rows)
        if len(r) > 10:
            ep.entities = r[10] or ""
        if len(r) > 11:
            ep.files_touched = r[11] or ""
        if len(r) > 12:
            ep.commitments = r[12] or ""
        if len(r) > 13:
            ep.duration_ms = r[13] or 0.0
        return ep

    def get_recent_episodes(self, limit: int = 10) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [self._row_to_episode(r) for r in rows]

    def get_ranked_episodes(
        self, query: str = "", limit: int = 20,
    ) -> list[Episode]:
        """Return episodes ranked by a weighted scoring formula.

        score = relevance * 0.5 + importance * 0.3 + recency * 0.2

        Where:
        - relevance: keyword overlap ratio between query and episode summary (0-1)
        - importance: episode.importance normalized to 0-1
        - recency: exponential decay exp(-age_hours / 168)  (168h ~ 1 week)
        """
        # Fetch a broad set of candidates to allow re-ranking
        candidate_limit = max(limit * 5, 100)
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
            (candidate_limit,),
        )
        episodes = [self._row_to_episode(r) for r in rows]

        if not episodes:
            return []

        now = datetime.now(UTC)
        query_terms = set(query.lower().split()) if query.strip() else set()

        scored: list[tuple[float, Episode]] = []
        for ep in episodes:
            # Relevance: keyword overlap ratio
            if query_terms:
                summary_terms = set((ep.task_summary or "").lower().split())
                overlap = len(query_terms & summary_terms)
                relevance = overlap / len(query_terms)
            else:
                relevance = 0.0

            # Importance: already 0-1 by convention, clamp just in case
            importance = max(0.0, min(1.0, ep.importance))

            # Recency: exponential decay with 1-week half-life (168 hours)
            try:
                ts = datetime.fromisoformat(ep.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                age_hours = (now - ts).total_seconds() / 3600.0
            except (ValueError, TypeError):
                age_hours = 168.0 * 4  # default to ~4 weeks old if unparseable
            recency = math.exp(-age_hours / 168.0)

            score = relevance * 0.5 + importance * 0.3 + recency * 0.2
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    # ----- Commitments (Phase 1: Episode Memory) ----------------------------

    def save_commitment(
        self, episode_id: str, text: str, deadline: str | None = None,
    ) -> None:
        """Save a detected commitment linked to an episode."""
        self.conn.execute(
            """INSERT INTO episode_commitments
               (episode_id, commitment_text, deadline)
               VALUES (?, ?, ?)""",
            (episode_id, text, deadline),
        )

    def get_open_commitments(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return open (unfulfilled) commitments, most recent first."""
        rows = self.conn.execute(
            """SELECT ec.id, ec.episode_id, ec.commitment_text, ec.deadline,
                      ec.detected_at, e.task_summary
               FROM episode_commitments ec
               JOIN episodes e ON ec.episode_id = e.id
               WHERE ec.status = 'open'
               ORDER BY ec.detected_at DESC
               LIMIT ?""",
            (limit,),
        )
        return [
            {
                "id": r[0], "episode_id": r[1], "text": r[2],
                "deadline": r[3], "detected_at": r[4],
                "task_summary": r[5],
            }
            for r in rows
        ]

    def resolve_commitment(self, commitment_id: int) -> None:
        """Mark a commitment as resolved."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            "UPDATE episode_commitments SET status = 'resolved', resolved_at = ? WHERE id = ?",
            (now, commitment_id),
        )

    def resolve_all_commitments(self) -> int:
        """Resolve ALL open commitments. Returns count resolved."""
        now = datetime.now(UTC).isoformat()
        cursor = self.conn.execute(
            "UPDATE episode_commitments SET status = 'resolved', resolved_at = ? WHERE status = 'open'",
            (now,),
        )
        return cursor.rowcount if hasattr(cursor, 'rowcount') else 0

    def get_episodes_by_file(self, file_path: str, limit: int = 10) -> list[Episode]:
        """Find episodes that touched a specific file."""
        rows = self.conn.execute(
            """SELECT * FROM episodes
               WHERE files_touched LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (f"%{file_path}%", limit),
        )
        return [self._row_to_episode(r) for r in rows]

    def get_episodes_by_entity(self, entity: str, limit: int = 10) -> list[Episode]:
        """Find episodes mentioning a specific entity (person, project, etc)."""
        rows = self.conn.execute(
            """SELECT * FROM episodes
               WHERE entities LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (f"%{entity}%", limit),
        )
        return [self._row_to_episode(r) for r in rows]

    def get_episodes_by_timerange(
        self, start: str, end: str, limit: int = 50,
    ) -> list[Episode]:
        """Find episodes within a time range (ISO format strings)."""
        rows = self.conn.execute(
            """SELECT * FROM episodes
               WHERE timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp DESC LIMIT ?""",
            (start, end, limit),
        )
        return [self._row_to_episode(r) for r in rows]

    # ----- Command History ---------------------------------------------------

    def log_command(self, command: str, success: bool, task_id: str = "") -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO command_history (id, command, timestamp, success, task_id)
               VALUES (?, ?, ?, ?, ?)""",
            (uuid4().hex[:16], command, now, int(success), task_id),
        )

    def get_recent_commands(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM command_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r[0], "command": r[1], "timestamp": r[2],
             "success": bool(r[3]), "task_id": r[4]}
            for r in rows
        ]

    # ----- Conversations -----------------------------------------------------

    def create_conversation(self, user_id: str = "", title: str = "") -> str:
        conv_id = uuid4().hex[:16]
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO conversations (id, user_id, title, status, created_at, updated_at)
               VALUES (?, ?, ?, 'active', ?, ?)""",
            (conv_id, user_id, title, now, now),
        )
        return conv_id

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get a conversation by ID."""
        row = self.conn.execute(
            "SELECT id, user_id, title, status, execution_context, created_at, updated_at "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "user_id": row[1], "title": row[2],
            "status": row[3], "execution_context": row[4],
            "created_at": row[5], "updated_at": row[6],
        }

    def list_conversations(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List conversations with optional user/status filter and pagination."""
        clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status:
            clauses.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])

        rows = self.conn.execute(
            f"SELECT id, user_id, title, status, execution_context, created_at, updated_at "
            f"FROM conversations {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            tuple(params),
        ).fetchall()

        return [
            {
                "id": r[0], "user_id": r[1], "title": r[2],
                "status": r[3], "execution_context": r[4],
                "created_at": r[5], "updated_at": r[6],
            }
            for r in rows
        ]

    def count_conversations(
        self,
        user_id: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count conversations matching the given filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status:
            clauses.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        (cnt,) = self.conn.execute(
            f"SELECT COUNT(*) FROM conversations {where}",
            tuple(params),
        ).fetchone()
        return cnt

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its turns. Returns True if it existed."""
        existing = self.get_conversation(conversation_id) is not None
        self.conn.execute(
            "DELETE FROM conversation_turns WHERE conversation_id = ?",
            (conversation_id,),
        )
        self.conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return existing

    def get_conversation_turns(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get turns for a conversation, ordered chronologically."""
        rows = self.conn.execute(
            "SELECT id, conversation_id, role, content, channel, timestamp "
            "FROM conversation_turns WHERE conversation_id = ? ORDER BY created_order LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "conversation_id": r[1], "role": r[2],
                "content": r[3], "channel": r[4], "timestamp": r[5],
            }
            for r in rows
        ]

    def count_conversation_turns(self, conversation_id: str) -> int:
        """Count turns in a conversation."""
        (cnt,) = self.conn.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return cnt

    def add_turn(
        self, conversation_id: str, role: str, content: str, channel: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        # Determine next created_order for this conversation
        (max_order,) = self.conn.execute(
            "SELECT COALESCE(MAX(created_order), -1) FROM conversation_turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        self.conn.execute(
            """INSERT INTO conversation_turns
               (id, conversation_id, role, content, channel, timestamp, created_order)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uuid4().hex[:16], conversation_id, role, content, channel, now, max_order + 1),
        )

    # ----- Learned Patterns --------------------------------------------------

    def save_learned_pattern(
        self,
        time_slot: str,
        activity: str,
        *,
        pattern_id: str = "",
        count: int = 1,
        day_type: str = "",
        avg_duration_minutes: float = 0.0,
        frequent_commands: list[str] | None = None,
        frequent_file_patterns: list[str] | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        pid = pattern_id or uuid4().hex[:16]

        self.conn.execute(
            """INSERT OR REPLACE INTO learned_patterns
               (id, time_slot, day_type, activity, count,
                avg_duration_minutes, frequent_commands, frequent_file_patterns,
                last_seen, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid,
                time_slot,
                day_type,
                activity,
                count,
                avg_duration_minutes,
                json_encode(frequent_commands) if frequent_commands else None,
                json_encode(frequent_file_patterns) if frequent_file_patterns else None,
                now,
                now,
                now,
            ),
        )

    def get_learned_patterns(
        self,
        time_slot: str | None = None,
        day_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if time_slot and day_type:
            rows = self.conn.execute(
                """SELECT * FROM learned_patterns
                   WHERE time_slot = ? AND day_type = ?
                   ORDER BY count DESC LIMIT ?""",
                (time_slot, day_type, limit),
            )
        elif time_slot:
            rows = self.conn.execute(
                """SELECT * FROM learned_patterns
                   WHERE time_slot = ? ORDER BY count DESC LIMIT ?""",
                (time_slot, limit),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM learned_patterns ORDER BY count DESC LIMIT ?",
                (limit,),
            )

        def _parse_json_list(val: str | None) -> list[str]:
            if not val:
                return []
            try:
                return json_decode(val)
            except (ValueError, TypeError):
                return []

        return [
            {
                "id": r[0],
                "time_slot": r[1],
                "day_type": r[2],
                "activity": r[3],
                "count": r[4],
                "avg_duration_minutes": r[5] or 0.0,
                "frequent_commands": _parse_json_list(r[6]),
                "frequent_file_patterns": _parse_json_list(r[7]),
                "last_seen": r[8],
                "created_at": r[9],
                "updated_at": r[10],
            }
            for r in rows
        ]

    # ----- Behavior N-grams --------------------------------------------------

    def store_ngram(
        self,
        ngram: str,
        *,
        ngram_id: str = "",
        count: int = 1,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        nid = ngram_id or uuid4().hex[:16]
        self.conn.execute(
            """INSERT OR REPLACE INTO behavior_ngrams
               (id, ngram, count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (nid, ngram, count, now),
        )

    def get_ngrams_by_key(
        self, ngram: str,
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM behavior_ngrams WHERE ngram = ?", (ngram,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "ngram": row[1], "count": row[2], "updated_at": row[3],
        }

    def get_ngrams(self, min_count: int = 1, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM behavior_ngrams
               WHERE count >= ? ORDER BY count DESC LIMIT ?""",
            (min_count, limit),
        )
        return [
            {"id": r[0], "ngram": r[1], "count": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def delete_stale_ngrams(self, max_age_days: int, min_count: int = 2) -> int:
        cutoff = datetime.fromtimestamp(
            time.time() - max_age_days * 86400, tz=UTC,
        ).isoformat()

        # Count items to delete
        (cnt,) = self.conn.execute(
            """SELECT COUNT(*) FROM behavior_ngrams
               WHERE updated_at < ? AND count < ?""",
            (cutoff, min_count),
        ).fetchone()

        if cnt > 0:
            self.conn.execute(
                "DELETE FROM behavior_ngrams WHERE updated_at < ? AND count < ?",
                (cutoff, min_count),
            )

        # Halve counts on old but frequent entries
        self.conn.execute(
            """UPDATE behavior_ngrams
               SET count = MAX(1, count / 2)
               WHERE updated_at < ? AND count >= ?""",
            (cutoff, min_count),
        )

        return cnt

    # ----- Channel Preferences -----------------------------------------------

    def save_channel_preference(
        self,
        user_id: str,
        channel: str,
        preference: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        pref_id = uuid4().hex[:16]
        self.conn.execute(
            """INSERT INTO channel_preferences
               (id, user_id, channel, preference, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, channel) DO UPDATE SET
                 preference = excluded.preference,
                 updated_at = excluded.updated_at""",
            (pref_id, user_id, channel, preference, now),
        )

    def get_channel_preference(
        self, user_id: str, channel: str | None = None,
    ) -> list[dict[str, Any]]:
        if channel:
            rows = self.conn.execute(
                """SELECT * FROM channel_preferences
                   WHERE user_id = ? AND channel = ?""",
                (user_id, channel),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM channel_preferences WHERE user_id = ?",
                (user_id,),
            )
        return [
            {
                "id": r[0], "user_id": r[1], "channel": r[2],
                "preference": r[3], "updated_at": r[4],
            }
            for r in rows
        ]

    # ----- Reflexion Strategy ------------------------------------------------

    def save_reflexion_strategy(
        self,
        domain: str,
        strategy: str,
        success_rate: float = 0.0,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO reflexion_strategy
               (domain, strategy, success_rate, updated_at)
               VALUES (?, ?, ?, ?)""",
            (domain, strategy, success_rate, now),
        )

    def get_reflexion_strategies(
        self, domain: str | None = None,
    ) -> list[dict[str, Any]]:
        if domain:
            rows = self.conn.execute(
                """SELECT * FROM reflexion_strategy
                   WHERE domain = ? ORDER BY success_rate DESC""",
                (domain,),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM reflexion_strategy ORDER BY success_rate DESC",
            )
        return [
            {
                "id": r[0], "domain": r[1], "strategy": r[2],
                "success_rate": r[3], "updated_at": r[4],
            }
            for r in rows
        ]

    # ----- Rejection History -------------------------------------------------

    def save_rejection(
        self,
        suggestion_id: str,
        reason: str = "",
        domain: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO rejection_history
               (suggestion_id, reason, domain, created_at)
               VALUES (?, ?, ?, ?)""",
            (suggestion_id, reason, domain, now),
        )

    def get_rejections(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM rejection_history
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "id": r[0], "suggestion_id": r[1], "reason": r[2],
                "domain": r[3], "created_at": r[4],
            }
            for r in rows
        ]

    def prune_rejections(self, max_age_days: int = 30) -> int:
        cutoff = datetime.fromtimestamp(
            time.time() - max_age_days * 86400, tz=UTC,
        ).isoformat()

        (cnt,) = self.conn.execute(
            "SELECT COUNT(*) FROM rejection_history WHERE created_at < ?",
            (cutoff,),
        ).fetchone()

        if cnt > 0:
            self.conn.execute(
                "DELETE FROM rejection_history WHERE created_at < ?",
                (cutoff,),
            )
        return cnt

    # ----- Engagement Metrics ------------------------------------------------

    def save_engagement_metric(
        self,
        metric_type: str,
        value: float,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO engagement_metrics
               (metric_type, value, recorded_at)
               VALUES (?, ?, ?)""",
            (metric_type, value, now),
        )

    def get_engagement_metrics(
        self,
        metric_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if metric_type:
            rows = self.conn.execute(
                """SELECT * FROM engagement_metrics
                   WHERE metric_type = ? ORDER BY recorded_at DESC LIMIT ?""",
                (metric_type, limit),
            )
        else:
            rows = self.conn.execute(
                """SELECT * FROM engagement_metrics
                   ORDER BY recorded_at DESC LIMIT ?""",
                (limit,),
            )
        return [
            {
                "id": r[0], "metric_type": r[1], "value": r[2],
                "recorded_at": r[3],
            }
            for r in rows
        ]

    # ----- Proactive Suggestions State ---------------------------------------

    def save_suggestion_state(
        self,
        suggestion_type: str,
        state: str = "pending",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        meta_json = json_encode(metadata or {})
        self.conn.execute(
            """INSERT INTO proactive_suggestions_state
               (suggestion_type, state, metadata, updated_at)
               VALUES (?, ?, ?, ?)""",
            (suggestion_type, state, meta_json, now),
        )
        (row_id,) = self.conn.execute("SELECT last_insert_rowid()").fetchone()
        return row_id

    def get_suggestion_state(
        self, suggestion_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if suggestion_type:
            rows = self.conn.execute(
                """SELECT * FROM proactive_suggestions_state
                   WHERE suggestion_type = ? ORDER BY updated_at DESC""",
                (suggestion_type,),
            )
        else:
            rows = self.conn.execute(
                """SELECT * FROM proactive_suggestions_state
                   ORDER BY updated_at DESC""",
            )
        return [
            {
                "id": r[0], "suggestion_type": r[1], "state": r[2],
                "metadata": json_decode(r[3] or "{}"), "updated_at": r[4],
            }
            for r in rows
        ]

    def delete_suggestion_state(self, suggestion_id: int) -> None:
        self.conn.execute(
            "DELETE FROM proactive_suggestions_state WHERE id = ?",
            (suggestion_id,),
        )

    def prune_expired_suggestions(self, max_age_days: int = 7) -> int:
        cutoff = datetime.fromtimestamp(
            time.time() - max_age_days * 86400, tz=UTC,
        ).isoformat()

        (cnt,) = self.conn.execute(
            """SELECT COUNT(*) FROM proactive_suggestions_state
               WHERE updated_at < ?""",
            (cutoff,),
        ).fetchone()

        if cnt > 0:
            self.conn.execute(
                "DELETE FROM proactive_suggestions_state WHERE updated_at < ?",
                (cutoff,),
            )
        return cnt

    # ----- Proactive Conversations -------------------------------------------

    def save_proactive_conversation(
        self,
        conversation_id: str,
        channel: str = "",
        topic: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO proactive_conversations
               (conversation_id, channel, topic, created_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, channel, topic, now),
        )

    def get_recent_proactive_conversations(
        self, limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM proactive_conversations
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "id": r[0], "conversation_id": r[1], "channel": r[2],
                "topic": r[3], "created_at": r[4],
            }
            for r in rows
        ]

    # ----- Per-User Engagement Store (TS EngagementStore adapter) ------------

    def get_user_engagement_metrics(
        self, user_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve per-user engagement metrics as a JSON dict (TS getEngagementMetrics)."""
        row = self.conn.execute(
            "SELECT data FROM user_engagement_metrics WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        try:
            return json_decode(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def store_user_engagement_metrics(
        self, user_id: str, data: dict[str, Any],
    ) -> None:
        """Upsert per-user engagement metrics as a JSON blob (TS storeEngagementMetrics)."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO user_engagement_metrics (user_id, data, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 data = excluded.data,
                 updated_at = excluded.updated_at""",
            (user_id, json.dumps(data, default=str), now),
        )

    def get_user_channel_preferences(
        self, user_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve per-user channel preferences as a JSON dict (TS getChannelPreferences)."""
        row = self.conn.execute(
            "SELECT data FROM user_channel_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        try:
            return json_decode(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def store_user_channel_preferences(
        self, user_id: str, data: dict[str, Any],
    ) -> None:
        """Upsert per-user channel preferences as a JSON blob (TS storeChannelPreferences)."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO user_channel_preferences (user_id, data, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 data = excluded.data,
                 updated_at = excluded.updated_at""",
            (user_id, json.dumps(data, default=str), now),
        )

    def store_proactive_conversation_record(
        self, record: dict[str, Any],
    ) -> None:
        """Store a proactive conversation record (TS storeProactiveConversation)."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO proactive_conversation_records
               (id, user_id, category, topic, message, channel, response, response_time_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("id", uuid4().hex[:16]),
                record.get("user_id", ""),
                record.get("category", ""),
                record.get("topic"),
                record.get("message"),
                record.get("channel"),
                record.get("response"),
                record.get("response_time_ms"),
                now,
            ),
        )

    # ----- Tool Call Log -----------------------------------------------------

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        *,
        params: dict[str, Any] | None = None,
        result_success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        params_json = json_encode(params or {})
        self.conn.execute(
            """INSERT INTO tool_call_log
               (session_id, tool_name, params, result_success, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, tool_name, params_json, int(result_success), duration_ms, now),
        )

    def get_recent_tool_calls(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM tool_call_log
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "id": r[0], "session_id": r[1], "tool_name": r[2],
                "params": json_decode(r[3] or "{}"),
                "result_success": bool(r[4]), "duration_ms": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    def prune_tool_calls(self, max_age_days: int = 30) -> int:
        cutoff = datetime.fromtimestamp(
            time.time() - max_age_days * 86400, tz=UTC,
        ).isoformat()

        (cnt,) = self.conn.execute(
            "SELECT COUNT(*) FROM tool_call_log WHERE created_at < ?",
            (cutoff,),
        ).fetchone()

        if cnt > 0:
            self.conn.execute(
                "DELETE FROM tool_call_log WHERE created_at < ?",
                (cutoff,),
            )
        return cnt

    # ----- Runs --------------------------------------------------------------

    _MAX_RESULT_ANSWER_CHARS = 10_000

    def upsert_run(
        self,
        run_id: str,
        goal: str = "",
        status: str = "running",
        steps: int = 0,
        tokens_used: int = 0,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO runs
               (run_id, goal, status, steps, tokens_used, started_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET
                 goal = excluded.goal,
                 status = excluded.status,
                 steps = excluded.steps,
                 tokens_used = excluded.tokens_used""",
            (run_id, goal, status, steps, tokens_used, now),
        )

    def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        steps: int | None = None,
        tokens_used: int | None = None,
        finished_at: str | None = None,
    ) -> None:
        sets = ["status = ?"]
        params: list[Any] = [status]

        if steps is not None:
            sets.append("steps = ?")
            params.append(steps)
        if tokens_used is not None:
            sets.append("tokens_used = ?")
            params.append(tokens_used)
        if finished_at is not None:
            sets.append("finished_at = ?")
            params.append(finished_at)
        elif status in ("completed", "failed", "cancelled"):
            sets.append("finished_at = ?")
            params.append(datetime.now(UTC).isoformat())

        params.append(run_id)
        self.conn.execute(
            f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?",
            tuple(params),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "run_id": row[1], "goal": row[2],
            "status": row[3], "steps": row[4], "tokens_used": row[5],
            "started_at": row[6], "finished_at": row[7],
        }

    def list_runs(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if status:
            rows = self.conn.execute(
                """SELECT * FROM runs
                   WHERE status = ? ORDER BY started_at DESC LIMIT ? OFFSET ?""",
                (status, limit, offset),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [
            {
                "id": r[0], "run_id": r[1], "goal": r[2],
                "status": r[3], "steps": r[4], "tokens_used": r[5],
                "started_at": r[6], "finished_at": r[7],
            }
            for r in rows
        ]

    # ----- Cron Jobs ---------------------------------------------------------

    def create_cron_job(
        self,
        *,
        job_id: str = "",
        name: str,
        schedule: str,
        command: str,
        enabled: bool = True,
        max_runs: int | None = None,
    ) -> str:
        """Create a new cron job and return its ID."""
        jid = job_id or uuid4().hex[:16]
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT INTO cron_jobs
               (id, name, schedule, command, enabled, run_count, max_runs,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)""",
            (jid, name, schedule, command, int(enabled), max_runs, now, now),
        )
        return jid

    def get_cron_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM cron_jobs WHERE id = ?", (job_id,),
        ).fetchone()
        if not row:
            return None
        return self._cron_row_to_dict(row)

    def list_cron_jobs(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        if enabled_only:
            rows = self.conn.execute(
                "SELECT * FROM cron_jobs WHERE enabled = 1 ORDER BY created_at DESC",
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM cron_jobs ORDER BY created_at DESC",
            ).fetchall()
        return [self._cron_row_to_dict(r) for r in rows]

    def update_cron_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        schedule: str | None = None,
        command: str | None = None,
        enabled: bool | None = None,
        max_runs: int | None = ...,  # type: ignore[assignment]
        run_count: int | None = None,
        last_run_at: str | None = None,
        next_run_at: str | None = ...,  # type: ignore[assignment]
    ) -> bool:
        """Update fields of a cron job. Returns True if the row existed."""
        sets: list[str] = []
        params: list[Any] = []

        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if schedule is not None:
            sets.append("schedule = ?")
            params.append(schedule)
        if command is not None:
            sets.append("command = ?")
            params.append(command)
        if enabled is not None:
            sets.append("enabled = ?")
            params.append(int(enabled))
        if max_runs is not ...:
            sets.append("max_runs = ?")
            params.append(max_runs)
        if run_count is not None:
            sets.append("run_count = ?")
            params.append(run_count)
        if last_run_at is not None:
            sets.append("last_run_at = ?")
            params.append(last_run_at)
        if next_run_at is not ...:
            sets.append("next_run_at = ?")
            params.append(next_run_at)

        if not sets:
            return self.get_cron_job(job_id) is not None

        sets.append("updated_at = ?")
        params.append(datetime.now(UTC).isoformat())
        params.append(job_id)

        self.conn.execute(
            f"UPDATE cron_jobs SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )
        # Check if row was actually found
        return self.get_cron_job(job_id) is not None

    def delete_cron_job(self, job_id: str) -> bool:
        """Delete a cron job. Returns True if it existed."""
        existing = self.get_cron_job(job_id) is not None
        self.conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        return existing

    def toggle_cron_job(self, job_id: str) -> bool | None:
        """Toggle enabled state. Returns new enabled state, or None if not found."""
        job = self.get_cron_job(job_id)
        if job is None:
            return None
        new_enabled = not job["enabled"]
        self.update_cron_job(job_id, enabled=new_enabled)
        return new_enabled

    def record_cron_run(self, job_id: str) -> None:
        """Increment run_count and set last_run_at."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """UPDATE cron_jobs
               SET run_count = run_count + 1,
                   last_run_at = ?,
                   updated_at = ?
               WHERE id = ?""",
            (now, now, job_id),
        )

    @staticmethod
    def _cron_row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "schedule": row[2],
            "command": row[3],
            "enabled": bool(row[4]),
            "run_count": row[5],
            "max_runs": row[6],
            "last_run_at": row[7],
            "next_run_at": row[8],
            "created_at": row[9],
            "updated_at": row[10],
        }

    # ----- Proactive Feedback ------------------------------------------------

    def save_proactive_feedback(
        self,
        *,
        feedback_id: str | None = None,
        suggestion_type: str = "",
        context_summary: str | None = None,
        response: str = "",
        confidence: float | None = None,
    ) -> str:
        """Store user feedback on a proactive suggestion.

        Matches TS schema: ``(id TEXT PK, timestamp, suggestion_type,
        context_summary, response, confidence)``.

        Returns the generated feedback id.
        """
        fb_id = feedback_id or f"fb_{uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO proactive_feedback
               (id, timestamp, suggestion_type, context_summary,
                response, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (fb_id, now, suggestion_type, context_summary,
             response, confidence),
        )
        return fb_id

    def get_proactive_feedback(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve proactive feedback records ordered by most recent first."""
        rows = self.conn.execute(
            """SELECT id, timestamp, suggestion_type, context_summary,
                      response, confidence
               FROM proactive_feedback
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "suggestion_type": r[2],
                "context_summary": r[3],
                "response": r[4],
                "confidence": r[5],
            }
            for r in rows
        ]

    # ----- Sequence Patterns -------------------------------------------------

    def save_sequence_pattern(
        self,
        from_activity: str,
        to_activity: str,
        *,
        count: int = 1,
        avg_transition_minutes: float = 0.0,
    ) -> None:
        """Store or upsert a sequence pattern (activity transition).

        Uses INSERT OR REPLACE keyed on the UNIQUE(from_activity, to_activity)
        constraint, matching the TS ``storeSequencePattern`` behaviour.
        """
        now = datetime.now(UTC).isoformat()

        # Check for existing row so we can merge counts
        existing = self.conn.execute(
            "SELECT id, count, avg_transition_minutes FROM sequence_patterns "
            "WHERE from_activity = ? AND to_activity = ?",
            (from_activity, to_activity),
        ).fetchone()

        if existing:
            new_count = existing[1] + count
            # Weighted average of transition minutes
            old_total = existing[2] * existing[1]
            new_total = avg_transition_minutes * count
            new_avg = (old_total + new_total) / new_count if new_count else 0.0
            self.conn.execute(
                """UPDATE sequence_patterns
                   SET count = ?,
                       avg_transition_minutes = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (new_count, new_avg, now, existing[0]),
            )
        else:
            self.conn.execute(
                """INSERT INTO sequence_patterns
                   (id, from_activity, to_activity, count, avg_transition_minutes, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid4()), from_activity, to_activity, count, avg_transition_minutes, now),
            )

    def get_sequence_patterns(
        self,
        min_count: int = 1,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve sequence patterns ordered by count descending."""
        rows = self.conn.execute(
            """SELECT id, from_activity, to_activity, count,
                      avg_transition_minutes, updated_at
               FROM sequence_patterns
               WHERE count >= ? ORDER BY count DESC LIMIT ?""",
            (min_count, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "from_activity": r[1],
                "to_activity": r[2],
                "count": r[3],
                "avg_transition_minutes": r[4],
                "updated_at": r[5],
            }
            for r in rows
        ]

    # ----- Utility methods ---------------------------------------------------

    def get_fact_by_key(self, key: str) -> Fact | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE key = ?", (key,),
        ).fetchone()
        if not row:
            return None
        return Fact(
            id=row[0], category=row[1], key=row[2], value=row[3],
            source=row[4], confidence=row[5], last_verified=row[6],
            created_at=row[7], updated_at=row[8],
        )

    def archive_stale_conversations(self, idle_minutes: int = 30) -> int:
        cutoff = datetime.fromtimestamp(
            time.time() - idle_minutes * 60, tz=UTC,
        ).isoformat()

        (cnt,) = self.conn.execute(
            """SELECT COUNT(*) FROM conversations
               WHERE status = 'active' AND updated_at < ?""",
            (cutoff,),
        ).fetchone()

        if cnt > 0:
            self.conn.execute(
                """UPDATE conversations SET status = 'archived'
                   WHERE status = 'active' AND updated_at < ?""",
                (cutoff,),
            )
        return cnt

    def get_stats(self) -> dict[str, Any]:
        tables = [
            "episodes", "facts", "safety_rules", "command_history",
            "conversations", "conversation_turns", "learned_patterns",
            "behavior_ngrams", "channel_preferences", "reflexion_strategy",
            "rejection_history", "engagement_metrics",
            "proactive_suggestions_state", "proactive_conversations",
            "user_engagement_metrics", "user_channel_preferences",
            "proactive_conversation_records",
            "tool_call_log", "runs", "cron_jobs", "proactive_feedback", "sequence_patterns",
        ]
        counts: dict[str, int] = {}
        for table in tables:
            (cnt,) = self.conn.execute(
                f"SELECT COUNT(*) FROM {table}",  # noqa: S608
            ).fetchone()
            counts[table] = cnt

        page_count = self.conn.pragma("page_count") or 0
        page_size = self.conn.pragma("page_size") or 0
        counts["database_size"] = page_count * page_size
        return counts


# Module singleton

_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
