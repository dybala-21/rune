"""Durable web execution records in the conversation database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from filelock import FileLock


class RunStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._db: sqlite3.Connection | None = None
        self._lock: FileLock | None = None

    def open(self) -> bool:
        if self._db is not None:
            return False
        from rune.utils.paths import conversations_db_path

        path = (self._path or conversations_db_path()).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only the API owner may declare old executions interrupted. A second
        # server sharing this home must not invalidate the first one's work.
        lock = FileLock(str(path) + ".runs.lock", timeout=0, mode=0o600)
        lock.acquire()
        db = None
        try:
            db = sqlite3.connect(path, timeout=5)
            path.chmod(0o600)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript("""
                BEGIN;
                CREATE TABLE IF NOT EXISTS web_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    snapshot TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_web_runs_session ON web_runs(session_id);
                CREATE TABLE IF NOT EXISTS web_run_events (
                    run_id TEXT NOT NULL REFERENCES web_runs(run_id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );
                CREATE TABLE IF NOT EXISTS web_run_interactions (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES web_runs(run_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_id TEXT,
                    response TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_web_interactions_run ON web_run_interactions(run_id);
                CREATE TABLE IF NOT EXISTS web_tool_attempts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES web_runs(run_id) ON DELETE CASCADE,
                    record TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_web_attempts_run ON web_tool_attempts(run_id);
                CREATE TABLE IF NOT EXISTS web_run_resumptions (
                    parent_id TEXT PRIMARY KEY REFERENCES web_runs(run_id) ON DELETE CASCADE,
                    child_id TEXT NOT NULL REFERENCES web_runs(run_id) ON DELETE CASCADE
                );
                COMMIT;
            """)
        except BaseException:
            if db is not None:
                db.close()
            lock.release()
            raise
        self._db, self._lock = db, lock
        return True

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("Run store is not open")
        return self._db

    def create(self, snapshot: dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO web_runs VALUES (?, ?, ?, ?, ?)",
                (snapshot["runId"], snapshot["sessionId"], snapshot["status"],
                 snapshot["seq"], json.dumps(snapshot, ensure_ascii=False)),
            )

    def append(
        self, run: dict[str, Any], event: str, data: dict[str, Any], timestamp: float,
        *, checkpoint: dict[str, Any] | None = None,
        status: str,
        response: tuple[str, str, dict[str, Any]] | None = None,
    ) -> None:
        session_id = data.get("sessionId", run["sessionId"]) or run["sessionId"]
        with self.db:
            updated = self.db.execute(
                "UPDATE web_runs SET seq = ?, status = ?, session_id = ? WHERE run_id = ? AND seq = ?",
                (run["seq"] + 1, status, session_id, run["runId"], run["seq"]),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Execution record changed or was deleted")
            if checkpoint is not None:
                self.db.execute("UPDATE web_runs SET snapshot = ? WHERE run_id = ?",
                                (json.dumps(checkpoint, ensure_ascii=False), run["runId"]))
                self.db.execute("DELETE FROM web_run_events WHERE run_id = ? AND seq <= ?",
                                (run["runId"], run["seq"]))
            self.db.execute("INSERT INTO web_run_events VALUES (?, ?, ?, ?, ?)",
                            (run["runId"], run["seq"] + 1, event,
                             json.dumps(data, ensure_ascii=False), timestamp))
            if event in {"question", "approval_request"}:
                self.db.execute("INSERT INTO web_run_interactions VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                                (data["id"], run["runId"], "question" if event == "question" else "approval",
                                 json.dumps(data, ensure_ascii=False),
                                 "autonomous" if data.get("autonomous") or data.get("autoApproved") else "pending"))
            if response is not None:
                interaction_id, response_id, payload = response
                updated = self.db.execute(
                    "UPDATE web_run_interactions SET status = 'answered', response_id = ?, response = ? "
                    "WHERE id = ? AND run_id = ? AND status = 'pending'",
                    (response_id, json.dumps(payload, ensure_ascii=False), interaction_id, run["runId"]),
                )
                if updated.rowcount != 1:
                    raise ValueError("Interaction is no longer pending")
            elif event in {"question_closed", "approval_closed"}:
                self.db.execute("UPDATE web_run_interactions SET status = 'closed' WHERE id = ? AND status = 'pending'",
                                (data["id"],))
            if status in {"completed", "failed", "cancelled", "interrupted"}:
                self.db.execute("UPDATE web_run_interactions SET status = ? WHERE run_id = ? AND status = 'pending'",
                                (status, run["runId"]))

    def load(self, run_id: str) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], float]]] | None:
        row = self.db.execute("SELECT snapshot FROM web_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        snapshot = json.loads(row[0])
        events = self.db.execute(
            "SELECT event, data, timestamp FROM web_run_events WHERE run_id = ? AND seq > ? ORDER BY seq",
            (run_id, snapshot["seq"]),
        ).fetchall()
        return snapshot, [(event, json.loads(data), timestamp) for event, data, timestamp in events]

    def latest_id(self, session_id: str) -> str | None:
        row = self.db.execute("SELECT run_id FROM web_runs WHERE session_id = ? ORDER BY rowid DESC LIMIT 1",
                              (session_id,)).fetchone()
        return row[0] if row else None

    def active_ids(self) -> list[str]:
        return [row[0] for row in self.db.execute(
            "SELECT run_id FROM web_runs WHERE status NOT IN ('completed', 'failed', 'cancelled', 'interrupted')"
        )]

    def attempts(self, run_id: str) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT record FROM web_tool_attempts WHERE run_id = ? ORDER BY rowid", (run_id,),
        )]

    def save_attempt(self, record: dict[str, Any]) -> None:
        with self.db:
            row = self.db.execute("SELECT status FROM web_runs WHERE run_id = ?", (record["run_id"],)).fetchone()
            if row is None or row[0] in {"completed", "failed", "cancelled", "interrupted"}:
                raise RuntimeError("Execution is no longer active")
            self.db.execute(
                "INSERT INTO web_tool_attempts VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET record = excluded.record",
                (record["id"], record["run_id"], json.dumps(record, ensure_ascii=False)),
            )

    def resumed_child(self, parent_id: str) -> str | None:
        row = self.db.execute("SELECT child_id FROM web_run_resumptions WHERE parent_id = ?", (parent_id,)).fetchone()
        return row[0] if row else None

    def create_resumption(self, parent_id: str, snapshot: dict[str, Any]) -> None:
        with self.db:
            row = self.db.execute("SELECT status FROM web_runs WHERE run_id = ?", (parent_id,)).fetchone()
            if row is None or row[0] != "interrupted":
                raise ValueError("Only interrupted executions can be resumed")
            self.db.execute("INSERT INTO web_runs VALUES (?, ?, ?, ?, ?)",
                            (snapshot["runId"], snapshot["sessionId"], snapshot["status"],
                             snapshot["seq"], json.dumps(snapshot, ensure_ascii=False)))
            self.db.execute("INSERT INTO web_run_resumptions VALUES (?, ?)", (parent_id, snapshot["runId"]))

    def replay(self, interaction_id: str, response_id: str, payload: dict[str, Any]) -> bool:
        if not response_id:
            return False
        row = self.db.execute("SELECT response_id, response FROM web_run_interactions WHERE id = ? AND status = 'answered'",
                              (interaction_id,)).fetchone()
        if row is None:
            return False
        if row[0] != response_id or json.loads(row[1]) != payload:
            raise ValueError("This interaction already has a different response")
        return True

    def interactions(self, run_id: str) -> list[dict[str, Any]]:
        return [{"id": row[0], "kind": row[1], "request": json.loads(row[2]),
                 "status": row[3], "response": json.loads(row[4]) if row[4] else None}
                for row in self.db.execute(
                    "SELECT id, kind, request, status, response FROM web_run_interactions WHERE run_id = ? ORDER BY rowid",
                    (run_id,))]

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None
