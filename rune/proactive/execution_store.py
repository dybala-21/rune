"""Persist execution claims; unfinished claims remain unknown until their effects are checked."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


class ExecutionStore:
    def __init__(self, path: Path | None = None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path else ":memory:", isolation_level=None)
        self._db.execute("PRAGMA busy_timeout=1000")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                started_at REAL NOT NULL, result TEXT
            )
        """)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(operations)")}
        if "owner_pid" not in columns:
            self._db.execute("ALTER TABLE operations ADD COLUMN owner_pid INTEGER")
        self._db.execute("""CREATE TABLE IF NOT EXISTS resources (
            resource TEXT PRIMARY KEY, operation_id TEXT NOT NULL
        )""")

    def get(self, operation_id: str) -> tuple[str, dict[str, Any] | None] | None:
        row = self._db.execute(
            "SELECT fingerprint, result FROM operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return (row[0], json.loads(row[1]) if row[1] else None) if row else None

    def claim(self, operation_id: str, fingerprint: str, limit: int, *, resource: str = "") -> str:
        """Reserve the operation ID and hourly budget in one transaction."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            if self.get(operation_id) is not None:
                decision = "exists"
            elif resource and self._db.execute(
                "SELECT 1 FROM resources WHERE resource = ?", (resource,),
            ).fetchone():
                decision = "busy"
            elif self.started_since(time.time() - 3600) >= limit:
                decision = "rate_limited"
            else:
                self._db.execute(
                    "INSERT INTO operations (id, fingerprint, started_at, owner_pid) VALUES (?, ?, ?, ?)",
                    (operation_id, fingerprint, time.time(), os.getpid()),
                )
                if resource:
                    self._db.execute("INSERT INTO resources VALUES (?, ?)", (resource, operation_id))
                decision = "claimed"
            self._db.execute("COMMIT")
            return decision
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def finish(self, operation_id: str, result: dict[str, Any]) -> None:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            changed = self._db.execute(
                "UPDATE operations SET result = ? WHERE id = ? AND result IS NULL",
                (json.dumps(result, ensure_ascii=False), operation_id),
            ).rowcount
            # An interrupted action keeps its lease until its effects are inspected.
            if changed and result.get("status") != "interrupted" and not result.get("execution_unknown"):
                self._db.execute("DELETE FROM resources WHERE operation_id = ?", (operation_id,))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def recent(self, prefix: str, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(id=row[0], started_at=row[1], result=json.loads(row[2]) if row[2] else None,
                     blocked=bool(row[3])) for row in self._db.execute(
            """SELECT o.id, o.started_at, o.result, r.resource FROM operations o
               LEFT JOIN resources r ON r.operation_id = o.id
               WHERE substr(o.id, 1, ?) = ? ORDER BY o.started_at DESC LIMIT ?""",
            (len(prefix), prefix, min(max(limit, 1), 100)),
        )]

    def reconcile(self, resource: str, operation_id: str, note: str) -> bool:
        """Release a stopped run after the user has inspected its external effects."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute(
                """SELECT o.result, o.owner_pid FROM operations o JOIN resources r ON o.id = r.operation_id
                   WHERE r.resource = ? AND o.id = ?""", (resource, operation_id),
            ).fetchone()
            if row is None:
                self._db.execute("COMMIT")
                return False
            if row[0] is None:
                # Missing legacy ownership or a live PID cannot prove execution stopped.
                if row[1] is None:
                    raise ValueError("Execution ownership is unknown; inspect the worker before recovery")
                try:
                    os.kill(row[1], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError("The worker is still running; wait for it to stop")
            original = json.loads(row[0]) if row[0] else None
            result = {"status": "reconciled", "verified": False, "output": "",
                      "reviewed_by": "user", "note": note, "previous_result": original}
            self._db.execute("UPDATE operations SET result = ? WHERE id = ?",
                             (json.dumps(result, ensure_ascii=False), operation_id))
            self._db.execute("DELETE FROM resources WHERE resource = ? AND operation_id = ?",
                             (resource, operation_id))
            self._db.execute("COMMIT")
            return True
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def latest_result(self, prefix: str) -> dict[str, Any] | None:
        row = self._db.execute(
            """SELECT result FROM operations WHERE substr(id, 1, ?) = ? AND result IS NOT NULL
               ORDER BY started_at DESC LIMIT 1""", (len(prefix), prefix),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def started_since(self, since: float) -> int:
        return int(self._db.execute(
            "SELECT COUNT(*) FROM operations WHERE started_at > ?", (since,)
        ).fetchone()[0])

    def close(self) -> None:
        self._db.close()
