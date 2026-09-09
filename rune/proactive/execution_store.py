"""Persist execution claims and results for proactive suggestions.

An unfinished claim after restart has an unknown outcome: the external action
may have completed before the result was saved.
"""

from __future__ import annotations

import json
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

    def get(self, operation_id: str) -> tuple[str, dict[str, Any] | None] | None:
        row = self._db.execute(
            "SELECT fingerprint, result FROM operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return (row[0], json.loads(row[1]) if row[1] else None) if row else None

    def claim(self, operation_id: str, fingerprint: str, limit: int) -> str:
        """Reserve the operation ID and hourly budget in one transaction."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            if self.get(operation_id) is not None:
                decision = "exists"
            elif self.started_since(time.time() - 3600) >= limit:
                decision = "rate_limited"
            else:
                self._db.execute(
                    "INSERT INTO operations (id, fingerprint, started_at) VALUES (?, ?, ?)",
                    (operation_id, fingerprint, time.time()),
                )
                decision = "claimed"
            self._db.execute("COMMIT")
            return decision
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def finish(self, operation_id: str, result: dict[str, Any]) -> None:
        self._db.execute(
            "UPDATE operations SET result = ? WHERE id = ? AND result IS NULL",
            (json.dumps(result, ensure_ascii=False), operation_id),
        )

    def started_since(self, since: float) -> int:
        return int(self._db.execute(
            "SELECT COUNT(*) FROM operations WHERE started_at > ?", (since,)
        ).fetchone()[0])

    def close(self) -> None:
        self._db.close()
