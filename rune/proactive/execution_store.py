"""Store execution claims, results and tool records for recovery."""

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
        if path is not None:
            path.chmod(0o600)
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
        self._db.execute("""CREATE TABLE IF NOT EXISTS operation_usage (
            operation_id TEXT PRIMARY KEY, model_requests INTEGER NOT NULL, tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL, requests_known INTEGER NOT NULL,
            tokens_known INTEGER NOT NULL, cost_known INTEGER NOT NULL
        )""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS tool_attempts (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, record TEXT NOT NULL
        )""")
        self._db.execute("CREATE INDEX IF NOT EXISTS attempts_run ON tool_attempts(run_id)")
        self._db.execute("""CREATE TABLE IF NOT EXISTS continuations (
            parent_id TEXT PRIMARY KEY, child_id TEXT NOT NULL UNIQUE
        )""")
        from rune.proactive.routine_budget import recorded_usage

        for operation_id, raw in self._db.execute("""
            SELECT id, result FROM operations WHERE result IS NOT NULL
            AND id NOT IN (SELECT operation_id FROM operation_usage)
        """).fetchall():
            self._save_usage(operation_id, recorded_usage(json.loads(raw)))
        if self._db.execute("PRAGMA user_version").fetchone()[0] < 1:
            self._restore_approval_holds()
            self._db.execute("PRAGMA user_version=1")

    def _restore_approval_holds(self) -> None:
        # Restore legacy approval holds only for each task's latest run.
        for operation_id, in self._db.execute("""
            SELECT o.id FROM operations o WHERE json_extract(o.result, '$.status') = 'needs_approval'
            AND NOT EXISTS (SELECT 1 FROM continuations c WHERE c.parent_id = o.id)
        """).fetchall():
            parts = operation_id.split(":", 2)
            if len(parts) != 3 or parts[0] != "cron":
                continue
            prefix = f"cron:{parts[1]}:"
            latest = self._db.execute("SELECT id FROM operations WHERE id >= ? AND id < ? ORDER BY started_at DESC, rowid DESC LIMIT 1",
                                      (prefix, prefix + "\uffff")).fetchone()
            if latest and latest[0] == operation_id:
                self._db.execute("INSERT OR IGNORE INTO resources VALUES (?, ?)", (prefix[:-1], operation_id))

    def _save_usage(self, operation_id: str, usage: dict) -> None:
        self._db.execute("INSERT OR IGNORE INTO operation_usage VALUES (?, ?, ?, ?, ?, ?, ?)", (
            operation_id, usage["modelRequests"], usage["tokens"], usage["costUsd"],
            usage["requestsKnown"], usage["tokensKnown"], usage["costKnown"],
        ))

    def usage(self, prefix: str, *, exclude: str = "") -> dict:
        row = self._db.execute("""
            SELECT COALESCE(SUM(u.model_requests), 0), COALESCE(SUM(u.tokens), 0),
                   COALESCE(SUM(u.cost_usd), 0),
                   COUNT(*) - COALESCE(SUM(u.requests_known), 0),
                   COUNT(*) - COALESCE(SUM(u.tokens_known), 0),
                   COUNT(*) - COALESCE(SUM(u.cost_known), 0)
            FROM operations o LEFT JOIN operation_usage u ON o.id = u.operation_id
            WHERE o.id >= ? AND o.id < ? AND o.id != ?
        """, (prefix, prefix + "\uffff", exclude)).fetchone()
        return {"modelRequests": row[0], "tokens": row[1], "costUsd": row[2],
                "requestsKnown": row[3] == 0, "tokensKnown": row[4] == 0, "costKnown": row[5] == 0}

    def get(self, operation_id: str) -> tuple[str, dict[str, Any] | None] | None:
        row = self._db.execute(
            "SELECT fingerprint, result FROM operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return (row[0], json.loads(row[1]) if row[1] else None) if row else None

    @staticmethod
    def _state(result: dict | None, owner_pid: int | None) -> str:
        if result is not None:
            return "recovery_required" if result.get("execution_unknown") else result.get("status", "unverified")
        if owner_pid is not None:
            try:
                os.kill(owner_pid, 0)
                return "running"
            except OSError:
                return "recovery_required"
        return "recovery_required"

    def state(self, operation_id: str) -> str:
        row = self._db.execute("SELECT result, owner_pid FROM operations WHERE id = ?", (operation_id,)).fetchone()
        return self._state(json.loads(row[0]) if row[0] else None, row[1]) if row else "unavailable"

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
            if changed:
                from rune.proactive.routine_budget import recorded_usage

                self._save_usage(operation_id, recorded_usage(result))
            # Pending approvals and uncertain outcomes keep the task blocked.
            if changed and result.get("status") not in {"interrupted", "needs_approval"} and not result.get("execution_unknown"):
                self._db.execute("DELETE FROM resources WHERE operation_id = ?", (operation_id,))
                if result.get("status") in {"completed", "verified", "unchanged"}:
                    self._clear_attempts(operation_id)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def recent(self, prefix: str, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(id=row[0], started_at=row[1], result=json.loads(row[2]) if row[2] else None,
                     blocked=bool(row[3]), state=self._state(json.loads(row[2]) if row[2] else None, row[4])) for row in self._db.execute(
            """SELECT o.id, o.started_at, o.result, r.resource, o.owner_pid FROM operations o
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
            if original and original.get("status") == "needs_approval":
                raise ValueError("Approve or dismiss the pending action before clearing this task")
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

    def latest_result(self, prefix: str, *, executed_only: bool = False) -> dict[str, Any] | None:
        row = self._db.execute(
            """SELECT result FROM operations WHERE substr(id, 1, ?) = ? AND result IS NOT NULL
               AND (? = 0 OR json_extract(result, '$.not_started') IS NOT 1)
               ORDER BY started_at DESC, rowid DESC LIMIT 1""", (len(prefix), prefix, executed_only),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_attempt(self, record: dict[str, Any]) -> None:
        self._db.execute("INSERT INTO tool_attempts VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                         (record["id"], record["run_id"], json.dumps(record, ensure_ascii=False)))

    def attempts(self, run_id: str) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self._db.execute(
            "SELECT record FROM tool_attempts WHERE run_id = ? ORDER BY rowid", (run_id,))]

    def history(self, operation_id: str) -> list[dict[str, Any]]:
        chain, seen = [], set()
        while operation_id:
            if operation_id in seen or len(seen) >= 100:
                raise ValueError("The task's continuation history is invalid or too long")
            seen.add(operation_id)
            chain.append(self.attempts(operation_id))
            row = self._db.execute("SELECT parent_id FROM continuations WHERE child_id = ?", (operation_id,)).fetchone()
            operation_id = row[0] if row else ""
        return [record for records in reversed(chain) for record in records]

    def continued(self, operation_id: str) -> str | None:
        row = self._db.execute("SELECT child_id FROM continuations WHERE parent_id = ?", (operation_id,)).fetchone()
        return row[0] if row else None

    def _clear_attempts(self, operation_id: str) -> None:
        # Drop recovery payloads after completion; keep outcomes and usage.
        seen = set()
        while operation_id and operation_id not in seen:
            seen.add(operation_id)
            self._db.execute("DELETE FROM tool_attempts WHERE run_id = ?", (operation_id,))
            row = self._db.execute("SELECT parent_id FROM continuations WHERE child_id = ?", (operation_id,)).fetchone()
            operation_id = row[0] if row else ""

    def continue_approval(self, resource: str, operation_id: str, child_id: str, fingerprint: str) -> None:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute("SELECT operation_id FROM resources WHERE resource = ?", (resource,)).fetchone()
            saved = self.get(operation_id)
            if (not row or row[0] != operation_id or not saved or saved[0] != fingerprint
                    or (saved[1] or {}).get("status") != "needs_approval" or self.continued(operation_id)):
                raise ValueError("This approval is no longer pending or the task changed")
            self._db.execute("INSERT INTO operations (id, fingerprint, started_at, owner_pid) VALUES (?, ?, ?, ?)",
                             (child_id, fingerprint, time.time(), os.getpid()))
            self._db.execute("INSERT INTO continuations VALUES (?, ?)", (operation_id, child_id))
            previous = {**saved[1], "status": "continued", "continued_as": child_id}
            self._db.execute("UPDATE operations SET result = ? WHERE id = ?", (json.dumps(previous), operation_id))
            self._db.execute("UPDATE resources SET operation_id = ? WHERE resource = ?", (child_id, resource))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def dismiss_approval(self, resource: str, operation_id: str) -> None:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute("SELECT operation_id FROM resources WHERE resource = ?", (resource,)).fetchone()
            saved = self.get(operation_id)
            if not row or row[0] != operation_id or not saved or (saved[1] or {}).get("status") != "needs_approval":
                raise ValueError("This approval is no longer pending")
            result = {**saved[1], "status": "dismissed", "success": False, "verified": False}
            self._db.execute("UPDATE operations SET result = ? WHERE id = ?", (json.dumps(result), operation_id))
            self._db.execute("DELETE FROM resources WHERE resource = ?", (resource,))
            self._clear_attempts(operation_id)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def started_since(self, since: float) -> int:
        return int(self._db.execute(
            "SELECT COUNT(*) FROM operations WHERE started_at > ?", (since,)
        ).fetchone()[0])

    def close(self) -> None:
        self._db.close()
