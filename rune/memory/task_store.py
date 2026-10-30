"""Store task records by owner and workspace."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class TaskConflict(RuntimeError):
    pass


class TaskStore:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from rune.utils.paths import rune_data

            path = rune_data() / "tasks.db"
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=1, isolation_level=None)
        try:
            self.path.chmod(0o600)
            db.row_factory = sqlite3.Row
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                owner TEXT NOT NULL, workspace TEXT NOT NULL, id TEXT NOT NULL,
                data TEXT NOT NULL, revision INTEGER NOT NULL, updated REAL NOT NULL,
                lease TEXT NOT NULL DEFAULT '', pid INTEGER,
                PRIMARY KEY (owner, workspace, id))""")
            db.execute("CREATE INDEX IF NOT EXISTS task_recent ON tasks(owner, workspace, updated DESC)")
            yield db
        finally:
            db.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        record = json.loads(row["data"])
        return {**record, "revision": row["revision"], "updated_at": row["updated"]}

    def get(self, owner: str, workspace: str, task_id: str) -> dict | None:
        if not self.path.exists():
            return None
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE owner=? AND workspace=? AND id=?",
                (owner, workspace, task_id),
            ).fetchone()
            return self._record(row) if row is not None else None

    def list(self, owner: str, workspace: str, *, status: str = "", category: str = "",
             limit: int = 20, offset: int = 0, unfinished_first: bool = False) -> list[dict]:
        if not self.path.exists():
            return []
        query = "SELECT * FROM tasks WHERE owner=? AND workspace=?"
        args: list[Any] = [owner, workspace]
        for field, value in (("status", status), ("category", category)):
            if value:
                query += f" AND json_extract(data, '$.{field}')=?"
                args.append(value)
        query += " ORDER BY "
        if unfinished_first:
            query += "json_extract(data, '$.status') IN ('completed','verified'), "
        query += "updated DESC, id LIMIT ? OFFSET ?"
        args.extend((max(1, min(limit, 100)), max(0, offset)))
        with self._connect() as db:
            return [self._record(row) for row in db.execute(query, args)]

    def create(self, owner: str, workspace: str, records: Sequence[dict]) -> None:
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            for record in records:
                db.execute("INSERT INTO tasks(owner, workspace, id, data, revision, updated) VALUES(?,?,?,?,1,?)",
                           (owner, workspace, record["id"], json.dumps(record, ensure_ascii=False), time.time()))

    @staticmethod
    def _busy(row: sqlite3.Row | None, lease: str = "") -> bool:
        if not row or not row["lease"] or row["lease"] == lease:
            return False
        # A worker that exited no longer owns its task lease.
        try:
            os.kill(row["pid"], 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def update(self, owner: str, workspace: str, task_id: str, changes: dict) -> dict | None:
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE owner=? AND workspace=? AND id=?",
                             (owner, workspace, task_id)).fetchone()
            if row is None:
                return None
            if self._busy(row):
                raise TaskConflict("This task is running. Wait for it to stop before changing its record.")
            data = json.loads(row["data"])
            if data.get("kind") != "manual":
                raise TaskConflict("This record follows an execution. Continue the task in chat; use cron_update to change a schedule.")
            data.update(changes)
            db.execute("UPDATE tasks SET data=?, revision=revision+1, updated=? WHERE owner=? AND workspace=? AND id=?",
                       (json.dumps(data, ensure_ascii=False), time.time(), owner, workspace, task_id))
        return self.get(owner, workspace, task_id)

    def claim(self, owner: str, workspace: str, task_id: str, revision: int, lease: str) -> None:
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE owner=? AND workspace=? AND id=?",
                             (owner, workspace, task_id)).fetchone()
            if row is None or row["revision"] != revision:
                raise TaskConflict("The task changed since it was loaded. Reload its latest state before continuing.")
            if self._busy(row, lease):
                raise TaskConflict("This task is already running in another conversation. No task action was started.")
            db.execute("UPDATE tasks SET lease=?, pid=? WHERE owner=? AND workspace=? AND id=?",
                       (lease, os.getpid(), owner, workspace, task_id))

    def checkpoint(self, owner: str, workspace: str, task: dict, *, lease: str,
                   conversation_id: str = "", kind: str = "conversation", routine_id: str = "") -> int:
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE owner=? AND workspace=? AND id=?",
                             (owner, workspace, task["id"])).fetchone()
            if self._busy(row, lease):
                raise TaskConflict("The task is being updated by another run.")
            if row and row["revision"] != task.get("_revision"):
                raise TaskConflict("The task changed before its result could be saved.")
            previous = json.loads(row["data"]) if row else {}
            checkpoint = {key: value for key, value in task.items() if not key.startswith("_")}
            data = {**previous, "id": task["id"], "goal": task["request"],
                    "status": task.get("status", "pending"), "kind": kind,
                    "category": task.get("goal_type", previous.get("category", "project")),
                    "checkpoint": checkpoint, "conversation_id": conversation_id,
                    "routine_id": routine_id or previous.get("routine_id", "")}
            data.pop("progress", None)
            revision = (row["revision"] if row else 0) + 1
            db.execute("""INSERT INTO tasks(owner,workspace,id,data,revision,updated,lease,pid) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(owner,workspace,id) DO UPDATE SET data=excluded.data, revision=excluded.revision,
                updated=excluded.updated, lease=excluded.lease, pid=excluded.pid""",
                (owner, workspace, task["id"], json.dumps(data, ensure_ascii=False), revision,
                 time.time(), lease, os.getpid() if lease else None))
        return revision

    def release(self, owner: str, workspace: str, lease: str) -> None:
        if not self.path.exists():
            return
        with self._connect() as db:
            db.execute("UPDATE tasks SET lease='', pid=NULL WHERE owner=? AND workspace=? AND lease=?",
                       (owner, workspace, lease))
