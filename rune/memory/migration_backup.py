"""Snapshot a SQLite database before a schema migration can drop or rewrite tables."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import apsw

from rune.utils.logger import get_logger

log = get_logger(__name__)


def backup_before_migration(conn: apsw.Connection, db_path: str, from_version: int, to_version: int) -> Path | None:
    """Write ``<db>.v<from>-before-v<to>.bak`` once, or raise and leave the database untouched.

    Fail closed: an unprotected migration of episode history is worse than a store that
    refuses to open. The backup is a consistent copy (``VACUUM INTO``) even under WAL.
    """
    if db_path in {"", ":memory:"} or db_path.startswith("file::memory:"):
        return None
    backup = Path(f"{db_path}.v{from_version}-before-v{to_version}.bak")
    if backup.exists():
        log.info("migration_backup_exists", path=str(backup))
        return backup
    # Write a uniquely named copy and rename it: a crash never leaves a partial file that
    # looks finished, and two processes migrating at once never swap half-written copies.
    partial = backup.with_name(f"{backup.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial")
    try:
        conn.execute("VACUUM INTO ?", (str(partial),))
        partial.replace(backup)
    except (apsw.Error, OSError) as exc:
        partial.unlink(missing_ok=True)
        if backup.exists():
            # Another process finished the same backup first.
            return backup
        raise RuntimeError(
            f"Could not back up {db_path} before migrating it from schema v{from_version}; "
            f"the database was left unchanged. Check free disk space and permissions ({exc})."
        ) from exc
    log.info("migration_backup", path=str(backup), from_version=from_version, to_version=to_version)
    return backup


def has_tables(conn: apsw.Connection) -> bool:
    """True when the file already holds a schema, even one written before user_version existed."""
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone() is not None
