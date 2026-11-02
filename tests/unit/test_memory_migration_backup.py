"""Migrations of the episode store are backed up first and never stamp a newer schema down."""

import apsw
import pytest

from rune.memory import store as store_module
from rune.memory.store import MemoryStore


def _stamp(path, version):
    conn = apsw.Connection(str(path))
    conn.execute(f"PRAGMA user_version = {version}")
    conn.close()


def _open(path):
    store = MemoryStore(path)
    _ = store.conn
    store.close()


def _version(path):
    conn = apsw.Connection(str(path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_fresh_database_needs_no_backup(tmp_path):
    db = tmp_path / "memory.db"
    _open(db)
    assert _version(db) == store_module._CURRENT_SCHEMA_VERSION
    assert not list(tmp_path.glob("*.bak"))


def test_older_database_is_backed_up_once_before_migrating(tmp_path):
    db = tmp_path / "memory.db"
    current = store_module._CURRENT_SCHEMA_VERSION
    first = MemoryStore(db)
    first.conn.execute("INSERT INTO cron_jobs (id, name, schedule, command) VALUES ('j1', 'n', '* * * * *', 'c')")
    first.close()
    _stamp(db, current - 1)
    _open(db)
    backup = tmp_path / f"memory.db.v{current - 1}-before-v{current}.bak"
    assert backup.is_file() and _version(db) == current
    copy = apsw.Connection(str(backup))
    assert copy.execute("SELECT id FROM cron_jobs").fetchall() == [("j1",)]
    assert copy.execute("PRAGMA user_version").fetchone()[0] == current - 1
    copy.close()
    assert not list(tmp_path.glob("*.partial"))


def test_newer_database_is_not_stamped_down(tmp_path):
    db = tmp_path / "memory.db"
    _open(db)
    newer = store_module._CURRENT_SCHEMA_VERSION + 3
    _stamp(db, newer)
    _open(db)
    assert _version(db) == newer
    assert not list(tmp_path.glob("*.bak"))


def test_failed_backup_refuses_to_migrate_and_does_not_cache_the_connection(tmp_path, monkeypatch):
    db = tmp_path / "memory.db"
    MemoryStore(db).close()
    old = store_module._CURRENT_SCHEMA_VERSION - 1
    _stamp(db, old)

    def broken(*_args, **_kwargs):
        raise RuntimeError("Could not back up")

    monkeypatch.setattr("rune.memory.migration_backup.backup_before_migration", broken)
    store = MemoryStore(db)
    with pytest.raises(RuntimeError, match="Could not back up"):
        _ = store.conn
    assert store._conn is None and _version(db) == old


def test_backup_failure_leaves_no_partial_file(tmp_path):
    from rune.memory.migration_backup import backup_before_migration

    db = tmp_path / "memory.db"
    conn = apsw.Connection(str(db))
    conn.execute("CREATE TABLE t (x)")
    locked = tmp_path / "locked"
    locked.mkdir()
    target = locked / "memory.db"
    locked.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match="left unchanged"):
            backup_before_migration(conn, str(target), 3, 4)
        assert not [p for p in locked.iterdir()]
    finally:
        locked.chmod(0o700)
        conn.close()


def test_unversioned_database_with_data_is_backed_up(tmp_path):
    db = tmp_path / "memory.db"
    conn = apsw.Connection(str(db))
    conn.execute("CREATE TABLE legacy_notes (id TEXT)")
    conn.execute("INSERT INTO legacy_notes VALUES ('keep-me')")
    conn.close()
    _open(db)
    backups = list(tmp_path.glob("memory.db.v0-before-*.bak"))
    assert len(backups) == 1
    copy = apsw.Connection(str(backups[0]))
    assert copy.execute("SELECT id FROM legacy_notes").fetchall() == [("keep-me",)]
    copy.close()


def test_concurrent_backup_by_another_process_counts_as_done(tmp_path, monkeypatch):
    from rune.memory import migration_backup

    db = tmp_path / "memory.db"
    conn = apsw.Connection(str(db))
    conn.execute("CREATE TABLE t (x)")
    target = tmp_path / "memory.db.v3-before-v4.bak"

    class Racing:
        def execute(self, sql, params):
            # The other process renamed its finished copy into place while ours was writing.
            target.write_text("finished by the other process")
            raise apsw.IOError("disk I/O error")

    assert migration_backup.backup_before_migration(Racing(), str(db), 3, 4) == target
    assert target.read_text() == "finished by the other process"
    assert not list(tmp_path.glob("*.partial"))
    conn.close()
