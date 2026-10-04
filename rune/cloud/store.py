"""Operator-owned routing and login state, separate from conversation databases."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from rune.connectors.store import private_directory


def cloud_home() -> Path:
    return Path(os.environ.get("RUNE_CLOUD_HOME", "~/.rune-cloud")).expanduser()


def owner_key(owner: str) -> str:
    if not owner.strip() or len(owner) > 200:
        raise ValueError("Owner must contain 1–200 characters")
    return hashlib.sha256(owner.encode()).hexdigest()


def endpoint_origin(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Worker endpoint must be an HTTP(S) origin")
    if parsed.scheme == "http" and not ipaddress.ip_address(parsed.hostname).is_private:
        raise ValueError("Unencrypted worker transport is restricted to private IP addresses")
    return endpoint.rstrip("/")


class CloudStore:
    def __init__(self, root: Path):
        private_directory(root)
        self.root, self.path = root, root / "cloud.db"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workers (
                    owner TEXT PRIMARY KEY, endpoint TEXT NOT NULL UNIQUE,
                    upstream_token TEXT NOT NULL, login_hash TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    hash TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS machines (
                    owner TEXT PRIMARY KEY, slot INTEGER NOT NULL UNIQUE,
                    name TEXT NOT NULL UNIQUE, status TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, owner: str, endpoint: str, upstream_token: str) -> str:
        owner_key(owner)
        endpoint = endpoint_origin(endpoint)
        if not upstream_token.startswith("rune_") or any(c.isspace() for c in upstream_token):
            raise ValueError("A Rune worker API token is required")
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE owner=?", (owner,))
            db.execute("""INSERT INTO workers VALUES (?, ?, ?, ?, 1)
                          ON CONFLICT(owner) DO UPDATE SET endpoint=excluded.endpoint,
                          upstream_token=excluded.upstream_token, login_hash=excluded.login_hash, enabled=1""",
                       (owner, endpoint, upstream_token, self.digest(token)))
        return token

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def authenticate(self, token: str, *, session: bool = False) -> dict | None:
        ready = "AND NOT EXISTS (SELECT 1 FROM machines m WHERE m.owner=w.owner AND m.status!='running')"
        with self.connect() as db:
            if session:
                row = db.execute("""SELECT w.*, s.expires FROM workers w JOIN sessions s ON w.owner=s.owner
                                    WHERE s.hash=? AND s.expires>? AND w.enabled=1 """ + ready,
                                 (self.digest(token), time.time())).fetchone()
            else:
                row = db.execute("SELECT w.* FROM workers w WHERE login_hash=? AND enabled=1 " + ready, (self.digest(token),)).fetchone()
        return dict(row) if row else None

    def session(self, owner: str) -> str:
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires<=?", (time.time(),))
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (self.digest(token), owner, time.time() + 8 * 3600))
        return token

    def disable(self, owner: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE workers SET enabled=0 WHERE owner=?", (owner,))
            db.execute("DELETE FROM sessions WHERE owner=?", (owner,))

    def logout(self, token: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE hash=?", (self.digest(token),))

    def revoke_sessions(self, owner: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE owner=?", (owner,))

    def reserve(self, owner: str) -> dict:
        digest = owner_key(owner)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM machines WHERE owner=?", (owner,)).fetchone()
            if row:
                return dict(row)
            occupied = {r[0] for r in db.execute("SELECT slot FROM machines")}
            slot = next((s for s in range(1, 16383) if s not in occupied), None)
            if slot is None:
                raise ValueError("VM address pool exhausted")
            name = f"rune-{digest[:20]}"
            db.execute("INSERT INTO machines VALUES (?, ?, ?, 'reserved')", (owner, slot, name))
        return {"owner": owner, "slot": slot, "name": name, "status": "reserved"}

    def status(self, owner: str, status: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE machines SET status=? WHERE owner=?", (status, owner))
