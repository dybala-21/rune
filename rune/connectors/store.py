"""Broker-owned policy and credentials. No secret is returned by listing connectors."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from rune.connectors.models import ConnectorPolicy


def broker_home() -> Path:
    return Path(os.environ.get("RUNE_BROKER_HOME", "~/.rune-broker")).expanduser()


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Private state directory must not be a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


class ConnectorStore:
    def __init__(self, root: Path):
        private_directory(root)
        self.path = root / "connectors.db"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS connectors (
                    name TEXT PRIMARY KEY, policy TEXT NOT NULL, secret TEXT NOT NULL, revision TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nonces (nonce TEXT PRIMARY KEY, expires REAL NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, policy: ConnectorPolicy, secret: str) -> None:
        if not 8 <= len(secret) <= 16384 or any(ord(c) < 32 or ord(c) > 126 for c in secret):
            raise ValueError("Credential must contain 8–16384 printable ASCII characters")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO connectors VALUES (?, ?, ?, ?)",
                       (policy.name, policy.model_dump_json(), secret, secrets.token_hex(16)))
        (self.path.parent / "configured").touch(mode=0o600)

    def get(self, name: str) -> tuple[ConnectorPolicy, str, str]:
        with self.connect() as db:
            row = db.execute("SELECT policy, secret, revision FROM connectors WHERE name=?", (name,)).fetchone()
        if row is None:
            raise ValueError("Connector is not configured")
        return ConnectorPolicy.model_validate_json(row[0]), row[1], row[2]

    def list(self) -> list[dict]:
        with self.connect() as db:
            return [{**ConnectorPolicy.model_validate_json(row[0]).model_dump(), "revision": row[1]}
                    for row in db.execute("SELECT policy, revision FROM connectors ORDER BY name")]

    def remove(self, name: str) -> None:
        with self.connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("DELETE FROM connectors WHERE name=?", (name,))
            if db.execute("SELECT COUNT(*) FROM connectors").fetchone()[0] == 0:
                (self.path.parent / "configured").unlink(missing_ok=True)

    def redeem(self, nonce: str, expires: int) -> bool:
        now = time.time()
        if not now <= expires <= now + 60 or len(nonce) != 64:
            return False
        with self.connect() as db:
            db.execute("DELETE FROM nonces WHERE expires < ?", (now,))
            try:
                db.execute("INSERT INTO nonces VALUES (?, ?)", (nonce, expires))
            except sqlite3.IntegrityError:
                return False
        return True


def authority_key(root: Path, *, create: bool = False) -> bytes:
    path = root / "controller.key"
    if create and not path.exists():
        private_directory(root)
        with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600), "wb") as stream:
            stream.write(secrets.token_bytes(32))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if info.st_mode & 0o077:
            raise ValueError("Controller key must be owner-only")
        value = stream.read(33)
    if len(value) != 32:
        raise ValueError("Invalid controller key")
    return value


def signature(key: bytes, operation: str, body: bytes, nonce: str, expires: int) -> str:
    payload = b"\n".join((operation.encode(), str(expires).encode(), nonce.encode(), body))
    return hmac.new(key, payload, hashlib.sha256).hexdigest()
