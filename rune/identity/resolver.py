"""Identity resolver for RUNE.

Ported from src/identity/resolver.ts - maps channel-specific sender
identifiers to a unified user ID.

Default behaviour (single-user): all channels resolve to ``"default"``.
TUI/CLI channels resolve to ``"local_user"`` (optionally scoped by
workspace path).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)


class IdentityResolver:
    """Resolve channel senders to unified user identities.

    Maintains an in-memory cache backed by a lightweight SQLite store
    (``~/.rune/data/identity.db``).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._cache: dict[str, str] = {}
        self._db_path = db_path or (rune_data() / "identity.db")
        self._db: sqlite3.Connection | None = None

    # -- Database ------------------------------------------------------------

    def _ensure_db(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_map (
                cache_key  TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
        self._db = conn
        return conn

    def close(self) -> None:
        """Close the backing database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    # -- Resolution ----------------------------------------------------------

    def resolve_identity(self, channel: str, sender_id: str) -> str:
        """Resolve a *channel*/*sender_id* pair to a unified user ID.

        * **tui / cli** channels with an absolute path ``sender_id``
          produce ``"local_user:<path>"`` for workspace scoping.
        * **tui / cli** without a path produce ``"local_user"``.
        * All other channels first check the cache, then the DB,
          then fall back to ``"default"``.

        Parameters
        ----------
        channel:
            The channel type (``"tui"``, ``"cli"``, ``"telegram"``,
            ``"discord"``, etc.).
        sender_id:
            The sender identifier from the channel (user ID, chat ID,
            workspace path, etc.).

        Returns
        -------
        str
            A unified user ID string.
        """
        # TUI / CLI: local user (optionally scoped by workspace path)
        if channel in ("tui", "cli"):
            if sender_id and sender_id.startswith("/"):
                return f"local_user:{sender_id}"
            return "local_user"

        # Cache lookup
        cache_key = f"{channel}:{sender_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # DB lookup
        try:
            db = self._ensure_db()
            row = db.execute(
                "SELECT user_id FROM identity_map WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                user_id: str = row[0]
                self._cache[cache_key] = user_id
                return user_id
        except Exception as exc:
            log.debug("identity_lookup_failed", error=str(exc))

        # Single-user default
        user_id = "default"
        self._cache[cache_key] = user_id
        return user_id

    # -- Linking -------------------------------------------------------------

    def link(self, user_id: str, channel: str, sender_id: str) -> None:
        """Persistently associate a channel sender with a user ID.

        Subsequent calls to :meth:`resolve_identity` for the same
        ``(channel, sender_id)`` will return *user_id*.
        """
        cache_key = f"{channel}:{sender_id}"
        self._cache[cache_key] = user_id

        try:
            db = self._ensure_db()
            db.execute(
                """
                INSERT INTO identity_map (cache_key, user_id)
                VALUES (?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET user_id = excluded.user_id,
                              updated_at = datetime('now')
                """,
                (cache_key, user_id),
            )
            db.commit()
        except Exception as exc:
            log.warning("identity_link_failed", error=str(exc))

    # -- Convenience ---------------------------------------------------------

    def unlink(self, channel: str, sender_id: str) -> None:
        """Remove a channel-sender mapping."""
        cache_key = f"{channel}:{sender_id}"
        self._cache.pop(cache_key, None)
        try:
            db = self._ensure_db()
            db.execute(
                "DELETE FROM identity_map WHERE cache_key = ?",
                (cache_key,),
            )
            db.commit()
        except Exception as exc:
            log.warning("identity_unlink_failed", error=str(exc))

    def clear_cache(self) -> None:
        """Clear the in-memory cache (DB is untouched)."""
        self._cache.clear()


# ============================================================================
# Singleton
# ============================================================================

_resolver: IdentityResolver | None = None


def get_identity_resolver() -> IdentityResolver:
    """Return the global :class:`IdentityResolver` singleton."""
    global _resolver
    if _resolver is None:
        _resolver = IdentityResolver()
    return _resolver
