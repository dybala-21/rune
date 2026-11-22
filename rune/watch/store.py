"""What the watch keeps between runs: the owner's check time, the watched pages, what each read gave."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

from rune.watch.item import FETCH, Item, Price

MAX_WATCHING = 20
KEEP_OBSERVATIONS = 60
DEFAULT_TIME = time(9, 0)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL UNIQUE, fetch_url TEXT NOT NULL, host TEXT NOT NULL,
    label TEXT NOT NULL, tell_when TEXT NOT NULL, below TEXT, currency TEXT NOT NULL,
    told_low TEXT NOT NULL, told_high TEXT NOT NULL, told_at TEXT NOT NULL, renewed_at TEXT NOT NULL,
    candidate_low TEXT, candidate_high TEXT, candidate_at TEXT, confirm_tried INTEGER NOT NULL DEFAULT 0,
    read_day TEXT, failed_days INTEGER NOT NULL DEFAULT 0, counted_days INTEGER NOT NULL DEFAULT 0,
    gone_days INTEGER NOT NULL DEFAULT 0, told_failure INTEGER NOT NULL DEFAULT 0,
    last_low TEXT, last_high TEXT, last_read_at TEXT,
    status TEXT NOT NULL DEFAULT 'watching', ended_why TEXT NOT NULL DEFAULT '', ended_at TEXT,
    version INTEGER NOT NULL DEFAULT 0, via TEXT NOT NULL DEFAULT 'fetch'
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL, observed_at TEXT NOT NULL,
    http_status INTEGER NOT NULL DEFAULT 0, low TEXT, high TEXT, offers INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '', failure TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS observations_item ON observations(item_id, id);
"""
# The columns a save writes, in the order `_values` gives them.
_SAVED = ("fetch_url", "label", "tell_when", "below", "currency", "told_low", "told_high", "told_at", "renewed_at",
          "candidate_low", "candidate_high", "candidate_at", "confirm_tried", "read_day", "failed_days",
          "counted_days", "gone_days", "told_failure", "last_low", "last_high", "last_read_at", "status",
          "ended_why", "ended_at", "via")


class TooMany(Exception):
    """As many pages are being watched as the watch takes on."""


def default_path() -> Path:
    from rune.utils.paths import rune_data

    return rune_data() / "watch.db"


class Store:
    def __init__(self, path: Path | None = None) -> None:
        path = path or default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), isolation_level=None)
        path.chmod(0o600)  # the addresses an owner watches are theirs alone
        self._db.row_factory = sqlite3.Row
        # The chat's tools and the scheduled run both write here.
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_SCHEMA)
        _keep_way(self._db)

    def close(self) -> None:
        self._db.close()

    def check_time(self) -> time:
        """When in the owner's day pages are read; nine in the morning until they say otherwise."""
        row = self._db.execute("SELECT value FROM settings WHERE key = 'check_time'").fetchone()
        return time.fromisoformat(row["value"]) if row else DEFAULT_TIME

    def set_check_time(self, when: time) -> None:
        self._db.execute("INSERT INTO settings VALUES ('check_time', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                         (when.replace(second=0, microsecond=0).isoformat(timespec="minutes"),))

    def add(self, *, url: str, fetch_url: str, host: str, label: str, tell_when: str, below: Decimal | None,
            price: Price, now: datetime, via: str = FETCH) -> Item:
        """Start watching a page from the price just read. Raises ``TooMany`` at the limit."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            watching = self._db.execute("SELECT COUNT(*) FROM items WHERE status = 'watching'").fetchone()[0]
            if watching >= MAX_WATCHING:
                raise TooMany
            # The reading it starts from is that day's read: the page is not asked again the same day.
            item = Item(id=0, url=url, fetch_url=fetch_url, host=host, label=label, tell_when=tell_when, below=below,
                        told=price, told_at=now, renewed_at=now, last_read=price, last_read_at=now,
                        read_day=now.date(), via=via)
            columns = ("url", "host", *_SAVED)
            cursor = self._db.execute(
                f"INSERT INTO items ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                (url, host, *_values(item)))
            item.id = int(cursor.lastrowid or 0)
            self._db.execute("COMMIT")
            return item
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def get(self, item_id: int) -> Item | None:
        row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _item(row) if row else None

    def by_url(self, url: str) -> Item | None:
        row = self._db.execute("SELECT * FROM items WHERE url = ?", (url,)).fetchone()
        return _item(row) if row else None

    def watching(self) -> list[Item]:
        # Host by host, so one site's pages are read one after another with a pause between.
        return [_item(row) for row in self._db.execute("SELECT * FROM items WHERE status = 'watching' ORDER BY host, id")]

    def everything(self) -> list[Item]:
        return [_item(row) for row in self._db.execute("SELECT * FROM items ORDER BY status DESC, id")]

    def save(self, item: Item) -> bool:
        """Write the item back. False when it was saved by someone else since this copy was read
        (the owner changed it while its page was being read): nothing is written, and the copy is stale."""
        if not self._write(item):
            return False
        item.version += 1
        return True

    def save_all(self, items: list[Item]) -> list[Item]:
        """Write several items back as one change, so a run's readings are kept whole or not at all.
        Returns the ones written; an item saved by someone else since is left out, as in ``save``."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            written = [item for item in items if self._write(item)]
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        for item in written:
            item.version += 1
        return written

    def _write(self, item: Item) -> bool:
        assignments = ", ".join(f"{column} = ?" for column in _SAVED)
        cursor = self._db.execute(
            f"UPDATE items SET {assignments}, version = version + 1 WHERE id = ? AND version = ?",
            (*_values(item), item.id, item.version))
        return cursor.rowcount == 1

    def observed(self, item_id: int, at: datetime, *, http_status: int = 0, price: Price | None = None,
                 offers: int = 0, source: str = "", failure: str = "") -> None:
        """Keep what one read gave: the price, or the kind of failure. Only the latest are kept."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute(
                "INSERT INTO observations (item_id, observed_at, http_status, low, high, offers, source, failure) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, at.isoformat(), http_status, _text(price.low) if price else None,
                 _text(price.high) if price else None, offers, source, failure))
            self._db.execute(
                "DELETE FROM observations WHERE item_id = ? AND id NOT IN "
                "(SELECT id FROM observations WHERE item_id = ? ORDER BY id DESC LIMIT ?)",
                (item_id, item_id, KEEP_OBSERVATIONS))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def observations(self, item_id: int) -> list[dict[str, Any]]:
        """What the item's reads gave, latest first."""
        return [dict(row) for row in self._db.execute(
            "SELECT observed_at, http_status, low, high, offers, source, failure FROM observations "
            "WHERE item_id = ? ORDER BY id DESC", (item_id,))]

    def forget_ended(self, before: datetime) -> int:
        """Drop items that ended before ``before``, with what was read for them."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            ended = [row[0] for row in self._db.execute(
                "SELECT id FROM items WHERE status = 'ended' AND ended_at IS NOT NULL AND ended_at < ?",
                (before.isoformat(),))]
            for item_id in ended:
                self._db.execute("DELETE FROM observations WHERE item_id = ?", (item_id,))
                self._db.execute("DELETE FROM items WHERE id = ?", (item_id,))
            self._db.execute("COMMIT")
            return len(ended)
        except BaseException:
            self._db.execute("ROLLBACK")
            raise


def _keep_way(db: sqlite3.Connection) -> None:
    """Give a file kept before pages were opened in a browser the column for the way each is read.
    Every item in such a file was asked for."""
    def there() -> bool:
        return "via" in {row["name"] for row in db.execute("PRAGMA table_info(items)")}

    if there():
        return
    try:
        db.execute("ALTER TABLE items ADD COLUMN via TEXT NOT NULL DEFAULT 'fetch'")
    except sqlite3.OperationalError:
        # The chat's tools and the scheduled run opened the file at once: the other one added it.
        if not there():
            raise


def _text(amount: Decimal) -> str:
    """An amount as plain digits: 56.00 and 56 are kept, and compared, as the same price."""
    return format(amount.normalize(), "f")


def _when(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


def _values(item: Item) -> tuple[Any, ...]:
    candidate, last = item.candidate, item.last_read
    return (item.fetch_url, item.label, item.tell_when, _text(item.below) if item.below is not None else None,
            item.told.currency, _text(item.told.low), _text(item.told.high), item.told_at.isoformat(),
            item.renewed_at.isoformat(), _text(candidate.low) if candidate else None,
            _text(candidate.high) if candidate else None, _when(item.candidate_at), int(item.confirm_tried),
            item.read_day.isoformat() if item.read_day else None, item.failed_days, item.counted_days, item.gone_days,
            int(item.told_failure), _text(last.low) if last else None, _text(last.high) if last else None,
            _when(item.last_read_at), item.status, item.ended_why, _when(item.ended_at), item.via)


def _item(row: sqlite3.Row) -> Item:
    currency = row["currency"]

    def price(low: str | None, high: str | None) -> Price | None:
        return Price(Decimal(low), Decimal(high), currency) if low is not None and high is not None else None

    def moment(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    told = price(row["told_low"], row["told_high"])
    assert told is not None
    return Item(
        id=row["id"], url=row["url"], fetch_url=row["fetch_url"], host=row["host"], label=row["label"],
        tell_when=row["tell_when"], below=Decimal(row["below"]) if row["below"] is not None else None, told=told,
        told_at=datetime.fromisoformat(row["told_at"]), renewed_at=datetime.fromisoformat(row["renewed_at"]),
        candidate=price(row["candidate_low"], row["candidate_high"]), candidate_at=moment(row["candidate_at"]),
        confirm_tried=bool(row["confirm_tried"]), read_day=date.fromisoformat(row["read_day"]) if row["read_day"] else None,
        failed_days=row["failed_days"], counted_days=row["counted_days"], gone_days=row["gone_days"],
        told_failure=bool(row["told_failure"]), last_read=price(row["last_low"], row["last_high"]),
        last_read_at=moment(row["last_read_at"]), status=row["status"], ended_why=row["ended_why"],
        ended_at=moment(row["ended_at"]), via=row["via"], version=row["version"])
