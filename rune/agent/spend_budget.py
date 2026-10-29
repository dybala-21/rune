"""Reserve shared daily and monthly budgets before sending model requests."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rune.agent.request_budget import BudgetExceeded, rejected_before_generation
from rune.utils.paths import rune_data


def limits():
    from rune.config.loader import get_config
    return get_config().runtime


def enabled() -> bool:
    config = limits()
    return config.daily_cost_usd is not None or config.monthly_cost_usd is not None


class SpendLedger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or rune_data() / "spend.db"

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=1, isolation_level=None)
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS spend (
                id TEXT PRIMARY KEY, day TEXT NOT NULL, amount INTEGER NOT NULL, state TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS spend_day ON spend(day)")
            db.execute("""CREATE TABLE IF NOT EXISTS spend_periods (
                period TEXT PRIMARY KEY, spent INTEGER NOT NULL DEFAULT 0,
                held INTEGER NOT NULL DEFAULT 0, unknown INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS spend_reconciliations (
                request_id TEXT PRIMARY KEY, held INTEGER NOT NULL, cost INTEGER NOT NULL,
                evidence TEXT NOT NULL, reconciled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("CREATE TABLE IF NOT EXISTS spend_owners (request_id TEXT PRIMARY KEY, owner TEXT NOT NULL)")
        except BaseException:
            db.close()
            raise
        return db

    def reserve(self, cost: float, daily: float | None, monthly: float | None,
                *, now: datetime | None = None) -> str:
        day = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d")
        amount = math.ceil(cost * 1_000_000)
        request_id = uuid4().hex
        from rune.agent.spend_owner import owner_id
        owner = owner_id(self.path)
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            for prefix, limit, period in ((day, daily, "daily"), (day[:7], monthly, "monthly")):
                db.execute("INSERT OR IGNORE INTO spend_periods(period) VALUES (?)", (prefix,))
                spent, held, unknown = db.execute(
                    "SELECT spent, held, unknown FROM spend_periods WHERE period=?", (prefix,),
                ).fetchone()
                if limit is not None:
                    if unknown:
                        raise BudgetExceeded(f"A previous request has unknown usage; review it with `rune budget status` before using the shared {period} budget.")
                    if spent + held + amount > math.floor(limit * 1_000_000):
                        raise BudgetExceeded(f"The shared {period} model budget cannot cover this request. No request was sent.")
                db.execute("UPDATE spend_periods SET held=held+? WHERE period=?", (amount, prefix))
            db.execute("INSERT INTO spend VALUES (?, ?, ?, 'reserved')", (request_id, day, amount))
            db.execute("INSERT INTO spend_owners VALUES (?, ?)", (request_id, owner))
            db.execute("COMMIT")
            return request_id
        finally:
            db.close()

    def settle(self, request_id: str, cost: float | None, *, rejected: bool = False) -> None:
        self._settle(request_id, cost, rejected=rejected, evidence="provider_usage")

    def reconcile(self, request_id: str, cost: float, evidence: str) -> None:
        if not evidence.strip() or len(evidence) > 1000:
            raise ValueError("Provide a short reference to the confirmed usage or invoice.")
        self.recover()
        self._settle(request_id, cost, evidence=evidence.strip(), manual=True)

    def recover(self) -> int:
        """Mark abandoned reservations as unknown and keep their budget reserved."""
        if not self.path.exists():
            return 0
        from rune.agent.spend_owner import exited
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            owners = db.execute("SELECT DISTINCT owner FROM spend_owners").fetchall()
            recovered = 0
            for (owner,) in owners:
                if not exited(self.path, owner):
                    continue
                rows = db.execute("SELECT s.id, s.day FROM spend s JOIN spend_owners o ON o.request_id=s.id "
                                  "WHERE o.owner=? AND s.state='reserved'", (owner,)).fetchall()
                for request_id, day in rows:
                    db.execute("UPDATE spend SET state='unknown' WHERE id=?", (request_id,))
                    db.execute("UPDATE spend_periods SET unknown=unknown+1 WHERE period IN (?, ?)", (day, day[:7]))
                recovered += len(rows)
                db.execute("DELETE FROM spend_owners WHERE owner=?", (owner,))
            db.execute("COMMIT")
            return recovered
        finally:
            db.close()

    def unresolved(self) -> list[dict]:
        if not self.path.exists():
            return []
        db = self._connect()
        try:
            return [{"id": row[0], "day": row[1], "heldUsd": row[2] / 1_000_000, "state": row[3]}
                    for row in db.execute("SELECT id, day, amount, state FROM spend WHERE state != 'settled' ORDER BY day, rowid LIMIT 100")]
        finally:
            db.close()

    def _settle(self, request_id: str, cost: float | None, *, rejected: bool = False,
                evidence: str, manual: bool = False) -> None:
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError("Confirmed cost must be finite and non-negative.")
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT day, amount, state FROM spend WHERE id=?", (request_id,)).fetchone()
            if manual and (row is None or row[2] != "unknown"):
                raise ValueError("Only requests with unknown usage can be reconciled; active reservations are unchanged.")
            if row is None or row[2] == "settled":
                return
            day, held, state = row
            if rejected or cost is not None:
                amount = 0 if rejected else math.ceil(cost * 1_000_000)
                db.execute("UPDATE spend SET amount=?, state='settled' WHERE id=?", (amount, request_id))
                db.execute("DELETE FROM spend_owners WHERE request_id=?", (request_id,))
                db.execute("UPDATE spend_periods SET spent=spent+?, held=held-?, unknown=unknown-? WHERE period IN (?, ?)",
                           (amount, held, int(state == "unknown"), day, day[:7]))
                if state == "unknown":
                    db.execute("INSERT INTO spend_reconciliations(request_id, held, cost, evidence) VALUES (?, ?, ?, ?)",
                               (request_id, held, amount, evidence))
            elif state == "reserved":
                db.execute("UPDATE spend SET state='unknown' WHERE id=?", (request_id,))
                db.execute("UPDATE spend_periods SET unknown=unknown+1 WHERE period IN (?, ?)", (day, day[:7]))
            db.execute("COMMIT")
        finally:
            db.close()

    def prune(self) -> None:
        if not self.path.exists():
            return
        month = datetime.now(UTC).strftime("%Y-%m")
        db = self._connect()
        try:
            # Keep budget totals and unresolved requests when pruning old entries.
            db.execute("DELETE FROM spend WHERE id IN (SELECT id FROM spend "
                       "WHERE day < ? AND state='settled' LIMIT 2000)", (month,))
            from rune.agent.spend_owner import prune
            prune(self.path, {row[0] for row in db.execute("SELECT DISTINCT owner FROM spend_owners")})
        finally:
            db.close()


def reserve(params: dict, input_tokens: int) -> tuple[dict, tuple[SpendLedger, str] | None]:
    if not enabled():
        return params, None
    from rune.llm.pricing import reserve_request_cost

    params = dict(params)
    output_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
    output = params.get(output_key, 4096)
    cost = reserve_request_cost(params, input_tokens, output)
    if cost is None:
        raise BudgetExceeded("Pricing is unavailable; the shared model budget cannot authorize this request.")
    if "messages" in params:
        params.update({output_key: output, "num_retries": 0, "max_retries": 0})
    config = limits()
    ledger = SpendLedger()
    return params, (ledger, ledger.reserve(cost, config.daily_cost_usd, config.monthly_cost_usd))


def settle(reservation, row: dict, error: BaseException | None = None) -> None:
    if reservation is None:
        return
    ledger, request_id = reservation
    ledger.settle(request_id, row.get("costUsd") if error is None else None,
                  rejected=rejected_before_generation(error))
