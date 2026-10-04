"""Durable suggestion state shared by the API and daemon."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rune.proactive.types import Suggestion

if TYPE_CHECKING:
    from rune.memory.store import MemoryStore


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def restore(row: dict) -> Suggestion:
    meta = row["metadata"]
    return Suggestion(
        id=meta.get("suggestion_id", str(row["id"])),
        type=meta.get("type", "insight"),
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        confidence=float(meta.get("confidence", 0.5)),
        source=meta.get("source", "persisted"),
        status=row["state"],
        response_source="user" if meta.get("response_source") == "user" else None,
        created_at=_date(meta.get("created_at")) or datetime.now(UTC),
        expires_at=_date(meta.get("expires_at")),
        verification=list(meta.get("verification", [])),
        execution_status=meta.get("execution_status"),
        execution_result=meta.get("execution_result", {}),
    )


def _existing(store: MemoryStore, suggestion_id: str) -> dict | None:
    row = store.conn.execute(
        """SELECT id, state, metadata FROM proactive_suggestions_state
           WHERE json_extract(metadata, '$.suggestion_id') = ?
           ORDER BY updated_at DESC, id DESC LIMIT 1""",
        (suggestion_id,),
    ).fetchone()
    return {"id": row[0], "state": row[1], "metadata": json.loads(row[2])} if row else None


def save(store: MemoryStore, suggestion: Suggestion) -> None:
    meta = asdict(suggestion)
    meta["suggestion_id"] = meta.pop("id")
    state = meta.pop("status")
    meta["created_at"] = suggestion.created_at.isoformat()
    meta["expires_at"] = suggestion.expires_at.isoformat() if suggestion.expires_at else None
    with store.conn:
        existing = _existing(store, suggestion.id)
        if existing:
            previous = existing["metadata"]
            for key in ("title", "description", "verification", "source", "type", "expires_at"):
                if key not in previous:
                    continue
                if previous.get(key, [] if key == "verification" else "") != meta[key]:
                    raise ValueError("Changed suggestions require a new ID and new approval")
            # A stale daemon snapshot must not undo a response from the API.
            if existing["state"] == "expired" or previous.get("response_source") == "user":
                state = existing["state"]
            if previous.get("response_source") == "user":
                meta["response_source"] = "user"
            if meta["execution_status"] is None:
                meta["execution_status"] = previous.get("execution_status")
                meta["execution_result"] = previous.get("execution_result", {})
            store.conn.execute(
                """UPDATE proactive_suggestions_state SET state = ?, metadata = ?, updated_at = ?
                   WHERE id = ?""",
                (state, json.dumps(meta), datetime.now(UTC).isoformat(), existing["id"]),
            )
            store.conn.execute(
                """DELETE FROM proactive_suggestions_state
                   WHERE json_extract(metadata, '$.suggestion_id') = ? AND id != ?""",
                (suggestion.id, existing["id"]),
            )
        else:
            store.save_suggestion_state(suggestion.type, state, meta)


def expire(store: MemoryStore, suggestion_id: str) -> None:
    with store.conn:
        row = _existing(store, suggestion_id)
        if row is not None:
            store.conn.execute("UPDATE proactive_suggestions_state SET state = 'expired' WHERE id = ?",
                               (row["id"],))


def respond(store: MemoryStore, suggestion_id: str, accepted: bool) -> Suggestion | None:
    desired = "accepted" if accepted else "dismissed"
    with store.conn:
        row = _existing(store, suggestion_id)
        if row is None:
            return None
        suggestion = restore(row)
        if suggestion.response_source is None and suggestion.status in ("accepted", "dismissed"):
            suggestion.status = "pending"
        if suggestion.status == desired and suggestion.response_source == "user":
            return suggestion
        if suggestion.status not in ("pending", desired):
            raise ValueError("This suggestion already has a different response")
        if suggestion.expires_at and suggestion.expires_at <= datetime.now(UTC):
            raise ValueError("This suggestion has expired")
        suggestion.status = desired
        suggestion.response_source = "user"
        save(store, suggestion)
        return suggestion
