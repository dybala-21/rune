"""Zone 3 state file I/O for RUNE memory.

Manages the .state/ directory containing derived metadata:
- fact-meta.json: confidence, source, hit_count per fact
- suppressed.json: user-deleted facts (prevent re-extraction)
- conflicts.json: fact conflict history
- index-state.json: content hashes for incremental reindex
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


def _state_dir() -> Path:
    d = rune_home() / "memory" / ".state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.suffix == ".json" else []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("state_read_failed", path=str(path), error=str(exc))
        return {} if "conflicts" not in path.name else []


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        log.error("state_write_failed", path=str(path), error=str(exc))
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# fact-meta.json

def load_fact_meta() -> dict[str, Any]:
    return _read_json(_state_dir() / "fact-meta.json")


def save_fact_meta(data: dict[str, Any]) -> None:
    _write_json(_state_dir() / "fact-meta.json", data)


def update_fact_meta(key: str, updates: dict[str, Any]) -> None:
    meta = load_fact_meta()
    entry = meta.get(key, {})
    entry.update(updates)
    meta[key] = entry
    save_fact_meta(meta)


def increment_hit_count(key: str) -> None:
    meta = load_fact_meta()
    entry = meta.get(key, {})
    entry["hit_count"] = entry.get("hit_count", 0) + 1
    entry["last_hit"] = datetime.now(UTC).isoformat()
    meta[key] = entry
    save_fact_meta(meta)


# suppressed.json

def load_suppressed() -> dict[str, Any]:
    return _read_json(_state_dir() / "suppressed.json")


def save_suppressed(data: dict[str, Any]) -> None:
    _write_json(_state_dir() / "suppressed.json", data)


def is_suppressed(key: str) -> bool:
    return key in load_suppressed()


def suppress_fact(key: str, value: str, reason: str = "user_deleted") -> None:
    data = load_suppressed()
    data[key] = {
        "suppressed_at": datetime.now(UTC).isoformat(),
        "original_value": value,
        "reason": reason,
    }
    save_suppressed(data)


def unsuppress_fact(key: str) -> bool:
    data = load_suppressed()
    if key in data:
        del data[key]
        save_suppressed(data)
        return True
    return False


# conflicts.json

def load_conflicts() -> list[dict[str, Any]]:
    data = _read_json(_state_dir() / "conflicts.json")
    return data if isinstance(data, list) else []


def save_conflicts(data: list[dict[str, Any]]) -> None:
    _write_json(_state_dir() / "conflicts.json", data)


def record_conflict(
    key: str,
    old_value: str,
    new_value: str,
    old_source: str = "",
    new_source: str = "",
) -> None:
    conflicts = load_conflicts()
    conflicts.append({
        "key": key,
        "old_value": old_value,
        "new_value": new_value,
        "old_source": old_source,
        "new_source": new_source,
        "resolved_at": datetime.now(UTC).isoformat(),
        "resolution": "update",
    })
    # Keep last 100 conflicts
    if len(conflicts) > 100:
        conflicts = conflicts[-100:]
    save_conflicts(conflicts)


# index-state.json

def load_index_state() -> dict[str, Any]:
    return _read_json(_state_dir() / "index-state.json")


def save_index_state(data: dict[str, Any]) -> None:
    _write_json(_state_dir() / "index-state.json", data)
