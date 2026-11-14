"""What RUNE remembers, fact by fact, with where each came from (`memory.*` RPC)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rune.memory import markdown_store as md
from rune.memory.state import load_fact_meta, load_suppressed, suppress_fact, unsuppress_fact

Zone = Literal["learned", "memory", "profile"]
_LINE = Field(min_length=1, max_length=2000)


class SaveRequest(BaseModel):
    zone: Zone
    value: str = _LINE
    key: str = ""  # learned.md: the fact to change
    section: str = Field(default="", max_length=200)  # MEMORY.md / user-profile.md
    old: str | None = None  # the line to replace there; None adds a line


class ForgetRequest(BaseModel):
    zone: Zone
    key: str = ""
    section: str = ""
    line: str = ""


def _origin(meta: dict[str, Any], key: str) -> dict[str, Any]:
    origin = (meta.get(key) or {}).get("origin") or {}
    return {"kind": origin.get("kind", ""), "connectors": origin.get("connectors", []),
            "conversations": origin.get("conversations", [])}


def _sections(path: Path | None, meta: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for section, lines in md.parse_memory_md(path).items():
        for line in lines:
            key = line.split(":", 1)[0].strip() if ":" in line else ""
            out.append({"section": section, "line": line, "key": key,
                        "origin": _origin(meta, key) if key else None})
    return out


def snapshot() -> dict[str, Any]:
    from rune.memory.forget import connectors_remembered

    meta = load_fact_meta()
    learned = [{"key": f["key"], "category": f["category"], "value": f["value"],
                "confidence": f["confidence"], "hits": (meta.get(f["key"]) or {}).get("hit_count", 0),
                "origin": _origin(meta, f["key"])} for f in md.parse_learned_md()]
    suppressed = [{"key": key, "value": item.get("original_value", ""), "reason": item.get("reason", ""),
                   "at": item.get("suppressed_at", "")} for key, item in load_suppressed().items()]
    suppressed.sort(key=lambda s: s["at"], reverse=True)
    return {
        "learned": learned,
        "memory": _sections(None, meta),
        "profile": _sections(md.memory_dir() / "user-profile.md", meta),
        "suppressed": suppressed,
        "connectors": connectors_remembered(),
    }


def _path(zone: Zone) -> Path:
    return md.memory_dir() / ("MEMORY.md" if zone == "memory" else "user-profile.md")


def _save(req: SaveRequest) -> None:
    value = " ".join(req.value.split())  # one bullet line, whatever was pasted
    if req.zone == "learned":
        fact = next((f for f in md.parse_learned_md() if f["key"] == req.key), None)
        if fact is None:
            raise LookupError(f"No learned fact {req.key!r}")
        md.save_learned_fact(fact["category"], fact["key"], value, fact["confidence"])
        return
    if not req.section.strip():
        raise ValueError("A section is required")
    if req.old is None:
        if req.zone == "memory":
            md.append_to_memory_md(req.section.strip(), value)
        else:
            lines = md.parse_user_profile().get(req.section.strip(), [])
            md.update_user_profile_section(req.section.strip(), [*lines, value])
    elif not md.replace_section_line(req.section, req.old, value, _path(req.zone)):
        raise LookupError("That line has changed; reload and try again")


def _forget(req: ForgetRequest) -> None:
    if req.zone == "learned":
        value = md.learned_md_has_key(req.key)
        if value is None or not md.remove_learned_fact(req.key):
            raise LookupError(f"No learned fact {req.key!r}")
        # Otherwise the next run would draw it out again.
        suppress_fact(req.key, value, reason="user_deleted")
    elif not md.replace_section_line(req.section, req.line, None, _path(req.zone)):
        raise LookupError("That line has changed; reload and try again")


def _promote(key: str) -> None:
    fact = next((f for f in md.parse_learned_md() if f["key"].lower() == key.lower()), None)
    if fact is None:
        raise LookupError(f"No learned fact {key!r}")
    section = {"preference": "Preferences", "environment": "Environment"}.get(fact["category"], "Notes")
    md.append_to_memory_md(section, f"{fact['key']}: {fact['value']}")
    md.remove_learned_fact(fact["key"])  # the origin stays in fact-meta under the same key


async def _changed(fn: Any, *args: Any) -> dict[str, Any]:
    from rune.memory.manager import get_memory_manager

    await asyncio.to_thread(fn, *args)
    get_memory_manager().reload_facts()
    return await asyncio.to_thread(snapshot)


async def handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Raises ValueError for a bad request and LookupError when the thing named is gone."""
    if method == "memory.list":
        return await asyncio.to_thread(snapshot)
    if method == "memory.save":
        return await _changed(_save, SaveRequest(**params))
    if method == "memory.forget":
        return await _changed(_forget, ForgetRequest(**params))
    if method == "memory.promote":
        return await _changed(_promote, str(params.get("key", "")))
    if method == "memory.unsuppress":
        return await _changed(unsuppress_fact, str(params.get("key", "")))
    if method == "memory.forget_connector":
        from rune.memory.forget import forget_connector

        name = str(params.get("connector", "")).strip()
        if not name:
            raise ValueError("A connector name is required")
        forgotten = await forget_connector(name)
        return {"forgotten": forgotten.payload(), **await asyncio.to_thread(snapshot)}
    raise ValueError(f"Unknown method: {method}")
