"""Forget everything RUNE remembers from one connector.

Facts drawn from a run that called the connector are deleted and kept from coming back. The runs'
episodes, their daily-log entries and their search-index entries are deleted too. Conversation
transcripts are not: they are the owner's to delete, and the screen says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.memory import markdown_store as md
from rune.memory.provenance import decode, fact_connectors
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Forgotten:
    facts: int = 0
    episodes: int = 0
    daily_entries: int = 0

    def payload(self) -> dict[str, int]:
        return {"facts": self.facts, "episodes": self.episodes, "dailyEntries": self.daily_entries}


def connectors_remembered() -> list[dict[str, Any]]:
    """Each connector something is remembered from, with how much."""
    from rune.memory.state import load_fact_meta
    from rune.memory.store import get_memory_store

    counts: dict[str, dict[str, Any]] = {}

    def entry(name: str) -> dict[str, Any]:
        return counts.setdefault(name, {"name": name, "facts": 0, "episodes": 0})

    meta = load_fact_meta()
    present = {f["key"].lower() for f in md.parse_learned_md()} | _memory_md_keys()
    for key in meta:
        if key.lower() in present:
            for name in fact_connectors(meta, key):
                entry(name)["facts"] += 1
    for episode in get_memory_store().episodes_with_sources():
        for name in decode(episode.sources):
            entry(name)["episodes"] += 1
    return sorted(counts.values(), key=lambda c: c["name"])


def _memory_md_keys() -> set[str]:
    return {line.split(":", 1)[0].strip().lower()
            for lines in md.parse_memory_md().values() for line in lines if ":" in line}


async def forget_connector(name: str) -> Forgotten:
    from rune.memory.manager import get_memory_manager
    from rune.memory.state import load_fact_meta, save_fact_meta, suppress_fact

    done = Forgotten()
    meta = load_fact_meta()
    keys = {key.lower() for key in meta if name in fact_connectors(meta, key)}

    for fact in md.parse_learned_md():
        if fact["key"].lower() in keys and md.remove_learned_fact(fact["key"]):
            # Kept from being drawn out of the same data again.
            suppress_fact(fact["key"], fact["value"], reason=f"connector_forgotten:{name}")
            done.facts += 1
    for section, lines in md.parse_memory_md().items():
        for line in lines:
            if ":" in line and line.split(":", 1)[0].strip().lower() in keys:
                done.facts += md.replace_section_line(section, line, None)
    for key in [k for k in meta if k.lower() in keys]:
        del meta[key]
    save_fact_meta(meta)

    manager = get_memory_manager()
    episodes = [ep for ep in manager.store.episodes_with_sources() if name in decode(ep.sources)]
    ids = [ep.id for ep in episodes]
    done.episodes = manager.store.delete_episodes(ids) and len(ids)
    for episode_id in ids:
        manager.keywords.remove(episode_id)
        manager.vectors.delete_by_id(episode_id)
    if ids:
        manager.persist_vectors()

    done.daily_entries = md.remove_daily_entries(name)
    try:
        from rune.memory.markdown_indexer import incremental_reindex

        await incremental_reindex(manager.vectors)
    except Exception as exc:
        # The markdown is already gone; a later reindex drops its stale index entries.
        log.warning("forget_connector_reindex_failed", error=type(exc).__name__)
    manager.reload_facts()
    log.info("connector_memory_forgotten", connector=name, **done.payload())
    return done
