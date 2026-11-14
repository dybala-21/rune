"""Where each remembered thing came from, so what a connector returned can be shown and forgotten.

A run records the connectors it called. The episode saved for that run keeps them, and facts drawn
from the episode (or saved during the run) keep them in fact-meta.json as their origin. Nothing
recorded before this existed has an origin; it is shown as unknown, never as a connector's.

On a hosted machine, a connector call closes the open web to it (the broker marks it private), so an
instruction hidden in a page can't send the owner's data away. A remembered connector answer put
into a later run carries the same data, so it closes the web the same way first. If the broker
can't be told, the remembered answer stays out of the run.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_CONTACTED: ContextVar[set[str] | None] = ContextVar("rune_run_connectors", default=None)


@contextlib.contextmanager
def run_sources() -> Iterator[set[str]]:
    """Collect the connectors called inside. Nested scopes share the outer one's set."""
    current = _CONTACTED.get()
    if current is not None:
        yield current
        return
    contacted: set[str] = set()
    token = _CONTACTED.set(contacted)
    try:
        yield contacted
    finally:
        _CONTACTED.reset(token)


def record_connector(name: str) -> None:
    """Called before a connector request goes out, whatever it returns."""
    contacted = _CONTACTED.get()
    if contacted is None:
        log.warning("connector_called_outside_run", connector=name)
        return
    contacted.add(name)


def contacted() -> list[str]:
    return sorted(_CONTACTED.get() or ())


def encode(connectors: Iterable[str]) -> str:
    names = sorted(set(connectors))
    return json.dumps(names) if names else ""


def decode(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        names = json.loads(value)
    except ValueError:
        # Unreadable is treated as a connector's: forgetting more is the safe side.
        log.warning("episode_sources_unreadable", value=value[:80])
        return ["unknown"]
    return [n for n in names if isinstance(n, str)] if isinstance(names, list) else ["unknown"]


def merge_origin(meta: dict[str, Any], *, connectors: Iterable[str] = (), conversation: str = "",
                 episode: str = "", kind: str = "") -> dict[str, Any]:
    """The origin a fact has after it is saved again: every place it came from, never fewer."""
    origin = dict(meta.get("origin") or {})
    for field, value in (("connectors", connectors), ("conversations", [conversation] if conversation else []),
                         ("episodes", [episode] if episode else [])):
        merged = sorted(set(origin.get(field) or []) | set(value))
        if merged:
            origin[field] = merged
    if kind and not origin.get("kind"):
        origin["kind"] = kind
    return origin


def note_facts(keys: Iterable[str], *, connectors: Iterable[str] = (), conversation: str = "",
               episode: str = "", kind: str = "") -> None:
    """Add where these facts came from to their origin, in one write."""
    keys, connectors = list(keys), list(connectors)
    if not keys:
        return
    from rune.memory.state import load_fact_meta, save_fact_meta

    meta = load_fact_meta()
    for key in keys:
        entry = dict(meta.get(key) or {})
        entry["origin"] = merge_origin(entry, connectors=connectors, conversation=conversation,
                                       episode=episode, kind=kind)
        meta[key] = entry
    save_fact_meta(meta)


def fact_connectors(meta: dict[str, Any], key: str) -> list[str]:
    return list(((meta.get(key) or {}).get("origin") or {}).get("connectors") or [])


def private_keys(meta: dict[str, Any] | None = None) -> set[str]:
    """Keys of remembered facts that came from a connector, lower-cased as lookups compare them."""
    if meta is None:
        from rune.memory.state import load_fact_meta

        meta = load_fact_meta()
    return {key.lower() for key in meta if fact_connectors(meta, key)}


async def allow_private(what: str) -> bool:
    """May something a connector returned go into this run? On a hosted machine only once the
    broker has closed the web to it; anywhere else, always."""
    from rune.connectors.client import call_broker, hosted_approval

    if not hosted_approval():
        return True
    try:
        answer = await call_broker("private", {})
    except Exception as exc:
        log.warning("private_memory_withheld", what=what, error=type(exc).__name__)
        return False
    if answer.get("private") is not True:
        log.warning("private_memory_withheld", what=what, error="broker did not record it")
        return False
    from rune.connectors import web_access

    web_access.forget()
    return True


class PrivateFilter:
    """Decides, item by item, whether remembered things that came from a connector go into a run.

    With `allow`, they go in and `used` says one did, so the caller closes the web before the run
    sees them. Without it, they are left out.
    """

    def __init__(self, allow: bool) -> None:
        self.allow = allow
        self.used = False
        # Facts that went into the context, counted as used by the caller.
        self.hits: list[str] = []
        self._keys: set[str] | None = None

    def _keep(self, private: bool) -> bool:
        if not private:
            return True
        self.used = self.used or self.allow
        return self.allow

    def episode(self, episode: Any) -> bool:
        return self._keep(bool(decode(getattr(episode, "sources", ""))))

    def fact(self, key: str) -> bool:
        if self._keys is None:
            self._keys = private_keys()
        return self._keep(key.lower() in self._keys)

    def daily(self, daily: Any) -> bool:
        return self._keep(bool(getattr(daily, "connectors", None)))


class PrivateIndex:
    """What came from a connector, looked up once per search."""

    def __init__(self) -> None:
        from rune.memory.store import get_memory_store

        self.keys = private_keys()
        episodes = get_memory_store().episodes_with_sources()
        self.episodes = {ep.id for ep in episodes}
        self.conversations = {ep.conversation_id for ep in episodes if ep.conversation_id}
        self._daily: dict[str, set[str]] = {}

    def chunk(self, kind: str, item_id: str) -> bool:
        """Whether an indexed piece of memory holds what a connector returned."""
        if kind == "episode":
            return item_id in self.episodes
        if kind == "md_fact":
            return item_id.rsplit("::", 1)[-1].strip().lower() in self.keys
        if kind == "md_daily":
            name, _, at = item_id.partition("::")
            return at in self._daily_times(name)
        return False

    def _daily_times(self, name: str) -> set[str]:
        if name not in self._daily:
            from rune.memory.markdown_store import memory_dir, parse_daily_log

            self._daily[name] = {e["time"] for e in parse_daily_log(memory_dir() / name) if e.get("connectors")}
        return self._daily[name]
