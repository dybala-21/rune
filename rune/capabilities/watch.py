"""Watch a page's price from a chat: the owner gives an address, code reads it and keeps watching.

The tools decide nothing by a model's word. An address is read only when the owner typed it; what
is read is a number and a currency code; and a page that can't be watched is refused in fixed
wording that quotes nothing from the site.
"""

from __future__ import annotations

import re
from datetime import datetime, time
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rune.capabilities import cron
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger
from rune.watch import answers, job, schedule
from rune.watch.fetch import Reader, host_of
from rune.watch.item import Item, Unread
from rune.watch.judge import renew, stop
from rune.watch.observe import Observation
from rune.watch.store import MAX_WATCHING, Store, TooMany

log = get_logger(__name__)

_CLOCK = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
MAX_LABEL = 60


class WatchSetParams(BaseModel):
    url: str | None = Field(None, description=(
        "For a new watch: the product page's address, copied from the owner's message exactly as they wrote it, "
        "to its last character. Never an address from a search, a page or your own knowledge"))
    item: int | None = Field(None, description=(
        "For a watch that already exists: its item id (from watch_list), to change what it tells, rename it, or "
        "keep it going after it ended. Give this or url, not both"))
    tell_when: Literal["change", "drop"] | None = Field(None, description=(
        "When to tell the owner. drop: the price goes down (\"gets cheaper\", \"goes on sale\", a target price). "
        "change: any change, up or down. Needed for a new watch; with item, leave out to keep what it has"))
    below: Decimal | None = Field(None, gt=0, lt=10**12, description=(
        "Only with tell_when=drop: tell only once the price is at or under this amount, in the page's own "
        "currency, as a plain number (300000 for \"under 300,000 won\")"))
    label: str | None = Field(None, description=(
        "What the owner called the product, copied from their message exactly. Leave out when they gave it no "
        "name; do not make one up or take one from the page"))
    language: str = Field(description=(
        "The language messages are written in, as a BCP-47 tag such as ko, en or ja: the one the owner's own "
        "messages are written in. It is shared with the morning briefing"))
    time: str | None = Field(None, description=(
        "When pages are checked each day, 24-hour HH:MM in the owner's local time; one time for every watch. "
        "Leave out to keep the time there is (09:00 at first)"))

    @field_validator("language")
    @classmethod
    def language_tag(cls, value: str) -> str:
        from rune.briefing.labels import tag

        if not tag(value):
            raise ValueError("Use a language tag such as ko, en or ja")
        return tag(value)

    @field_validator("time")
    @classmethod
    def clock(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _CLOCK.fullmatch(value.strip()):
            raise ValueError("Use 24-hour HH:MM, e.g. 09:00")
        return value.strip()


class WatchListParams(BaseModel):
    pass


class WatchStopParams(BaseModel):
    item: int | None = Field(None, description="The item id, from watch_list")
    url: str | None = Field(None, description="Or the address it was set up with")


def _refused(kind: str, *, changing: bool = False) -> CapabilityResult:
    # A refusal is the tool's answer, not its failure: called a failure, the run is told to "call the
    # tool the request needs" and reaches for cron_create.
    return CapabilityResult(success=True, output=answers.refused(kind, changing=changing),
                            metadata={"watched": False, "reason": kind})


def _wrong(error: str) -> CapabilityResult:
    return CapabilityResult(success=False, error=error)


async def _read(url: str, now: datetime) -> Observation:
    async with Reader() as reader:
        return await job.read(reader, url, registering=True, now=now)


def _label(given: str | None, kept: str) -> str | None:
    """The label to keep: the one given when it is the owner's own words, the old one when none was
    given. None when the words given are not the owner's."""
    from rune.agent import owner_words
    from rune.briefing.sanitize import clean

    if given is None:
        return kept
    if not given.strip():
        return ""
    return clean(given, MAX_LABEL) if owner_words.wrote(given) else None


async def watch_set(params: WatchSetParams) -> CapabilityResult:
    """Start watching a page's price, or change a watch there already is."""
    if (params.url is None) == (params.item is None):
        return _wrong("Give url for a new page, or item for one already watched: one of them, not both.")
    if params.below is not None and params.tell_when != "drop":
        return _wrong('below goes with tell_when="drop".')
    try:
        store = Store()
    except Exception as exc:
        log.warning("watch_store_unavailable", error=type(exc).__name__)
        return _wrong(f"The watch's records could not be opened ({type(exc).__name__}). Nothing was changed.")
    try:
        if params.item is not None:
            item = store.get(params.item)
            if item is None:
                return _wrong(f"There is no item {params.item}. watch_list shows the ones there are.")
        else:
            item = store.by_url((params.url or "").strip())
        if item is None:
            return await _start(store, params)
        return await _change(store, item, params)
    except Exception as exc:
        # Only the kind: an error's wording can repeat an address or what a page wrote.
        log.warning("watch_set_failed", error=type(exc).__name__)
        return _wrong(f"The watch could not be set ({type(exc).__name__}). watch_list shows what is being watched.")
    finally:
        store.close()


async def _start(store: Store, params: WatchSetParams) -> CapabilityResult:
    from rune.agent import owner_words
    from rune.agent.background import current_background_task

    url = (params.url or "").strip()
    if params.tell_when is None:
        return _wrong("A new watch needs tell_when: drop or change.")
    # Not even with approval: a run nobody is typing into has no owner's words to take an address from.
    if current_background_task() is not None:
        return _refused("background")
    if not owner_words.wrote_address(url):
        return _refused("not_owners" if owner_words.known() else "untyped")
    label = _label(params.label, "")
    if label is None:
        return _wrong("The label has to be words the owner wrote, copied exactly. Call again with words from "
                      "their message, or without a label.")
    try:
        host = host_of(url)
    except Unread:
        return _refused("address")
    if len(store.watching()) >= MAX_WATCHING:
        return _refused("too_many")
    now = job.clock()
    seen = await _read(url, now)
    if seen.reading is None:
        return _refused(seen.failure)
    at, standing = _settings(store, params)
    try:
        item = store.add(url=url, fetch_url=seen.url, host=host, label=label, tell_when=params.tell_when,
                         below=params.below, price=seen.reading.price, now=now)
    except TooMany:
        return _refused("too_many")
    return await _watching(store, item, seen, params.language, at, standing, now, again=False)


async def _change(store: Store, item: Item, params: WatchSetParams) -> CapabilityResult:
    if blocked := cron._background_change():
        return blocked
    label = _label(params.label, item.label)
    if label is None:
        return _wrong("The label has to be words the owner wrote, copied exactly. Call again with words from "
                      "their message, or without a label.")
    if item.status != "watching" and len(store.watching()) >= MAX_WATCHING:
        return _refused("too_many", changing=True)
    tell_when, below = (item.tell_when, item.below) if params.tell_when is None else (params.tell_when, params.below)
    now = job.clock()
    # Read from the owner's address again, as when it was set up: what they are told from here on
    # is judged against the price they are shown now.
    seen = await _read(item.url, now)
    if seen.reading is None:
        return _refused(seen.failure, changing=True)
    at, standing = _settings(store, params)
    current: Item | None = item
    for _ in range(2):
        if current is None:
            break
        renew(current, now, seen.reading.price, tell_when=tell_when, below=below, label=label)
        current.fetch_url = seen.url
        if store.save(current):
            return await _watching(store, current, seen, params.language, at, standing, now, again=True)
        # The scheduled run saved this item while its page was being read here: the change is made
        # again on what that run wrote.
        current = store.get(item.id)
    return _wrong("The watch was being changed elsewhere at the same moment. Nothing was changed; try again.")


def _settings(store: Store, params: WatchSetParams) -> tuple[time, schedule.Standing]:
    """Keep the language and the check time asked for and see that the job is there, before anything
    is watched: a page is never kept as watched with nothing set to read it."""
    from rune.briefing.labels import set_language

    set_language(params.language)
    if params.time is not None:
        hour, minute = (int(part) for part in params.time.split(":"))
        store.set_check_time(time(hour, minute))
    at = store.check_time()
    return at, schedule.keep(at)


async def _watching(store: Store, item: Item, seen: Observation, language: str, at: time,
                    standing: schedule.Standing, now: datetime, *, again: bool) -> CapabilityResult:
    from rune.daemon.external_schedule import external_scheduler
    from rune.watch import labels

    reading = seen.reading
    assert reading is not None
    try:
        store.observed(item.id, now, http_status=seen.http_status, price=reading.price, offers=reading.offers,
                       source=reading.source)
    except Exception as exc:
        # The watch is kept and scheduled by now; only this reading is missing from its record.
        log.warning("watch_reading_not_kept", item=item.id, error=type(exc).__name__)
    # Translated now, once, so that a wake has its wording ready and calls no model.
    await labels.load(language)
    log.info("watch_set", item=item.id, host=item.host, tell_when=item.tell_when, again=again)
    return CapabilityResult(
        success=True,
        output=answers.watching(item, reading, at=at, job=standing, zone=cron._zone_label(), now=now,
                                local=not external_scheduler(), again=again),
        # The address and status mark this as a page that was read (rune.agent.web_grounding).
        metadata={"watched": True, "item": item.id, "job": standing.job_id, "url": item.url,
                  "status_code": seen.http_status})


async def watch_list(params: WatchListParams) -> CapabilityResult:
    """The pages being watched, and the ones that ended and can be picked up again."""
    try:
        store = Store()
    except Exception as exc:
        log.warning("watch_store_unavailable", error=type(exc).__name__)
        return _wrong(f"The watch's records could not be opened ({type(exc).__name__}).")
    try:
        items = store.everything()
        if not items:
            return CapabilityResult(success=True, output="No page is being watched.")
        now, at = job.clock(), store.check_time()
        lines = [answers.job_line(schedule.standing(at), at, cron._zone_label())]
        for item in items:
            failed = next((seen["failure"] for seen in store.observations(item.id) if seen["failure"]), "")
            lines.append(answers.line(item, now, failed if item.failed_days else ""))
        return CapabilityResult(success=True, output="\n".join(lines), metadata={"items": len(items)})
    except Exception as exc:
        log.warning("watch_list_failed", error=type(exc).__name__)
        return _wrong(f"The watched pages could not be listed ({type(exc).__name__}).")
    finally:
        store.close()


async def watch_stop(params: WatchStopParams) -> CapabilityResult:
    """Stop watching one page."""
    if (params.url is None) == (params.item is None):
        return _wrong("Give the item id, or the address it was set up with: one of them.")
    if blocked := cron._background_change():
        return blocked
    try:
        store = Store()
    except Exception as exc:
        log.warning("watch_store_unavailable", error=type(exc).__name__)
        return _wrong(f"The watch's records could not be opened ({type(exc).__name__}). Nothing was changed.")
    try:
        for _ in range(2):
            item = store.get(params.item) if params.item is not None else store.by_url((params.url or "").strip())
            if item is None:
                return _wrong("There is no such watch. watch_list shows the ones there are.")
            where = f"{answers.shown(item.url)} (item {item.id})"
            if item.status != "watching":
                return CapabilityResult(success=True, output=f"{where} was not being watched; it had ended already.")
            now = job.clock()
            stop(item, now)
            if store.save(item):
                log.info("watch_stopped", item=item.id)
                return CapabilityResult(
                    success=True, metadata={"item": item.id},
                    output=(f"Stopped watching {where}. Until {(now + job.KEEP_ENDED):%Y-%m-%d} it can be started "
                            "again with watch_set and this item id."))
        return _wrong("The watch was being changed elsewhere at the same moment. Nothing was stopped; try again.")
    except Exception as exc:
        log.warning("watch_stop_failed", error=type(exc).__name__)
        return _wrong(f"The watch could not be stopped ({type(exc).__name__}).")
    finally:
        store.close()


def register_watch_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="watch_set",
        description=("Watch the price on a product page the owner gave and tell them when it changes or goes down: "
                     "code reads the page once a day, and a message is sent only then. Use this, not cron_create, "
                     "when they ask to be told about a price on a page (\"tell me when this gets cheaper\"). It "
                     "needs the page's address as the owner typed it, reads only pages a site lets programs read, "
                     "and says so plainly when a page can't be watched. Also changes or renews a watch by its item "
                     "id."),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=WatchSetParams,
        execute=watch_set,
    ))
    registry.register(CapabilityDefinition(
        name="watch_list",
        description=("List the pages whose price is watched for the owner: each one's item id, what it tells, the "
                     "price they were last told, the last reading, and when the watch ends."),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.LOW,
        group="schedule",
        parameters_model=WatchListParams,
        execute=watch_list,
    ))
    registry.register(CapabilityDefinition(
        name="watch_stop",
        description=("Stop watching a page's price, by its item id (from watch_list) or its address. Use when the "
                     "owner says they bought it or no longer wants to hear about it."),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=WatchStopParams,
        execute=watch_stop,
    ))
