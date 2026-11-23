"""Run the watch as a scheduled job: read the pages that are due, judge, and say what changed.

No agent loop, no tools and no model: a page is asked for, or opened in RUNE's own browser where
its shop needs one, and code reads it either way. A wake with nothing due makes no request, and a
run with nothing to tell sends nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

from rune.utils.logger import get_logger
from rune.watch import labels
from rune.watch.browse import Browser
from rune.watch.compose import News, compose
from rune.watch.fetch import Reader
from rune.watch.item import FETCH, Event, Item
from rune.watch.judge import due, settle
from rune.watch.observe import Observation, Pointed, observe
from rune.watch.store import Store

log = get_logger(__name__)

# What a watch job is marked with in the cron payload.
KIND = "watch"
# Shorter than the job's own timeout: a page not reached by then waits for the next wake.
DEADLINE = 300.0
# How long an ended item can still be picked up again by its id.
KEEP_ENDED = timedelta(days=90)
_QUIET = "Nothing to tell."


def zone() -> tzinfo:
    """The owner's time zone: the one schedules are read in, else the machine's."""
    from zoneinfo import ZoneInfo

    from rune.daemon.external_schedule import schedule_timezone

    name = schedule_timezone()
    return ZoneInfo(name) if name else (datetime.now().astimezone().tzinfo or UTC)


def clock() -> datetime:
    """Now, in the owner's time zone."""
    return datetime.now(zone())


async def run_watch(job: Any = None) -> dict[str, Any]:
    """One wake of the watch, as a scheduled run's result.

    It never raises. A run that could not be made says so with its date, so the hourly wakes of
    one bad day say it once, and the next wake runs as usual.
    """
    now = clock()
    try:
        news = await _look(now)
        if not news:
            return {"status": "completed", "success": True, "verified": True, "output": _QUIET,
                    "nothing_to_tell": True}
        t = await _labels()
        text, buttons = compose(news, now, t)
    except Exception as exc:
        # The owner is told only that it failed: an error's wording can repeat an address.
        log.warning("watch_run_failed", error=type(exc).__name__, exc_info=True)
        t = await _labels()
        return {"status": "failed", "success": False, "verified": False, "output": "", "title": t["title"],
                "error": t["failed"].format(date=f"{now:%Y-%m-%d}")}
    # Every figure was read from the pages by code; the words around them are fixed labels.
    return {"status": "completed", "success": True, "verified": True, "output": text, "title": t["title"],
            "buttons": [asdict(button) for button in buttons]}


async def read(reader: Reader, url: str, *, registering: bool, now: datetime, browser: Browser | None = None,
               via: str = FETCH, pointed_at: Pointed | None = None) -> Observation:
    """One reading for the watch. A read that got no answer on a hosted machine whose web is closed
    is said to be that, since nothing about the page was learned."""
    try:
        seen = await observe(reader, url, registering=registering, now=now, browser=browser, via=via,
                             pointed_at=pointed_at)
    except Exception as exc:
        log.warning("watch_read_failed", error=type(exc).__name__)
        return Observation(failure="error")
    if seen.failure == "unreachable":
        from rune.connectors import web_access

        if await web_access.closed() is not None:
            return replace(seen, failure="web_closed")
    return seen


async def _labels() -> dict[str, str]:
    try:
        from rune.briefing.labels import language

        return await labels.load(language())
    except Exception as exc:
        log.warning("watch_labels_unavailable", error=type(exc).__name__)
        return labels.ENGLISH


async def _look(now: datetime) -> list[News]:
    store = Store()
    try:
        store.forget_ended(now - KEEP_ENDED)
        at = store.check_time()
        wanted = [(item, kind) for item in store.watching() if (kind := due(item, now, at)) is not None]
        if not wanted:
            return []
        read_items: list[tuple[Item, list[Event], datetime]] = []
        stop = asyncio.get_running_loop().time() + DEADLINE
        # The browser is started only by the first page that has to be opened in one.
        async with Reader() as reader, Browser() as browser:
            for item, kind in wanted:
                try:
                    async with asyncio.timeout_at(stop):
                        seen = await read(reader, item.fetch_url, registering=False, now=now, browser=browser,
                                          via=item.via, pointed_at=Pointed(item.places, item.told.currency))
                except TimeoutError:
                    log.info("watch_out_of_time", read=len(read_items), due=len(wanted))
                    break
                try:
                    told_at = item.told_at
                    price = seen.reading.price if seen.reading is not None else None
                    store.observed(item.id, now, http_status=seen.http_status, price=price,
                                   offers=seen.reading.offers if seen.reading is not None else 0,
                                   source=seen.reading.source if seen.reading is not None else "",
                                   failure=seen.failure)
                    events = settle(item, now, kind, price if price is not None else seen.failure)
                except Exception as exc:
                    # This item is left as it was and read again at the next wake; the others go on.
                    log.warning("watch_item_failed", item=item.id, error=type(exc).__name__, exc_info=True)
                    continue
                read_items.append((item, events, told_at))
        # What was told changes here and nowhere else, in one write with everything else the run
        # learned: a run that stops before this has told nothing and reads again.
        saved = {item.id for item in store.save_all([item for item, _, _ in read_items])}
        for item, events, _ in read_items:
            if item.id not in saved:
                # The owner changed it while its page was being read: their change stands.
                log.info("watch_item_changed_meanwhile", item=item.id, dropped=len(events))
        return [News(item, event, told_at) for item, events, told_at in read_items if item.id in saved
                for event in events]
    finally:
        store.close()
