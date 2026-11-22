"""One reading of a watched page: the page is had one of two ways, then read for its price.

A page is asked for under RUNE's name first. Where the site refuses that, or the page it gives
has no price in it, the page is opened in RUNE's own browser instead; a page that was read that
way before is opened that way at once. A site that names RUNE in its robots.txt and refuses it is
not opened in the browser either: it has said what it wants of RUNE, not of programs in general.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime

from rune.utils.logger import get_logger
from rune.watch.browse import Browser
from rune.watch.extract import Reading, reading
from rune.watch.fetch import Reader
from rune.watch.item import BROWSER, FETCH, Unread

log = get_logger(__name__)

# What asking under RUNE's name can end in that opening the page in a browser may get past: the
# site lets no unnamed program in, it turned the request away, or its price is put in by a script.
OPENED_AFTER = frozenset({"robots", "blocked", "no_price"})


@dataclass(frozen=True, slots=True)
class Observation:
    """What a read gave: a reading, or the kind of failure. Never both, never a site's own words."""

    reading: Reading | None = None
    failure: str = ""
    url: str = ""  # where it was read from; kept to read from again, never shown
    http_status: int = 0
    via: str = FETCH  # the way that gave this: FETCH or BROWSER


async def observe(reader: Reader, url: str, *, registering: bool, now: datetime,
                  browser: Browser | None = None, via: str = FETCH) -> Observation:
    """Read ``url`` once. ``via`` is the way chosen when its watch was set; with no ``browser`` it is only asked for."""
    asked: Observation | None = None
    if via != BROWSER or browser is None:
        asked = await _through(reader, url, registering, now)
        if browser is None or asked.failure not in OPENED_AFTER:
            return asked
    if await reader.refuses_by_name(url):
        return asked or Observation(failure="robots")
    opened = await _through(browser, url, registering, now)
    if opened.failure == "no_browser":
        # No browser on this machine: what asking gave stands, and a page not asked for yet is asked for now.
        return asked or await _through(reader, url, registering, now)
    return replace(opened, via=BROWSER)


async def _through(way: Reader | Browser, url: str, registering: bool, now: datetime) -> Observation:
    try:
        page = await way.page(url, registering=registering)
    except Unread as refused:
        return Observation(failure=refused.kind, http_status=refused.status)
    try:
        # Parsing a few megabytes of a site's markup is kept off the event loop.
        found = await asyncio.to_thread(reading, page.text, now)
    except Unread as refused:
        return Observation(failure=refused.kind, url=page.url, http_status=page.status)
    except Exception as failed:
        # A page this code did not foresee is a failed read, not a failed run. Only the error's
        # kind is logged: its wording may repeat what the page wrote.
        log.warning("watch_reading_failed", error=type(failed).__name__)
        return Observation(failure="error", url=page.url, http_status=page.status)
    return Observation(reading=found, url=page.url, http_status=page.status)
