"""One reading of a watched page: the page is had one of two ways, then read for its price.

A page is asked for under RUNE's name first. Where the site refuses that, or the page it gives
has no price in it, the page is opened in RUNE's own browser instead; a page that was read that
way before is opened that way at once. A site that names RUNE in its robots.txt and refuses it is
not opened in the browser either: it has said what it wants of RUNE, not of programs in general.

The price is the one the page writes in its structured data. A page that writes none there is read
from the places that hold the amount its owner said it shows (``rune.watch.places``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from rune.utils.logger import get_logger
from rune.watch import places as pointed
from rune.watch.browse import Browser
from rune.watch.extract import Reading, reading
from rune.watch.fetch import Page, Reader
from rune.watch.item import BROWSER, FETCH, Price, Unread

log = get_logger(__name__)

# What asking under RUNE's name can end in that opening the page in a browser may get past: the
# site lets no unnamed program in, it turned the request away, or its price (or the amount its
# owner said it shows) is put in by a script.
OPENED_AFTER = frozenset({"robots", "blocked", "no_price", "not_shown"})


@dataclass(frozen=True, slots=True)
class Observation:
    """What a read gave: a reading, or the kind of failure. Never both, never a site's own words."""

    reading: Reading | None = None
    failure: str = ""
    url: str = ""  # where it was read from; kept to read from again, never shown
    http_status: int = 0
    via: str = FETCH  # the way that gave this: FETCH or BROWSER
    # For a price read from where it is written in the page: the places it was read from.
    places: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Pointed:
    """How a page that writes no structured price is read: from ``places`` kept for it, in
    ``currency``. ``said`` is the amount its owner has just said it shows, when they have."""

    places: tuple[str, ...] = ()
    currency: str = ""
    said: Decimal | None = None


async def observe(reader: Reader, url: str, *, registering: bool, now: datetime, browser: Browser | None = None,
                  via: str = FETCH, pointed_at: Pointed | None = None) -> Observation:
    """Read ``url`` once. ``via`` is the way chosen when its watch was set; with no ``browser`` it is only asked for."""
    how = pointed_at or Pointed()
    asked: Observation | None = None
    if via != BROWSER or browser is None:
        asked = await _through(reader, url, registering, now, how)
        if browser is None or asked.failure not in OPENED_AFTER:
            return asked
    if await reader.refuses_by_name(url):
        return asked or Observation(failure="robots")
    opened = await _through(browser, url, registering, now, how)
    if opened.failure == "no_browser":
        # No browser on this machine: what asking gave stands, and a page not asked for yet is asked for now.
        return asked or await _through(reader, url, registering, now, how)
    return replace(opened, via=BROWSER)


def _price(page: Page, now: datetime, how: Pointed) -> tuple[Reading, tuple[str, ...]]:
    """The page's price and, when it was read from where it is written, the places it was read from."""
    if how.places and how.said is None:
        amount = pointed.read(page.text, how.places)
        return Reading(Price(amount, amount, how.currency), 1, "pointed"), how.places
    if not how.places:
        # A page that writes its price as structured data is read from that, whatever was said.
        try:
            return reading(page.text, now), ()
        except Unread as none:
            if none.kind != "no_price" or how.said is None:
                raise
    assert how.said is not None
    held = pointed.holding(page.text, how.said, among=how.places)
    if not held:
        raise Unread("not_shown")
    return Reading(Price(how.said, how.said, how.currency), 1, "pointed"), held


async def _through(way: Reader | Browser, url: str, registering: bool, now: datetime, how: Pointed) -> Observation:
    try:
        page = await way.page(url, registering=registering)
    except Unread as refused:
        return Observation(failure=refused.kind, http_status=refused.status)
    try:
        # Parsing a few megabytes of a site's markup is kept off the event loop.
        found, held = await asyncio.to_thread(_price, page, now, how)
    except Unread as refused:
        return Observation(failure=refused.kind, url=page.url, http_status=page.status)
    except Exception as failed:
        # A page this code did not foresee is a failed read, not a failed run. Only the error's
        # kind is logged: its wording may repeat what the page wrote.
        log.warning("watch_reading_failed", error=type(failed).__name__)
        return Observation(failure="error", url=page.url, http_status=page.status)
    return Observation(reading=found, url=page.url, http_status=page.status, places=held)
