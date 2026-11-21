"""One reading of a watched page: fetched under the watch's rules, then read for its price."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from rune.utils.logger import get_logger
from rune.watch.extract import Reading, reading
from rune.watch.fetch import Reader
from rune.watch.item import Unread

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Observation:
    """What a read gave: a reading, or the kind of failure. Never both, never a site's own words."""

    reading: Reading | None = None
    failure: str = ""
    url: str = ""  # where it was read from; kept to read from again, never shown
    http_status: int = 0


async def observe(reader: Reader, url: str, *, registering: bool, now: datetime) -> Observation:
    try:
        page = await reader.page(url, registering=registering)
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
