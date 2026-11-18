"""Run the briefing as a scheduled job: read, rank, compose. No agent loop and no tools."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime, tzinfo
from typing import Any

from rune.briefing import news, rank, sources
from rune.briefing.compose import compose
from rune.utils.logger import get_logger

log = get_logger(__name__)

# What a briefing job is marked with in the cron payload.
KIND = "briefing"


def _zone() -> tzinfo:
    from zoneinfo import ZoneInfo

    from rune.daemon.external_schedule import schedule_timezone

    name = schedule_timezone()
    return ZoneInfo(name) if name else (datetime.now().astimezone().tzinfo or UTC)


async def run_briefing(job: Any = None) -> dict[str, Any]:
    """The briefing's text and buttons, as a scheduled run's result."""
    zone = _zone()
    day = datetime.now(zone)
    headlines = news.load()
    try:
        connectors = await sources.listed()
    except Exception as exc:
        log.warning("briefing_connectors_unavailable", error=type(exc).__name__)
        return {"status": "failed", "success": False, "verified": False, "output": "",
                "error": "Connected accounts could not be reached, so the briefing wasn't made."}
    calendar, mail = await asyncio.gather(sources.calendar(connectors, zone, day), sources.mail(connectors))
    ranked = await rank.rank(mail.items, headlines)
    text, buttons = compose(day, calendar, mail, ranked)
    log.info("briefing_composed", events=len(calendar.items), mail=len(mail.items), news=len(ranked.news),
             ranked=ranked.ranked, calendar_missing=calendar.missing, mail_missing=mail.missing)
    # Every line is read from the accounts by code; only the order and gists are the model's.
    return {"status": "completed", "success": True, "verified": True, "output": text,
            "buttons": [asdict(b) for b in buttons]}
