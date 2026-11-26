"""Run the briefing as a scheduled job: read, rank, compose. No agent loop and no tools."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime, tzinfo
from typing import Any

from rune.briefing import kept, labels, news, rank, sources
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


def _told() -> list[str]:
    """The accounts the last briefing sent said weren't connected."""
    missing = kept.read("told").get("missing")
    return [name for name in missing if isinstance(name, str)] if isinstance(missing, list) else []


def _remember_told(missing: list[str]) -> None:
    if missing == _told():
        return
    try:
        kept.write("told", {"missing": missing})
    except OSError as exc:
        # Not kept, so a later empty day says it again instead of staying quiet.
        log.warning("briefing_told_not_kept", error=type(exc).__name__)


async def run_briefing(job: Any = None) -> dict[str, Any]:
    """The briefing's text and buttons, as a scheduled run's result.

    A day with nothing to report is not sent: every connected account was read and had nothing, and
    an account that isn't connected was already said to be. A read that failed is always said, so a
    quiet morning means RUNE looked and found nothing, never that it couldn't look.
    """
    zone = _zone()
    day = datetime.now(zone)
    language = labels.language()
    t = await labels.load(language)
    headlines = news.load()
    try:
        connectors = await sources.listed()
    except Exception as exc:
        log.warning("briefing_connectors_unavailable", error=type(exc).__name__)
        return {"status": "failed", "success": False, "verified": False, "output": "", "title": t["title"],
                "error": t["accounts_failed"]}
    calendar, mail = await asyncio.gather(sources.calendar(connectors, zone, day), sources.mail(connectors))
    ranked = await rank.rank(mail.items, headlines, language)
    missing = sorted(name for name, fetched in ((sources.CALENDAR, calendar), (sources.GMAIL, mail)) if fetched.missing)
    log.info("briefing_read", events=len(calendar.items), mail=len(mail.items), news=len(ranked.news),
             ranked=ranked.ranked, missing=missing, calendar_failed=bool(calendar.error), mail_failed=bool(mail.error),
             language=language or "en")
    if (not (calendar.items or mail.items or ranked.news) and not (calendar.error or mail.error)
            and missing == _told()):
        return {"status": "completed", "success": True, "verified": True, "output": t["nothing"],
                "nothing_to_tell": True}
    text, buttons = compose(day, calendar, mail, ranked, t)
    _remember_told(missing)
    # Every line is read from the accounts by code; only the order and gists are the model's.
    return {"status": "completed", "success": True, "verified": True, "output": text, "title": t["title"],
            "buttons": [asdict(b) for b in buttons]}
