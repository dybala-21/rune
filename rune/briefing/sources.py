"""Today's events and unread mail, read from the owner's connected accounts through the broker.

Every fact shown in the briefing comes from here, from the API's own fields: the model never
writes a time, a name or who sent what. An account that isn't connected comes back with the link
the owner connects it at.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, tzinfo
from typing import Any

from rune.briefing.sanitize import clean, sender
from rune.connectors.client import call_broker
from rune.connectors.models import ConnectorRequest
from rune.memory.provenance import record_connector
from rune.utils.logger import get_logger

log = get_logger(__name__)

# The connector each account is reached through, and the kind the owner connects it as.
CALENDAR, GMAIL = "calendar", "gmail"
KINDS = {CALENDAR: "calendar-read", GMAIL: "gmail-read"}
MAX_EVENTS = 20
MAX_MAIL = 10
# Unread from the last day, leaving out what Gmail itself files away from the inbox's main tab.
MAIL_QUERY = "is:unread newer_than:1d -category:promotions -category:social -category:forums"


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Event:
    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    location: str = ""
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class Mail:
    id: str
    sender: str
    subject: str
    snippet: str
    important: bool = False
    received: datetime | None = None


@dataclass(slots=True)
class Fetched:
    items: list[Any] = field(default_factory=list)
    # The account isn't connected; ``connect_url`` is where the owner connects it, if there is a page.
    missing: bool = False
    connect_url: str = ""
    # Why it couldn't be read, for the log; the briefing says so in its own words (labels).
    error: str = ""


async def listed() -> dict[str, dict[str, Any]]:
    answer = await call_broker("list", {})
    return {c["name"]: c for c in answer.get("connectors") or [] if isinstance(c, dict) and "name" in c}


async def connect_url(kind: str) -> str:
    """The approval page's connect link for ``kind``, or "" where there is no such page."""
    try:
        status = await call_broker("status", {})
    except Exception as exc:
        log.debug("briefing_status_unavailable", error=type(exc).__name__)
        return ""
    page = status.get("approval_url") if isinstance(status, dict) else None
    return f"{page.rstrip('/')}/connect/{kind}" if isinstance(page, str) and page.startswith("https://") else ""


async def _get(connector: dict[str, Any], path: str, query: dict[str, str]) -> Any:
    request = ConnectorRequest(connector=connector["name"], origin=connector["origin"], revision=connector["revision"],
                               method="GET", path=path, query=query, max_length=50000)
    # Before it goes out, as for any connector call: what this run keeps may now hold its data.
    record_connector(request.connector)
    result = await call_broker("request", request.model_dump())
    if not result.get("success"):
        raise SourceError(str(result.get("error") or "The request failed"))
    try:
        return json.loads(result.get("output") or "")
    except ValueError as exc:
        raise SourceError("The answer could not be read") from exc


def _when(value: object, zone: tzinfo) -> tuple[datetime, bool] | None:
    if not isinstance(value, dict):
        return None
    try:
        if isinstance(value.get("dateTime"), str):
            return datetime.fromisoformat(value["dateTime"]).astimezone(zone), False
        if isinstance(value.get("date"), str):
            return datetime.combine(datetime.fromisoformat(value["date"]).date(), time(), zone), True
    except ValueError:
        return None
    return None


def events_of(answer: Any, zone: tzinfo) -> list[Event]:
    """Events from a Calendar events.list answer: cancelled and declined ones left out, overlaps marked."""
    found: list[tuple[Event, bool]] = []
    for item in (answer or {}).get("items") or []:
        if not isinstance(item, dict) or item.get("status") == "cancelled":
            continue
        if any(isinstance(a, dict) and a.get("self") and a.get("responseStatus") == "declined"
               for a in item.get("attendees") or []):
            continue
        start, end = _when(item.get("start"), zone), _when(item.get("end"), zone)
        if start is None or end is None:
            continue
        event = Event(title=clean(item.get("summary"), 120), start=start[0], end=end[0],
                      all_day=start[1], location=clean(item.get("location"), 80))
        found.append((event, item.get("transparency") != "transparent"))
    timed = [e for e, busy in found if busy and not e.all_day]
    events = []
    for event, busy in found:
        clash = busy and not event.all_day and any(o is not event and o.start < event.end and event.start < o.end
                                                   for o in timed)
        events.append(replace(event, conflict=clash))
    return sorted(events, key=lambda e: (not e.all_day, e.start))[:MAX_EVENTS]


def mail_of(answer: Any) -> Mail | None:
    """A Mail from a Gmail messages.get answer (format=metadata)."""
    if not isinstance(answer, dict) or not isinstance(answer.get("id"), str):
        return None
    headers = {h.get("name", "").lower(): h.get("value", "")
               for h in (answer.get("payload") or {}).get("headers") or [] if isinstance(h, dict)}
    received = None
    try:
        received = datetime.fromtimestamp(int(str(answer.get("internalDate"))) / 1000).astimezone()
    except (TypeError, ValueError):
        pass
    return Mail(id=answer["id"], sender=sender(headers.get("from")), subject=clean(headers.get("subject"), 120),
                snippet=clean(answer.get("snippet"), 300), important="IMPORTANT" in (answer.get("labelIds") or []),
                received=received)


async def calendar(connectors: dict[str, dict[str, Any]], zone: tzinfo, day: datetime) -> Fetched:
    connector = connectors.get(CALENDAR)
    if connector is None:
        return Fetched(missing=True, connect_url=await connect_url(KINDS[CALENDAR]))
    start = datetime.combine(day.date(), time(), zone)
    query = {"timeMin": start.isoformat(), "timeMax": (start + timedelta(days=1)).isoformat(),
             "singleEvents": "true", "orderBy": "startTime", "maxResults": str(MAX_EVENTS * 2),
             "fields": "items(summary,status,location,transparency,start,end,attendees(self,responseStatus))"}
    try:
        return Fetched(items=events_of(await _get(connector, "/calendar/v3/calendars/primary/events", query), zone))
    except Exception as exc:
        log.warning("briefing_calendar_failed", error=type(exc).__name__)
        return Fetched(error="The calendar could not be read")


async def mail(connectors: dict[str, dict[str, Any]]) -> Fetched:
    connector = connectors.get(GMAIL)
    if connector is None:
        return Fetched(missing=True, connect_url=await connect_url(KINDS[GMAIL]))
    try:
        listing = await _get(connector, "/gmail/v1/users/me/messages",
                             {"q": MAIL_QUERY, "maxResults": str(MAX_MAIL), "fields": "messages(id)"})
        ids = [m["id"] for m in (listing or {}).get("messages") or [] if isinstance(m, dict) and isinstance(m.get("id"), str)]
        gate = asyncio.Semaphore(4)

        async def one(message_id: str) -> Mail | None:
            async with gate:
                answer = await _get(connector, f"/gmail/v1/users/me/messages/{message_id}",
                                    {"format": "metadata", "fields": "id,labelIds,snippet,internalDate,payload/headers"})
            return mail_of(answer)

        found = await asyncio.gather(*(one(i) for i in ids[:MAX_MAIL] if i.isalnum()))
        return Fetched(items=[m for m in found if m is not None])
    except Exception as exc:
        log.warning("briefing_mail_failed", error=type(exc).__name__)
        return Fetched(error="Mail could not be read")
