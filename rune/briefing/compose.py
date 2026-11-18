"""The briefing as one chat message: facts written by code, the model's gists beside them."""

from __future__ import annotations

from datetime import datetime

from rune.briefing.rank import Ranked
from rune.briefing.sources import Event, Fetched
from rune.channels.types import LinkButton

# Shown at most; Telegram keeps 4096 characters to a message.
MAX_MAIL_LINES = 8
NEWS_BUTTONS = 3


def _event(e: Event) -> str:
    when = "All day" if e.all_day else f"{e.start:%H:%M}–{e.end:%H:%M}"
    where = f" · {e.location}" if e.location else ""
    clash = "  (overlaps another event)" if e.conflict else ""
    return f"• {when} {e.title}{where}{clash}"


def _section(title: str, fetched: Fetched, lines: list[str], empty: str, label: str) -> list[str]:
    if fetched.missing:
        return [title, f"{label} isn't connected." + (" Connect it with the button below." if fetched.connect_url else "")]
    if fetched.error:
        return [title, f"{fetched.error}."]
    return [title, *(lines or [empty])]


def compose(day: datetime, calendar: Fetched, mail: Fetched, ranked: Ranked) -> tuple[str, list[LinkButton]]:
    events: list[Event] = calendar.items
    parts = [f"Good morning. {day:%A, %d %B}", ""]
    parts += _section(f"Today ({len(events)})" if events else "Today", calendar, [_event(e) for e in events],
                      "Nothing on the calendar.", "Calendar")
    parts.append("")
    shown = ranked.mail[:MAX_MAIL_LINES]
    lines = [f"• {m.sender}: {m.subject}" + (f" — {gist}" if gist else "") for m, gist in shown]
    if len(ranked.mail) > len(shown):
        lines.append(f"…and {len(ranked.mail) - len(shown)} more in Gmail.")
    parts += _section(f"Unread mail ({len(ranked.mail)})" if ranked.mail else "Unread mail", mail, lines,
                      "No new unread mail.", "Gmail")
    if ranked.news:
        parts += ["", "Headlines", *(f"• {h.title}" + (f" ({h.source})" if h.source else "") for h in ranked.news)]
    buttons = [LinkButton(f"Connect {label}", f.connect_url)
               for label, f in (("Calendar", calendar), ("Gmail", mail)) if f.missing and f.connect_url]
    buttons += [LinkButton(_short(h.title), h.url) for h in ranked.news[:NEWS_BUTTONS]]
    return "\n".join(parts).strip(), buttons


def _short(title: str, limit: int = 40) -> str:
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"
