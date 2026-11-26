"""The briefing as one chat message: facts written by code, the model's gists beside them.

The wording around the facts comes from ``labels`` (the owner's language); a date and a time are
written as numbers, which read the same in any of them.
"""

from __future__ import annotations

from datetime import datetime

from rune.briefing.labels import ENGLISH
from rune.briefing.rank import Ranked
from rune.briefing.sources import Event, Fetched
from rune.channels.types import LinkButton

# Shown at most; Telegram keeps 4096 characters to a message.
MAX_MAIL_LINES = 8
NEWS_BUTTONS = 3


def _event(e: Event, t: dict[str, str]) -> str:
    when = t["all_day"] if e.all_day else f"{e.start:%H:%M}–{e.end:%H:%M}"
    where = f" · {e.location}" if e.location else ""
    clash = f"  {t['overlaps']}" if e.conflict else ""
    return f"• {when} {e.title or t['no_title']}{where}{clash}"


def _section(title: str, fetched: Fetched, lines: list[str], empty: str, missing: str, failed: str) -> list[str]:
    if fetched.missing:
        return [title, missing]
    if fetched.error:
        return [title, failed]
    return [title, *(lines or [empty])]


def _counted(title: str, count: int) -> str:
    return f"{title} ({count})" if count else title


def compose(day: datetime, calendar: Fetched, mail: Fetched, ranked: Ranked,
            t: dict[str, str] = ENGLISH) -> tuple[str, list[LinkButton]]:
    events: list[Event] = calendar.items
    below = f" {t['connect_below']}"
    parts = [f"{t['greeting']} {day:%Y-%m-%d}", ""]
    parts += _section(_counted(t["today"], len(events)), calendar, [_event(e, t) for e in events], t["no_events"],
                      t["calendar_missing"] + (below if calendar.connect_url else ""), t["calendar_failed"])
    parts.append("")
    shown = ranked.mail[:MAX_MAIL_LINES]
    lines = [f"• {m.sender or t['unknown_sender']}: {m.subject or t['no_subject']}" + (f" — {gist}" if gist else "")
             for m, gist in shown]
    if len(ranked.mail) > len(shown):
        lines.append(t["more_mail"].format(count=len(ranked.mail) - len(shown)))
    parts += _section(_counted(t["mail"], len(ranked.mail)), mail, lines, t["no_mail"],
                      t["mail_missing"] + (below if mail.connect_url else ""), t["mail_failed"])
    if ranked.news:
        parts += ["", t["headlines"], *(f"• {h.title}" + (f" ({h.source})" if h.source else "") for h in ranked.news)]
    buttons = [LinkButton(t[label], f.connect_url)
               for label, f in (("connect_calendar", calendar), ("connect_mail", mail)) if f.missing and f.connect_url]
    buttons += [LinkButton(_short(h.title), h.url) for h in ranked.news[:NEWS_BUTTONS]]
    return "\n".join(parts).strip(), buttons


def _short(title: str, limit: int = 40) -> str:
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"
