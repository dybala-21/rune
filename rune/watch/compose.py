"""What the watch has to tell, as one chat message.

The message holds what code read and nothing a site wrote: amounts, a currency code, dates, the
site's name from the address the owner gave, and what the owner called the product. The words
around them are labels (rune.watch.labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from rune.channels.types import LinkButton
from rune.watch.item import Ended, Event, Failing, Item, Price, Told
from rune.watch.judge import ASKS_OWNER, EXPIRES_AFTER
from rune.watch.labels import ENGLISH, why

# Shown at most; a message with more items still names them all.
MAX_BUTTONS = 5


@dataclass(frozen=True, slots=True)
class News:
    """One thing to tell about an item. ``told_at`` is when the owner last heard a price for it,
    before this."""

    item: Item
    event: Event
    told_at: datetime


def amount(value: Decimal) -> str:
    """An amount with its thousands grouped: whole when it is whole, else with two decimals or more."""
    value = value.normalize()
    exponent = value.as_tuple().exponent
    places = max(2, -exponent) if isinstance(exponent, int) and exponent < 0 else 0
    return f"{value:,.{places}f}"


def money(price: Price) -> str:
    """A price as it is shown: "369,000 KRW", or its lowest and highest when they differ."""
    shown = amount(price.low) if price.low == price.high else f"{amount(price.low)}–{amount(price.high)}"
    return f"{shown} {price.currency}"


def name(item: Item) -> str:
    """What the item is called: the owner's words for it and the site, or the site alone."""
    return f"{item.label} ({item.host})" if item.label else item.host


def compose(news: list[News], now: datetime, t: dict[str, str] = ENGLISH) -> tuple[str, list[LinkButton]]:
    """The message and its buttons. ``now`` is in the owner's time zone; every date is written in it."""
    told = [entry.event for entry in news if isinstance(entry.event, Told)]
    if len(told) != len(news):
        head = t["news"]
    elif all(event.now.low < event.before.low for event in told):
        head = t["went_down"]
    else:
        head = t["changed"]
    parts = [head]
    for entry in news:
        parts += ["", name(entry.item), *_lines(entry, now, t)]
    if told:
        parts += ["", t["as_written"]]
    items = list({entry.item.id: entry.item for entry in news}.values())
    buttons = [LinkButton(t["open"] if len(items) == 1 else _short(name(item)), item.url)
               for item in items[:MAX_BUTTONS]]
    return "\n".join(parts), buttons


def _lines(entry: News, now: datetime, t: dict[str, str]) -> list[str]:
    event = entry.event
    if isinstance(event, Told):
        return [f"{money(event.before)} → {money(event.now)}",
                t["told_and_read"].format(told=_day(entry.told_at, now), read=f"{now:%Y-%m-%d %H:%M}")]
    if isinstance(event, Failing):
        if event.kind == ASKS_OWNER:
            return [t["split"]]
        return [f"{t['failing'].format(days=event.days)} {t[why(event.kind)]}"]
    return _ended(entry.item, event, now, t)


def _ended(item: Item, event: Ended, now: datetime, t: dict[str, str]) -> list[str]:
    if event.why == "gone":
        return [t["ended_gone"]]
    if event.why == "unreadable":
        return [t["ended_unreadable"].format(days=item.counted_days)]
    lines = [t["ended_expired"].format(days=EXPIRES_AFTER.days)]
    if event.last is not None and item.last_read_at is not None:
        # The last price that was read, with its day: on a day the page couldn't be read, an older
        # price must not look like today's.
        lines.append(t["last_read"].format(amount=money(event.last), date=_day(item.last_read_at, now)))
    return lines


def _day(moment: datetime, now: datetime) -> str:
    return f"{moment.astimezone(now.tzinfo):%Y-%m-%d}"


def _short(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
