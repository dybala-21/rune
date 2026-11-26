"""When a watched price is told to its owner.

Every reading is compared with the price the owner was last told, never with yesterday's reading:
a price that rose quietly and came back is not a drop. A reading that meets the owner's condition
is told only after a second one, at least a quarter of an hour later, meets it too, so a value a
cache or a pricing test showed once is not passed on. The two need not be equal: on a page whose
price moves every hour they never would be.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal

from rune.watch.item import Ended, Event, Failing, Item, Price, Told

CONFIRM_AFTER = timedelta(minutes=15)
EXPIRES_AFTER = timedelta(days=90)
# Said on the second failed day running, the way monitors hold a notice until a failure repeats.
TELL_FAILURE_ON_DAY = 2
END_AFTER_FAILED_DAYS = 14
END_AFTER_GONE_DAYS = 2
# A day this machine's web was closed says nothing about the page: it never ends a watch.
NOT_THE_PAGE = "web_closed"
# The places a price is read from stopped agreeing (rune.watch.places). Only the owner can say
# which of the amounts is the price, and a price that just split is often one that just went
# down, so they are asked the day it happens.
ASKS_OWNER = "split"


def meets(item: Item, price: Price) -> bool:
    """Whether a reading is what the owner asked to be told about."""
    if item.tell_when == "change":
        return (price.low, price.high) != (item.told.low, item.told.high)
    return price.low < item.told.low and (item.below is None or price.low <= item.below)


def due(item: Item, now: datetime, check_time: time) -> str | None:
    """The read this wake should make for the item: "day", "confirm", or none.

    ``now`` is in the owner's time zone. A day's read happens once, at or after the owner's time; a
    waiting price gets one confirming read of its own, and after that the next day's read settles it.
    """
    if item.status != "watching":
        return None
    if item.read_day != now.date() and (now.hour, now.minute) >= (check_time.hour, check_time.minute):
        return "day"
    if _waited(item, now) and not item.confirm_tried:
        return "confirm"
    return None


def settle(item: Item, now: datetime, read: str, outcome: Price | str) -> list[Event]:
    """Take in what the read ``due`` asked for gave: a price, or the kind of failure. Changes ``item``."""
    events: list[Event] = []
    if isinstance(outcome, Price) and outcome.currency != item.told.currency:
        outcome = "incomparable"  # another currency can't be set beside the price the owner knows
    if read == "day":
        item.read_day = now.date()
        if isinstance(outcome, str):
            _failed(item, outcome, now, events)
        else:
            item.failed_days = item.counted_days = item.gone_days = 0
            item.told_failure = False
            item.last_read, item.last_read_at = outcome, now
            if not meets(item, outcome):
                _drop_candidate(item)
            elif _waited(item, now):
                _tell(item, outcome, now, events)  # this read is the second one
            elif item.candidate is None:
                item.candidate, item.candidate_at, item.confirm_tried = outcome, now, False
    else:
        item.confirm_tried = True
        if isinstance(outcome, Price):  # a failed confirm leaves it for the next day's read
            item.last_read, item.last_read_at = outcome, now
            if meets(item, outcome):
                _tell(item, outcome, now, events)
            else:
                _drop_candidate(item)  # it did not last: nothing is told
    # A waiting price is settled first, so a drop is not lost to the watch ending.
    if item.status == "watching" and item.candidate is None and now - item.renewed_at >= EXPIRES_AFTER:
        _end(item, "expired", now, events)
    return events


def renew(item: Item, now: datetime, price: Price, *, tell_when: str, below: Decimal | None, label: str) -> None:
    """Start the item again from a reading just shown to the owner: that is now the price they know,
    and that day's read. ``now`` is in the owner's time zone."""
    item.tell_when, item.below, item.label = tell_when, below, label
    item.told, item.told_at, item.renewed_at = price, now, now
    item.last_read, item.last_read_at, item.read_day = price, now, now.date()
    item.failed_days = item.counted_days = item.gone_days = 0
    item.told_failure = False
    item.status, item.ended_why, item.ended_at = "watching", "", None
    _drop_candidate(item)


def stop(item: Item, now: datetime) -> None:
    """End the watch because the owner said so. Nothing is told: they know."""
    item.status, item.ended_why, item.ended_at = "ended", "stopped", now
    _drop_candidate(item)


def _waited(item: Item, now: datetime) -> bool:
    return item.candidate is not None and item.candidate_at is not None and now - item.candidate_at >= CONFIRM_AFTER


def _drop_candidate(item: Item) -> None:
    item.candidate, item.candidate_at, item.confirm_tried = None, None, False


def _tell(item: Item, second: Price, now: datetime, events: list[Event]) -> None:
    first = item.candidate
    assert first is not None
    # A drop is told at the higher of the two readings, so it is never overstated; if the lower one
    # holds, the next day tells it. A change is told as it stands now.
    new = max(first, second, key=lambda price: price.low) if item.tell_when == "drop" else second
    events.append(Told(before=item.told, now=new))
    item.told, item.told_at = new, now
    _drop_candidate(item)


def _failed(item: Item, kind: str, now: datetime, events: list[Event]) -> None:
    item.failed_days += 1
    item.gone_days = item.gone_days + 1 if kind == "gone" else 0
    if kind != NOT_THE_PAGE:
        item.counted_days += 1
    if item.gone_days >= END_AFTER_GONE_DAYS:
        _end(item, "gone", now, events)
    elif item.counted_days >= END_AFTER_FAILED_DAYS:
        _end(item, "unreadable", now, events)
    elif (item.failed_days >= TELL_FAILURE_ON_DAY or kind == ASKS_OWNER) and not item.told_failure:
        item.told_failure = True
        events.append(Failing(kind=kind, days=item.failed_days))


def _end(item: Item, why: str, now: datetime, events: list[Event]) -> None:
    item.status, item.ended_why, item.ended_at = "ended", why, now
    _drop_candidate(item)
    events.append(Ended(why=why, last=item.last_read))
