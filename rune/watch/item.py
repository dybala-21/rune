"""What is kept about one watched page, and what the watch has to say about it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Price:
    """The lowest and the highest price a page writes, in one currency."""

    low: Decimal
    high: Decimal
    currency: str


# The two ways a page is read: asked for under RUNE's name, or opened in RUNE's own browser.
FETCH, BROWSER = "fetch", "browser"


@dataclass(slots=True)
class Item:
    id: int
    # As the owner wrote it: the one address that is shown and linked.
    url: str
    # Where that led when it was registered: read from, never shown (a server chose it).
    fetch_url: str
    host: str
    # What the owner called it, in their own words; "" when they gave none.
    label: str
    tell_when: str  # "change" | "drop"
    below: Decimal | None
    # The price the owner was last told. Registering counts as telling.
    told: Price
    told_at: datetime
    renewed_at: datetime
    # A price that met the owner's condition once and waits for a second reading.
    candidate: Price | None = None
    candidate_at: datetime | None = None
    confirm_tried: bool = False
    # The owner's local day of the last day's read, done or only tried.
    read_day: date | None = None
    failed_days: int = 0
    # Failed days, leaving out those when this machine's web was closed.
    counted_days: int = 0
    gone_days: int = 0
    told_failure: bool = False
    last_read: Price | None = None
    last_read_at: datetime | None = None
    status: str = "watching"  # "watching" | "ended"
    ended_why: str = ""  # "gone" | "unreadable" | "expired" | "stopped"
    ended_at: datetime | None = None
    # The way its page is read, chosen when its owner set the watch: FETCH or BROWSER.
    via: str = FETCH
    # For a page that writes no structured price: the places in it that held the amount its owner
    # said it shows, which are read again each day (rune.watch.places). Empty for any other page.
    places: tuple[str, ...] = ()
    # Goes up with every save, so a save from a copy read before another save is refused.
    version: int = 0


@dataclass(frozen=True, slots=True)
class Told:
    """The price met the owner's condition on two readings."""

    before: Price
    now: Price


@dataclass(frozen=True, slots=True)
class Failing:
    """The page could not be read on ``days`` days running; said once."""

    kind: str
    days: int


@dataclass(frozen=True, slots=True)
class Ended:
    """The watch is over: the page is gone, stayed unreadable, or its time ran out."""

    why: str  # "gone" | "unreadable" | "expired"
    last: Price | None


Event = Told | Failing | Ended


class Unread(Exception):
    """A page could not be read for the watch. ``kind`` is one of a fixed set of words, never the
    site's own text: it is what gets stored and what the owner's message is chosen by."""

    def __init__(self, kind: str, status: int = 0) -> None:
        super().__init__(kind)
        self.kind = kind
        self.status = status  # the HTTP status that said so, when one did
