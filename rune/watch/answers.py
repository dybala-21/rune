"""What the watch tools answer: fixed English, which the model passes on in the owner's language.

Nothing here quotes a site. A refusal is worded by its kind, and the only figures are ones code
read from the page, so the owner can hold them against the page they have open.
"""

from __future__ import annotations

from datetime import datetime, time
from urllib.parse import urlsplit

from rune.daemon import owner_thread
from rune.watch.compose import amount, money, name
from rune.watch.extract import Reading
from rune.watch.item import BROWSER, Item
from rune.watch.judge import EXPIRES_AFTER
from rune.watch.schedule import Standing
from rune.watch.store import MAX_WATCHING

_ELSEWHERE = "Another address for the same product (the brand's own shop, a price-comparison page) can be tried."
_LATER = "It can be tried again later."
# Why a page isn't watched, by kind (rune.watch.item.Unread and the tools' own), and what can be done.
_REASONS: dict[str, tuple[str, str]] = {
    "not_owners": ("the address has to be one the owner typed in this conversation, passed exactly as they wrote it.",
                   "If they wrote one, call watch_set again with it copied to its last character. If they did not, "
                   "ask them for the product page's address."),
    "untyped": ("a page can only start being watched from a message the owner typed, and this run did not begin "
                "with one.", ""),
    "background": ("a page can only start being watched when the owner asks in a chat, not from a scheduled or "
                   "background run.", ""),
    "too_many": (f"{MAX_WATCHING} pages are being watched already, which is the most.",
                 "Stopping one with watch_stop makes room."),
    "address": ("the address has to be an https page on a site's own name, with no port, login or bare IP address "
                "in it.", ""),
    "denied": ("this site's own terms forbid automated checking.", _ELSEWHERE),
    "private": ("the address does not lead to a public site.", ""),
    "unreachable": ("the site did not answer.", _LATER),
    "web_closed": ("this machine's web access is closed: it holds data from the owner's connected accounts, so it "
                   "reaches only sites the owner has allowed.",
                   "Pages can't be watched from it while that is so. Do not look for a way round it."),
    "blocked": ("the site refuses automated reading.", _ELSEWHERE),
    "robots": ("this site's robots.txt does not allow automated reading.", _ELSEWHERE),
    "no_price": ("the page does not write a price that can be read without guessing.",
                 "A page for one product, on a shop that writes its price in the page's data, can be watched; a "
                 "search or category page cannot."),
    "many_products": ("the page writes several products at different prices, so which one is meant can't be told.",
                      "A page for one product can be watched."),
    "not_shown": ("that amount is not written in the page RUNE got in a way code can read again.",
                  "The page RUNE gets can differ from the one the owner sees when they are signed in or hold a "
                  "coupon. Ask the owner to check the number against the page; do not try other numbers yourself."),
    "split": ("the places its price was read from now give more than one amount, so which is the price can't be "
              "told.",
              "Ask the owner what price the page shows now, then call watch_set with this item and shown_price."),
    "gone": ("the page was not found.", "Check the address with the owner."),
    "moved": ("the address now leads to another site.", ""),
    "incomparable": ("the price is now written in another currency.", ""),
    "server": ("the site answered with an error.", _LATER),
    "timeout": ("the site took too long to answer.", _LATER),
    "too_large": ("the page is too large to read.", ""),
}
_OTHER = ("the page could not be read.", _LATER)
_ENDED = {"gone": "the page was gone", "unreadable": "it could not be read for too long",
          "expired": "its time ran out", "stopped": "the owner stopped it"}


def refused(kind: str, *, changing: bool = False) -> str:
    """Why nothing is watched (or, for a watch there already is, why nothing was changed)."""
    reason, hint = _REASONS.get(kind, _OTHER)
    if changing:
        lines = [f"Not changed: {reason} The watch stays as it was.", "Tell the owner so."]
    else:
        lines = [f"Not watched, and nothing was scheduled: {reason}",
                 "Do not schedule an agent task for this instead. Tell the owner it is not being watched, and why."]
    return "\n".join([*lines, hint] if hint else lines)


def asks() -> str:
    """What is said of a page that shows a price without writing it as data: the owner has to say
    what the price is. Nothing from the page is passed on."""
    return "\n".join([
        "Not watched yet, and nothing was scheduled: the page shows a price but does not write it as data a "
        "program can read without guessing, so the owner has to say what price it shows.",
        "Ask the owner what price the page shows now, and in which currency. RUNE then finds where that amount is "
        "written in the page and reads the same places each day.",
        "Do not guess the price or look it up, and do not schedule an agent task instead. When they have answered, "
        "call watch_set again with the same url, shown_price and currency."])


def shown(url: str) -> str:
    """The owner's address without its scheme, and without a query that may be long or private."""
    parts = urlsplit(url)
    return f"{parts.hostname or ''}{parts.path}" + ("?…" if parts.query else "")


def condition(item: Item) -> str:
    if item.tell_when == "change":
        return "that price changes, up or down"
    if item.below is None:
        return "that price goes down"
    return f"that price goes down to {amount(item.below)} {item.told.currency} or less"


def watching(item: Item, reading: Reading, *, at: time, job: Standing, zone: str, now: datetime,
             local: bool, again: bool) -> str:
    """What was set up, with the price just read."""
    called = f' as "{item.label}"' if item.label else ""
    lines = [f"Watching {shown(item.url)}{called} (item {item.id})" + (", as changed just now." if again else ".")]
    price = reading.price
    if reading.source == "pointed":
        count = "1 place" if len(item.places) == 1 else f"{len(item.places)} places"
        lines.append(f"Read now: {money(price)}, the amount the owner said the page shows. The page does not write "
                     f"its price as data, so it is read from {count} in the page that hold exactly this amount. If "
                     "they stop agreeing, the owner is asked for the price again. Tell the owner so.")
    elif price.low == price.high:
        offers = "1 offer" if reading.offers == 1 else f"{reading.offers} offers, all at this price"
        lines.append(f"Read now: {money(price)}, the price written on the page ({offers}).")
    else:
        lines.append(f"Read now: {money(price)}, the lowest and the highest price written on the page "
                     f"({reading.offers} offers). A drop is judged by the lowest.")
    if reading.listed:
        low, high = amount(reading.listed[0]), amount(reading.listed[-1])
        other = f"{low} {price.currency}" if low == high else f"{low}–{high} {price.currency}"
        lines.append(f"The page writes {other} as its price and {money(price)} as its sale price; the sale price "
                     "is watched." if reading.source == "og_sale" else f"The page also writes {other} as a list price.")
    if item.via == BROWSER:
        lines.append("This shop does not give the page to a program that asks for it, so RUNE opens it in its own "
                     "browser on this machine each time it is checked: a minimized window that carries RUNE's name, "
                     "in which nothing is clicked or signed in to. Tell the owner so.")
    late = ", or when this machine is next running after that" if local else ""
    lines.append(f"Checked once a day at {at:%H:%M} ({zone}){late}. The owner is told when {condition(item)}. "
                 + owner_thread.where(job.channel))
    if item.below is not None and price.low <= item.below:
        lines.append("It is already at or under that, so the message comes the next time it goes lower.")
    lines.append(f"The watch ends on {(now + EXPIRES_AFTER):%Y-%m-%d} unless it is renewed.")
    lines.append("Tell the owner the price that was read, so they can check it against the page they see.")
    return "\n".join(lines)


def job_line(job: Standing, at: time, zone: str) -> str:
    if job.state == "ok":
        return f"Pages are checked once a day at {at:%H:%M} ({zone}). " + owner_thread.where(job.channel)
    wrong = {"missing": "is missing", "paused": "is paused", "moved": "is on another schedule"}[job.state]
    return (f"Nothing is being checked now: the scheduled job named Watch {wrong}. "
            "Calling watch_set for any item puts it back.")


def line(item: Item, now: datetime, failure: str) -> str:
    """One item as watch_list shows it. ``failure`` is the kind of its last failed read, when it is failing."""
    def day(moment: datetime) -> str:
        return f"{moment.astimezone(now.tzinfo):%Y-%m-%d}"

    parts = [f"told {money(item.told)} on {day(item.told_at)}"]
    if item.last_read is not None and item.last_read_at is not None:
        parts.append(f"last read {money(item.last_read)} on {item.last_read_at.astimezone(now.tzinfo):%Y-%m-%d %H:%M}")
    if item.status != "watching":
        ended = day(item.ended_at) if item.ended_at is not None else "earlier"
        state = f"ended {ended} ({_ENDED.get(item.ended_why, 'it ended')}); watch_set with this item starts it again"
    else:
        state = f"watching, tells when {condition(item)}; ends {day(item.renewed_at + EXPIRES_AFTER)}"
        if item.failed_days:
            days = "day" if item.failed_days == 1 else f"{item.failed_days} days"
            state += f"; could not be read on the last {days} it was tried ({_REASONS.get(failure, _OTHER)[0].rstrip('.')})"
        if item.candidate is not None:
            state += f"; {money(item.candidate)} was read once and waits for a second reading"
    if item.via == BROWSER:
        parts.append("opened in RUNE's own browser on this machine")
    if item.places:
        parts.append("read from where the amount the owner pointed at is written in the page")
    return f"Item {item.id}: {name(item)}, {shown(item.url)}. {state}; {'; '.join(parts)}."
