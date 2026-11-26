"""The watch's fixed wording, in the owner's language.

Every fact in a watch message (a price, a currency code, a date, a site's name) is written by code.
The words around the facts are these labels: English is built in, and another language is
translated once from this list alone and checked label by label (rune.briefing.labels), so no page
has a say in them.
"""

from __future__ import annotations

from rune.briefing import labels
from rune.briefing.labels import Wording

ENGLISH: dict[str, str] = {
    "title": "Price watch",
    "went_down": "A price you asked me to watch went down.",
    "changed": "A price you asked me to watch changed.",
    "news": "About the pages you asked me to watch:",
    "told_and_read": "Last told {told} · read {read}",
    "as_written": "Prices are as the page writes them for a visitor who is not signed in.",
    "failing": "Could not be read for {days} days in a row.",
    "split": "The page now gives more than one amount where its price was read, so I can't tell which is the "
             "price. Tell me the price it shows now and I will keep watching.",
    "ended_gone": "The page is gone, so it is no longer watched.",
    "ended_unreadable": "It could not be read for {days} days in a row, so it is no longer watched.",
    "ended_expired": "It has been watched for {days} days, so the watch ended. Tell me to keep watching and I will.",
    "last_read": "Last read: {amount} on {date}",
    "why_blocked": "The site refuses automated reading.",
    "why_rules": "The site's rules for automated programs do not allow reading it.",
    "why_web_closed": "This machine's web access is closed.",
    "why_no_price": "The page no longer writes a price that can be read.",
    "why_moved": "The address now leads to another site.",
    "why_currency": "The price is now written in another currency.",
    "why_gone": "The page was not found.",
    "why_unreachable": "The site did not answer, or answered with an error.",
    "open": "Open the page",
    "failed": "The pages you asked me to watch could not be checked on {date}.",
}

# The label that says why a read failed, by the kind of failure (rune.watch.item.Unread).
WHY = {"blocked": "why_blocked", "robots": "why_rules", "denied": "why_rules", "web_closed": "why_web_closed",
       "no_price": "why_no_price", "many_products": "why_no_price", "moved": "why_moved",
       "incomparable": "why_currency", "gone": "why_gone"}

_PROMPT = """You translate the fixed wording of a personal assistant's price-watch messages: the
assistant reads a product page's price for the person it works for and tells them when it changes.

You get a language tag (BCP-47) and a JSON object of English labels. Return the same keys, each
label written naturally in that language, in the polite register an assistant uses with the person
it works for, and about as short as the English. Keep every {placeholder} exactly as written, and
add nothing: no numbers, links or notes."""

WORDING = Wording(ENGLISH, _PROMPT, "watch-labels", "watch_labels")


def why(kind: str) -> str:
    """The label for a failed read of this kind; one that isn't named reads as no answer."""
    return WHY.get(kind, "why_unreachable")


async def load(language_tag: str) -> dict[str, str]:
    """The labels in ``language_tag``; English when it is English, unset, or can't be had."""
    return await labels.load(language_tag, WORDING)
