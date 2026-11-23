"""A price a page shows but does not write as structured data, read from where it is written.

Some shops put the price a visitor sees nowhere in their page's structured data. The number is
still in the page for the shop's own scripts: as a meta tag's content, or as the value of a key in
an inline script. Which of those places holds the price can't be told from the page, so it is not
guessed. The owner says what price the page shows; the places that hold exactly that amount, and
no other, are kept with the watch; and each day code reads the same places again.

- Only an amount written the way a program writes one is read (``rune.watch.extract.readable``).
  What a page shows to a person ("39,99") is never parsed.
- The places must agree. When they give different amounts, or one of them now holds more than one,
  nothing says which is the price: the read fails as ``split`` and the owner is asked again. No
  vote is taken, since places that were equal by chance could out-vote the one that is the price.
- A place that is gone says nothing. When all of them are gone, the page no longer writes the
  price where it did.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser

from rune.watch.extract import readable
from rune.watch.item import Unread

# As many places as are kept for one watch.
MAX_PLACES = 12
_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
# A key and the number written as its value, in a script: `"price": 19900`, `price = '19900.00'`,
# and the same inside a string that holds JSON (`\"price\":19900`). The value is a number as a
# program writes one, bare or alone between its quotes: "19,900" is text for a person and is passed
# over, not read as 19.
_PAIR = re.compile(r"""(?<![\w$.])(?:\\*["'])?([A-Za-z_$][\w$]{0,63})(?:\\*["'])?\s*[:=]\s*"""
                   r"""(?:(\\*["'])(-?[0-9]+(?:\.[0-9]+)?)\2|(-?[0-9]+(?:\.[0-9]+)?)(?![\w.]|,[0-9]))""")


class _Page(HTMLParser):
    """The page's meta tags and the text of its inline scripts; nothing else is kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.metas: list[tuple[str, str]] = []
        self.scripts: list[str] = []
        self._inline = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        given = {name.lower(): value or "" for name, value in attrs}
        if tag == "meta":
            key = (given.get("property") or given.get("name") or given.get("itemprop") or "").strip().lower()
            if key:
                self.metas.append((key, given.get("content", "").strip()))
        self._inline = tag == "script" and "src" not in given

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._inline = False

    def handle_data(self, data: str) -> None:
        if self._inline:
            self.scripts.append(data)


def written(html: str) -> dict[str, frozenset[Decimal]]:
    """Every place in the page that holds a number, with the numbers it holds. A place is named by
    its kind and its key: ``meta:eg:saleprice``, ``key:SellingPrice``."""
    page = _Page()
    page.feed(html)
    page.close()
    held: dict[str, set[Decimal]] = {}
    for key, content in page.metas:
        if _NUMBER.fullmatch(content):
            held.setdefault(f"meta:{key}", set()).update(_number(content))
    for script in page.scripts:
        for key, _, quoted, bare in _PAIR.findall(script):
            held.setdefault(f"key:{key}", set()).update(_number(quoted or bare))
    return {place: frozenset(numbers) for place, numbers in held.items()}


def holding(html: str, amount: Decimal, among: Sequence[str] = ()) -> tuple[str, ...]:
    """The places that hold ``amount`` and no other number. Those of ``among`` when any of them
    does: a place that held the price before and holds it now is the likelier one to keep doing so."""
    found = sorted(place for place, numbers in written(html).items() if numbers == {amount})
    before = set(among)
    kept = [place for place in found if place in before]
    return tuple((kept or found)[:MAX_PLACES])


def read(html: str, places: Sequence[str]) -> Decimal:
    """The amount the kept places hold now. Raises ``Unread``: "no_price" when none of them is in
    the page any more, "split" when they do not agree on one amount that can be read."""
    held = written(html)
    there = [held[place] for place in places if place in held]
    if not there:
        raise Unread("no_price")
    amounts = {amount for numbers in there for amount in numbers}
    if len(amounts) != 1 or (amount := readable(next(iter(amounts)))) is None:
        raise Unread("split")
    return amount


def _number(text: str) -> list[Decimal]:
    try:
        return [Decimal(text)]
    except InvalidOperation:
        return []

