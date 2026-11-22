"""What the owner typed, for tools that may act only on the owner's own words.

A tool that reads an address must not take one a model wrote: an instruction hidden in a page or a
mail could have the model write the owner's data into an address and send it away. Such a tool asks
here whether the address stands in text a person typed, in this turn or in the conversation's
earlier user turns.

Only an entry point that takes a person's typing fills this in, through ``prepare_agent_context``.
A run a schedule started, a sub-run and a goal a model wrote never do, and a sub-run only reads what
the run above it was given.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Iterator
from contextvars import ContextVar

_typed: ContextVar[tuple[str, ...] | None] = ContextVar("owner_words", default=None)
# Characters no address holds (RFC 3986), so one ends at any of them.
_ENDS = frozenset(' \t\r\n\f\v"<>\\^`{|}')
_CLOSING = frozenset(".,;:!?)]'*")
_OPENING = frozenset("(<[{\"'")


def set_typed(texts: Iterable[str] | None) -> None:
    """Keep what a person typed for the run that starts here; None when no person typed it."""
    if texts is None:
        _typed.set(None)
        return
    # Slack sends "&" as "&amp;", so each text counts as written and with its entities resolved.
    _typed.set(tuple(dict.fromkeys(form for text in texts for form in (text, html.unescape(text)) if form)))


def known() -> bool:
    """Whether this run began with something a person typed."""
    return _typed.get() is not None


def wrote(words: str) -> bool:
    """Whether the owner typed exactly these words."""
    words = words.strip()
    return bool(words) and any(words in text for text in _typed.get() or ())


def wrote_address(url: str) -> bool:
    """Whether the owner typed this address, whole: not a longer one that only begins with it.

    They may have left "https://" off, and words or punctuation may stand right against it
    ("shop.example/p/1을", "(https://shop.example/p/1).").
    """
    url = url.strip()
    bare = url.removeprefix("https://")
    if not url or not bare:
        return False
    for text in _typed.get() or ():
        if any(_starts(text[:at], full=True) and _ends(text[at + len(url):]) for at in _found(text, url)):
            return True
        if bare != url and any(_starts(text[:at], full=False) and _ends(text[at + len(bare):])
                               for at in _found(text, bare)):
            return True
    return False


def _found(text: str, part: str) -> Iterator[int]:
    at = text.find(part)
    while at != -1:
        yield at
        at = text.find(part, at + 1)


def _starts(before: str, *, full: bool) -> bool:
    """Whether an address can begin after ``before``. One written with its scheme begins anywhere a
    word doesn't run into it; one written without must stand free, or it is a piece of another
    address (the "shop.example/p" in "sub.shop.example/p")."""
    if not full and before.endswith("http://"):
        before, full = before[:-7], True
    if not before or before[-1].isspace() or not before[-1].isascii():
        return True
    return not before[-1].isalnum() if full else before[-1] in _OPENING


def _ends(after: str) -> bool:
    """Whether the address is over where the match is: up to the next character no address holds,
    only closing punctuation or words in another script follow."""
    for character in after:
        if character in _ENDS:
            break
        if character.isascii() and character not in _CLOSING:
            return False
    return True
