"""Make text from a mail, an event or a feed safe to show in the briefing.

Hidden markup and invisible characters go, and links and phone numbers are replaced: a briefing
that repeats a mail's link or number in RUNE's voice is how a summary becomes a phishing message.
The patterns below parse structured formats (markup, URLs, digits), not language.
"""

from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]*>")
_HIDDEN = re.compile(r"(?is)<(script|style|head|template)\b.*?</\1\s*>")
_URL = re.compile(r"(?i)\b(?:[a-z][a-z0-9+.-]*://|www\.)\S+")
# A bare domain such as "example.com/login": a dotted name ending in a letter-only label.
_DOMAIN = re.compile(r"(?i)\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_DIGITS = re.compile(r"\+?\d[\d ().-]{6,}\d")
_SPACE = re.compile(r"\s+")

LINK = "[link]"
NUMBER = "[number]"


def _numbers(text: str) -> str:
    # Nine digits or more: the shortest phone numbers, while dates (8 digits) and amounts stay.
    return _DIGITS.sub(lambda m: NUMBER if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), text)


def visible(text: str) -> str:
    """The text a person would see: markup and entities resolved, format characters dropped."""
    text = _HIDDEN.sub(" ", text)
    text = html.unescape(_TAG.sub(" ", text))
    # Cf holds zero-width and bidirectional controls, which can hide or reorder what is shown.
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def clean(text: object, limit: int = 200) -> str:
    """One line of shown text: no links, addresses or phone numbers, at most ``limit`` characters."""
    if not isinstance(text, str):
        return ""
    text = visible(text)
    text = _URL.sub(LINK, text)
    text = _EMAIL.sub(LINK, text)
    text = _DOMAIN.sub(LINK, text)
    text = _numbers(text)
    text = _SPACE.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def sender(value: object) -> str:
    """A mail's From header as a name and the sender's domain, e.g. "Kim (example.com)".

    The domain is the part a lookalike can't fake in the display name, so it is always shown."""
    if not isinstance(value, str):
        return "unknown sender"
    from email.utils import parseaddr

    name, address = parseaddr(value)
    domain = address.rsplit("@", 1)[-1].lower() if "@" in address else ""
    name = _SPACE.sub(" ", visible(name)).strip()
    # The name is shown as text, so a link or number hidden in it is removed like any other.
    name = _numbers(_DOMAIN.sub(LINK, _EMAIL.sub(LINK, _URL.sub(LINK, name))))[:60]
    if name and domain:
        return f"{name} ({domain})"
    return name or domain or "unknown sender"
