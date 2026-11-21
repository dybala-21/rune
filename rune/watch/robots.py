"""Whether a site's robots.txt lets RUNE read an address, read the RFC 9309 way.

Python's ``urllib.robotparser`` takes the first rule that matches (before 3.14); the standard, and
the sites, mean the longest one. Shops write ``*`` and ``$`` in their rules, and allow or refuse
assistants by name, so the difference decides whether a page may be read at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote, unquote_to_bytes, urlsplit

# The name RUNE's requests carry, and the one a site addresses RUNE by in its robots.txt.
TOKEN = "RUNE"
# RFC 9309 2.5: at least this much of a robots.txt is read.
MAX_BYTES = 500 * 1024
_NAME = re.compile(r"[A-Za-z_-]+")
_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_UNRESERVED = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


@dataclass(frozen=True, slots=True)
class Rules:
    """The rules that apply to RUNE on one host: (allow, pattern) pairs, and the pause it asks for."""

    rules: tuple[tuple[bool, str], ...] = ()
    crawl_delay: float = 0.0


def parse(text: str, token: str = TOKEN) -> Rules:
    """The rules for ``token``: every group that names it, else every group for ``*``."""
    groups: list[tuple[list[str], list[tuple[bool, str]], list[float]]] = []
    naming = False  # the record before was a user-agent line
    # A byte-order mark before the first record is not part of it; kept, the first group is lost.
    for raw in text.lstrip("\ufeff").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if not naming:
                groups.append(([], [], []))
            name = _NAME.match(value)
            groups[-1][0].append("*" if value == "*" else (name.group() if name else value).lower())
            naming = True
        elif key in ("allow", "disallow"):
            naming = False
            if groups and value:  # an empty pattern matches nothing
                groups[-1][1].append((key == "allow", value))
        elif key == "crawl-delay" and groups:
            try:
                groups[-1][2].append(float(value))
            except ValueError:
                pass  # not a number: there is no pause to honor
        # Other records (sitemap, host) are not part of a group and don't end one.
    named = [group for group in groups if token.lower() in group[0]]
    chosen = named or [group for group in groups if "*" in group[0]]
    delays = [delay for group in chosen for delay in group[2] if delay > 0]
    return Rules(tuple(rule for group in chosen for rule in group[1]), max(delays, default=0.0))


def allows(rules: Rules, url: str) -> bool:
    """Whether ``url`` may be read: the longest matching rule decides, and Allow wins a tie."""
    parts = urlsplit(url)
    target = _normal((parts.path or "/") + (f"?{parts.query}" if parts.query else ""))
    if target == "/robots.txt":
        return True
    best: tuple[int, bool] | None = None
    for allow, pattern in rules.rules:
        pattern = _normal(pattern)
        if _matcher(pattern).match(target):
            rank = (len(pattern), allow)
            if best is None or rank > best:
                best = rank
    return best is None or best[1]


def read(status: int, body: bytes) -> Rules | str:
    """What a robots.txt response means: the rules, or the kind of refusal ("blocked", "robots")."""
    if 200 <= status < 300:
        return parse(body[:MAX_BYTES].decode("utf-8", "replace"))
    if status in (401, 403, 429):
        # Stricter than the standard, which treats any 4xx as "no rules": a site that refuses even
        # this file to RUNE is refusing RUNE.
        return "blocked"
    if 400 <= status < 500:
        return Rules()  # RFC 9309 2.3.1.3: no robots.txt, nothing is restricted
    return "robots"  # RFC 9309 2.3.1.4: unreachable, so assume it disallows


def _normal(path: str) -> str:
    """One spelling for comparison: escapes in upper case, plain characters unescaped, the rest escaped."""
    def spell(match: re.Match[str]) -> str:
        byte = unquote_to_bytes(match.group())
        return byte.decode("ascii") if byte[0] in _UNRESERVED else match.group().upper()

    return _ESCAPE.sub(spell, quote(path, safe="/%?=&*$:@!'(),;+~._-#[]"))


def _matcher(pattern: str) -> re.Pattern[str]:
    anchored = pattern.endswith("$")
    pieces = (pattern[:-1] if anchored else pattern).split("*")
    return re.compile(".*".join(re.escape(piece) for piece in pieces) + ("$" if anchored else ""))
