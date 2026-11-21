"""Sites RUNE does not watch because their terms forbid automated checking.

robots.txt says what a site lets a program read; terms of use can forbid it apart from that. A
site is listed here only with where its rule is written and when that was read.
"""

from __future__ import annotations

# Host (and every host under it): the rule, and when it was read.
DENIED: dict[str, str] = {
    "musinsa.com": ("Terms of use in force since 2024-03-28, article 11 (2): no monitoring by automated means, "
                    "members or not. Read 2026-11-20 through the operator's own store terms and press; its "
                    "robots.txt refuses unnamed programs as well."),
}


def denied(host: str) -> bool:
    return any(host == site or host.endswith("." + site) for site in DENIED)
