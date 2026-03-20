"""Browser profile auto-detection for RUNE.

Determines the appropriate browser profile (managed or relay) based on
the user's goal using tiered heuristics.

- **managed** (headless): scraping, data extraction, screenshots - user
  doesn't need to see the browser.
- **relay** (user's Chrome): login-required sites, "open/show me",
  interactive tasks where the user wants to watch or has auth sessions.

Ported from src/browser/auto-detect.ts - Tier 1 regex patterns.
"""

from __future__ import annotations

import re
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

BrowserProfile = Literal["managed", "relay"]

# Tier 1: Relay indicators (user wants to SEE or needs AUTH)

_RELAY_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    # Direct auth requests (highest confidence)
    (re.compile(r"로그인|sign\s*in|log\s*in", re.I), "auth_direct", 0.95),
    # "Open/show/launch browser" intent (Korean + English)
    (re.compile(r"(브라우저|browser).{0,10}(열|띄|켜|실행|open|launch|show)", re.I), "browser_open", 0.90),
    (re.compile(r"(열어|띄워|켜서|실행해).{0,10}(봐|줘|주세요|보여)", re.I), "open_show", 0.85),
    # Possessive / personal ("내 X 확인해")
    (re.compile(r"내\s+.{1,20}(확인|보여|열어|체크)", re.I), "possessive", 0.70),
    # Auth-required services
    (re.compile(r"\b(gmail|outlook|메일|mail|slack|notion|dashboard|대시보드)\b", re.I), "auth_service", 0.70),
    # Auth domains
    (re.compile(r"(mail\.google|accounts\.google|github\.com/settings)", re.I), "auth_domain", 0.85),
]

# Tier 1: Managed indicators (scraping, data extraction - no UI needed)

_MANAGED_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    # Explicit scraping / data extraction
    (re.compile(r"(스크래핑|scrape|scraping|crawl|크롤링)", re.I), "scraping", 0.90),
    # "Search/lookup/query" without "show me" → background data retrieval
    (re.compile(r"(조회|검색|찾아|알려줘|알아봐).{0,5}$", re.I), "data_query", 0.75),
    # Screenshot to file
    (re.compile(r"(스크린샷|screenshot|캡처).{0,10}(저장|찍어|해줘)", re.I), "screenshot", 0.80),
    # Explicit headless / background intent
    (re.compile(r"(headless|백그라운드|background)", re.I), "headless_explicit", 0.95),
]


def detect_browser_profile(goal: str) -> BrowserProfile:
    """Detect the appropriate browser profile for a given goal.

    Tier 1: Pattern matching with confidence scores.
    Tier 2: Default to managed (headless) - safest for automation.

    Returns ``"relay"`` if the user likely wants to see the browser or
    needs login sessions, ``"managed"`` otherwise.
    """
    if not goal:
        return "managed"

    best_relay = 0.0
    best_relay_reason = ""
    best_managed = 0.0
    best_managed_reason = ""

    for pattern, reason, confidence in _RELAY_PATTERNS:
        if pattern.search(goal) and confidence > best_relay:
            best_relay = confidence
            best_relay_reason = reason

    for pattern, reason, confidence in _MANAGED_PATTERNS:
        if pattern.search(goal) and confidence > best_managed:
            best_managed = confidence
            best_managed_reason = reason

    # Relay wins if it has higher confidence, or if both are equal
    # (prefer showing the browser when ambiguous with user intent).
    if best_relay > 0 and best_relay >= best_managed:
        log.debug(
            "browser_profile_detected",
            profile="relay",
            reason=best_relay_reason,
            confidence=best_relay,
        )
        return "relay"

    if best_managed > 0:
        log.debug(
            "browser_profile_detected",
            profile="managed",
            reason=best_managed_reason,
            confidence=best_managed,
        )
        return "managed"

    # Default: managed (headless) - safest for automation tasks.
    log.debug("browser_profile_detected", profile="managed", reason="default")
    return "managed"
