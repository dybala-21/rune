"""Browser-based DuckDuckGo search provider for RUNE.

Ported from src/capabilities/search/browser-search.ts - last-resort
fallback that uses Playwright to render DDG Lite when HTTP requests fail.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote as url_quote

from rune.capabilities.search.duckduckgo import parse_ddg_lite_results
from rune.capabilities.search.provider import SearchOptions, SearchProvider, SearchResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Block detection

def _detect_block(html: str, status: int) -> str:
    """Detect rate-limiting / captcha / blocking.

    Returns one of ``"ok"``, ``"captcha"``, ``"ratelimit"``, ``"blocked"``.
    """
    if status == 429:
        return "ratelimit"
    if status == 403:
        return "blocked"
    if "captcha" in html or "challenge-form" in html:
        return "captcha"
    if "result-link" not in html and len(html) < 5000:
        return "blocked"
    return "ok"


# Provider

class BrowserSearchProvider(SearchProvider):
    """Playwright-backed DuckDuckGo search fallback.

    Requires a ``page_pool`` object with ``acquire()`` and
    ``release(page)`` methods (e.g. ``BrowserPagePool``).
    """

    def __init__(self, page_pool: Any) -> None:
        self._pool = page_pool

    @property
    def name(self) -> str:  # noqa: D102
        return "browser-duckduckgo"

    async def search(self, options: SearchOptions) -> list[SearchResult]:
        """Perform a search via a headless browser."""
        effective_query = (
            f"site:{options.site} {options.query}" if options.site else options.query
        )

        page = None
        try:
            page = await self._pool.acquire()
            url = f"https://lite.duckduckgo.com/lite/?q={url_quote(effective_query)}"

            response = await page.goto(url, wait_until="domcontentloaded", timeout=10_000)
            html: str = await page.content()
            status = response.status if response else 200
            block_status = _detect_block(html, status)

            if block_status == "ok":
                return parse_ddg_lite_results(html, options.max_results)

            if block_status == "ratelimit":
                log.debug("browser_search_rate_limited")
                await asyncio.sleep(5.0)
                await page.goto(url, wait_until="domcontentloaded", timeout=10_000)
                html = await page.content()
                return parse_ddg_lite_results(html, options.max_results)

            # captcha / blocked
            log.debug("browser_search_blocked", status=block_status)
            return []

        except Exception as exc:
            log.error("browser_search_failed", error=str(exc))
            return []
        finally:
            if page is not None:
                self._pool.release(page)
