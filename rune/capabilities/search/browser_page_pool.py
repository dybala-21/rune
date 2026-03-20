"""Browser page pool for parallel search operations.

Ported from src/capabilities/search/browser-page-pool.ts - manages a
fixed-size pool of Playwright pages, applying anti-bot fingerprinting
to each.  Uses the same acquire/release pattern as SearchRateLimiter.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import TYPE_CHECKING

from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page

log = get_logger(__name__)

CHROME_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# Anti-bot injection (mirrors browser.ts:502-516)

async def _apply_anti_bot(page: Page) -> None:
    """Inject fingerprint-spoofing scripts into a page."""
    await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => false });
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                { name: 'Native Client', filename: 'internal-nacl-plugin' },
            ],
        });
        Object.defineProperty(navigator, 'languages', {
            get: () => ['ko-KR', 'ko', 'en-US', 'en'],
        });
        window.chrome = { runtime: {}, loadTimes: () => {}, csi: () => {}, app: {} };
    """)


# BrowserPagePool

class BrowserPagePool:
    """Fixed-size pool of Playwright pages for parallel search.

    Usage::

        pool = BrowserPagePool(max_pages=3)
        await pool.warm_up()

        page = await pool.acquire()
        try:
            await page.goto("https://example.com")
        finally:
            pool.release(page)

        await pool.destroy()
    """

    def __init__(self, max_pages: int = 3) -> None:
        self._max_pages = max_pages
        self._pages: list[Page] = []
        self._available: list[Page] = []
        self._waiting: deque[asyncio.Future[Page]] = deque()
        self._browser: Browser | None = None

    # -- warm-up / lazy init ------------------------------------------------

    async def warm_up(self) -> None:
        """Eagerly acquire the shared browser singleton.

        Call this from goal classification when ``web``/``browser`` is
        detected to avoid cold-start latency on the first search.
        """
        if self._browser is None:
            self._browser = await self._get_shared_browser()

    async def _get_shared_browser(self) -> Browser:
        """Obtain a shared browser instance.

        Tries the capabilities browser module first, falls back to
        launching a new Playwright instance.
        """
        try:
            from rune.capabilities.browser import get_shared_browser
            return await get_shared_browser()
        except ImportError:
            pass

        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.launch(headless=True, channel="chrome")
        except Exception:
            browser = await pw.chromium.launch(headless=True)
        return browser

    # -- acquire / release --------------------------------------------------

    async def acquire(self) -> Page:
        """Acquire a page from the pool (blocking if pool is saturated)."""
        if self._browser is None:
            self._browser = await self._get_shared_browser()

        # 1. Return a healthy idle page
        while self._available:
            candidate = self._available.pop()
            if not candidate.is_closed():
                return candidate
            # Discard crashed page
            self._pages = [p for p in self._pages if p is not candidate]
            log.debug("page_pool_removed_closed_page")

        # 2. Create new page if under limit
        if len(self._pages) < self._max_pages:
            page = await self._browser.new_page(
                user_agent=CHROME_USER_AGENT,
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1,
                has_touch=False,
                is_mobile=False,
                locale="ko-KR",
                timezone_id="Asia/Seoul",
            )
            await _apply_anti_bot(page)
            self._pages.append(page)
            log.debug(
                "page_pool_created_page",
                current=len(self._pages),
                max=self._max_pages,
            )
            return page

        # 3. Pool saturated - queue and wait
        log.debug("page_pool_full_queuing")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Page] = loop.create_future()
        self._waiting.append(future)
        return await future

    def release(self, page: Page) -> None:
        """Return a page to the pool after use."""
        if page.is_closed():
            self._pages = [p for p in self._pages if p is not page]
            log.debug("page_pool_discarded_closed_on_release")
            # If someone is waiting, create a new page for them
            if self._waiting:
                asyncio.create_task(self._fulfill_waiting())
            return

        if self._waiting:
            future = self._waiting.popleft()
            if not future.done():
                future.set_result(page)
        else:
            self._available.append(page)

    async def _fulfill_waiting(self) -> None:
        """Create a new page for the next waiter (after a closed-page release)."""
        if not self._waiting:
            return
        try:
            page = await self.acquire()
            future = self._waiting.popleft()
            if not future.done():
                future.set_result(page)
            else:
                self._available.append(page)
        except Exception as exc:
            log.warning("page_pool_fulfill_failed", error=str(exc))

    # -- cleanup ------------------------------------------------------------

    async def destroy(self) -> None:
        """Close all pages and release the browser reference."""
        for page in self._pages:
            with contextlib.suppress(Exception):
                await page.close()
        self._pages.clear()
        self._available.clear()
        # Cancel pending waiters
        while self._waiting:
            future = self._waiting.popleft()
            if not future.done():
                future.cancel()
        self._browser = None
        log.debug("page_pool_destroyed")

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Total pages currently managed by the pool."""
        return len(self._pages)

    @property
    def available_count(self) -> int:
        """Number of idle pages ready for immediate use."""
        return len(self._available)
