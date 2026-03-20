"""BrowserTool - web automation via Playwright.

Ported from src/tools/browser.ts.  Playwright is loaded lazily so
that the dependency is optional at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rune.tools.base import Tool
from rune.types import Domain, RiskLevel, ToolResult
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext

log = get_logger(__name__)

# Default deny-list domains (mirrors defaultPolicy.browser.denyDomains)
_DENY_DOMAINS: list[str] = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
]

_CHROME_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class BrowserTool(Tool):
    """Web browser automation (navigate, observe, screenshot, interact).

    Uses Playwright under the hood.  System Chrome is preferred; falls back
    to bundled Chromium.
    """

    # -- Tool properties ----------------------------------------------------

    @property
    def name(self) -> str:
        return "browser"

    @property
    def domain(self) -> Domain:
        return Domain.BROWSER

    @property
    def description(self) -> str:
        return "Web browser automation (navigate, observe, screenshot, interact)"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM

    @property
    def actions(self) -> list[str]:
        return [
            "navigate",
            "observe",
            "screenshot",
            "extract_text",
            "click",
            "input",
            "submit",
            "close",
        ]

    # -- internal state -----------------------------------------------------

    def __init__(self) -> None:
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._initialized: bool = False

    # -- lazy init ----------------------------------------------------------

    async def _initialize(self) -> None:
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright

            pw = await async_playwright().start()

            # Try system Chrome first, then Chromium fallback
            try:
                self._browser = await pw.chromium.launch(
                    headless=False, channel="chrome"
                )
                log.info("browser_tool_initialized", engine="system Chrome")
            except Exception:
                log.debug("system_chrome_not_found")
                self._browser = await pw.chromium.launch(headless=False)
                log.info("browser_tool_initialized", engine="Chromium")

            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=_CHROME_USER_AGENT,
            )
            self._initialized = True

        except Exception as exc:
            log.error("browser_init_failed", error=str(exc))
            raise RuntimeError(
                "Browser not available. Install Playwright: "
                "pip install playwright && playwright install chromium"
            ) from exc

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _matches_domain(url: str, pattern: str) -> bool:
        """Return True if *url* matches the deny-domain *pattern*."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
            hostname = parsed.hostname or ""
            return hostname == pattern or hostname.endswith(f".{pattern}")
        except Exception:
            return False

    # -- abstract method implementations ------------------------------------

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        action = params.get("action", "")
        if not action:
            return False, "Missing action parameter"
        if action not in self.actions:
            return False, f"Unknown action: {action}"

        if action == "navigate":
            url = params.get("url", "")
            if not url:
                return False, "Missing url parameter"
            for pattern in _DENY_DOMAINS:
                if self._matches_domain(url, pattern):
                    return False, f"Domain denied by policy: {url}"

        return True, ""

    async def simulate(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        # Read-only actions can be executed directly
        if action in ("observe", "screenshot", "extract_text"):
            return await self.execute(params)
        return self.success(
            data={
                "simulation": True,
                "action": action,
                "params": params,
                "message": "This action would be executed",
            }
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")

        valid, err = await self.validate(params)
        if not valid:
            return self.failure(err)

        await self._initialize()
        assert self._context is not None

        try:
            if action == "navigate":
                return await self._navigate(params)
            elif action == "observe":
                return await self._observe(params)
            elif action == "screenshot":
                return await self._screenshot(params)
            elif action == "extract_text":
                return await self._extract_text(params)
            elif action == "click":
                return await self._click(params)
            elif action == "input":
                return await self._input(params)
            elif action == "submit":
                return await self._submit(params)
            elif action == "close":
                return await self._close()
            else:
                return self.failure(f"Unknown action: {action}")
        except Exception as exc:
            return self.failure(f"Browser action failed: {exc}")

    async def rollback(self, rollback_data: dict[str, Any]) -> ToolResult:
        # Browser actions are generally not rollback-able
        return self.success(data={"message": "Browser actions cannot be rolled back"})

    async def health_check(self) -> bool:
        try:
            from playwright.async_api import async_playwright  # noqa: F401
            return True
        except ImportError:
            return False

    # -- action implementations ---------------------------------------------

    async def _navigate(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        url: str = params["url"]
        page = await self._context.new_page()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        status = response.status if response else 0
        title = await page.title()
        return self.success(data={"url": url, "status": status, "title": title})

    async def _observe(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        pages = self._context.pages
        if not pages:
            return self.failure("No open pages")
        page = pages[-1]
        title = await page.title()
        url = page.url
        return self.success(data={"url": url, "title": title})

    async def _screenshot(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        pages = self._context.pages
        if not pages:
            return self.failure("No open pages")
        page = pages[-1]
        path = params.get("path", "/tmp/rune_screenshot.png")
        await page.screenshot(path=path)
        return self.success(data={"path": path})

    async def _extract_text(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        pages = self._context.pages
        if not pages:
            return self.failure("No open pages")
        page = pages[-1]
        selector = params.get("selector", "body")
        text = await page.inner_text(selector)
        return self.success(data={"text": text[:10_000]})

    async def _click(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        pages = self._context.pages
        if not pages:
            return self.failure("No open pages")
        page = pages[-1]
        selector = params.get("selector", "")
        if not selector:
            return self.failure("Missing selector parameter")
        await page.click(selector)
        return self.success(data={"clicked": selector})

    async def _input(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        pages = self._context.pages
        if not pages:
            return self.failure("No open pages")
        page = pages[-1]
        selector = params.get("selector", "")
        text = params.get("text", "")
        if not selector:
            return self.failure("Missing selector parameter")
        await page.fill(selector, text)
        return self.success(data={"filled": selector, "text": text})

    async def _submit(self, params: dict[str, Any]) -> ToolResult:
        assert self._context is not None
        pages = self._context.pages
        if not pages:
            return self.failure("No open pages")
        page = pages[-1]
        selector = params.get("selector", "form")
        await page.press(selector, "Enter")
        return self.success(data={"submitted": selector})

    async def _close(self) -> ToolResult:
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        self._initialized = False
        return self.success(data={"message": "Browser closed"})
