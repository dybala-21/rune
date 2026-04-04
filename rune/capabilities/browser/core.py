"""Browser core — singleton management, navigation, and open.

Split from browser.py to keep files under 800 lines. This module owns
the Playwright browser/page lifecycle and the two navigation capabilities
(``browser_navigate`` for headless, ``browser_open`` for visible).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

# URL validation
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _validate_browser_url(url: str) -> str | None:
    """Return an error message if *url* should not be navigated to, else None."""
    if not url or not url.strip():
        return "Empty URL"
    try:
        parsed = urlparse(url)
    except Exception:
        return f"Malformed URL: {url}"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"Blocked URL scheme '{parsed.scheme}' — only http/https allowed"
    return None

# Browser singleton management
_browser_instance: Any = None
_page_instance: Any = None
_pw_instance: Any = None
_active_profile: str = "managed"
_lock = asyncio.Lock()


async def _get_browser(profile: str | None = None) -> tuple[Any, Any]:
    """Get or create a singleton Playwright browser and page.

    Args:
        profile: ``"managed"`` (headless) or ``"relay"`` (user's Chrome
                 via CDP, fallback to headed Playwright).  If *None*,
                 reuses the current profile.

    Returns (browser, page) tuple.
    """
    global _browser_instance, _page_instance, _pw_instance, _active_profile

    async with _lock:
        # Profile switch - close existing browser.
        if profile and profile != _active_profile and _browser_instance is not None:
            log.info("browser_profile_switch", old=_active_profile, new=profile)
            with contextlib.suppress(Exception):
                await _browser_instance.close()
            _browser_instance = None
            _page_instance = None
            _active_profile = profile

        if profile:
            _active_profile = profile

        if _page_instance is not None:
            try:
                await _page_instance.evaluate("1")
                return _browser_instance, _page_instance
            except Exception:
                _page_instance = None
                _browser_instance = None

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Browser tools require it.\n"
                "Install with: pip install rune-ai[browser]\n"
                "Then run: playwright install chromium"
            ) from None

        try:
            if _pw_instance is None:
                _pw_instance = await async_playwright().start()

            from rune.config.loader import get_config
            browser_cfg = get_config().browser

            if _active_profile == "relay":
                # Try CDP relay (user's Chrome via Extension).
                _browser_instance, _page_instance = await _try_relay_connect(
                    _pw_instance, browser_cfg,
                )
                if _browser_instance is not None:
                    return _browser_instance, _page_instance
                # Fallback: launch headed Playwright so user can see.
                log.info("relay_unavailable_falling_back_to_headed")

            # Managed (headless) or relay fallback (headed).
            headless = _active_profile == "managed"
            _browser_instance = await _pw_instance.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )

            context = await _browser_instance.new_context(
                viewport={
                    "width": browser_cfg.viewport_width,
                    "height": browser_cfg.viewport_height,
                },
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            _page_instance = await context.new_page()

            return _browser_instance, _page_instance

        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            ) from None


async def _try_relay_connect(pw: Any, browser_cfg: Any) -> tuple[Any, Any] | tuple[None, None]:
    """Try to connect to the user's Chrome via the CDP relay server."""
    try:
        import httpx

        from rune.browser.relay_server import DISCOVERY_PORT_END, DISCOVERY_PORT_START

        for port in range(DISCOVERY_PORT_START, DISCOVERY_PORT_END + 1):
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
                if resp.status_code == 200 and resp.json().get("extensionConnected"):
                    cdp_url = f"http://127.0.0.1:{port}/cdp"
                    browser = await pw.chromium.connect_over_cdp(cdp_url)
                    contexts = browser.contexts
                    if contexts:
                        pages = contexts[0].pages
                        if pages:
                            page = pages[-1]
                            log.info("relay_connected", port=port)
                            return browser, page
                    page = await browser.new_page()
                    log.info("relay_connected_new_page", port=port)
                    return browser, page
            except Exception:
                continue
    except Exception as exc:
        log.debug("relay_connect_failed", error=str(exc))
    return None, None


def _is_same_domain(current_url: str, target_url: str) -> bool:
    """Check if two URLs share the same effective domain."""
    try:
        current = urlparse(current_url).netloc.replace("www.", "")
        target = urlparse(target_url).netloc.replace("www.", "")
        if not current or not target:
            return False
        return current == target or current.endswith("." + target) or target.endswith("." + current)
    except Exception:
        return False


def _current_page_hint() -> str:
    """Return a hint about the currently open page for error messages."""
    if _page_instance is not None:
        try:
            url = _page_instance.url
            if url and url != "about:blank":
                return (
                    f"\n\U0001f4a1 Browser is currently on: {url}\n"
                    f"The name might be a menu item on this site — "
                    f"try browser_find to search the current page."
                )
        except Exception:
            pass
    return ""


async def _close_browser() -> None:
    """Close the singleton browser instance."""
    global _browser_instance, _page_instance, _pw_instance

    async with _lock:
        # Detach network monitor before closing browser (#P2)
        with contextlib.suppress(Exception):
            from rune.capabilities.browser.network import get_network_monitor
            await get_network_monitor().detach()

        if _browser_instance is not None:
            with contextlib.suppress(Exception):
                await _browser_instance.close()
            _browser_instance = None
            _page_instance = None

        if _pw_instance is not None:
            with contextlib.suppress(Exception):
                await _pw_instance.stop()
            _pw_instance = None


# Accessibility snapshot (shared utility)
_SNAPSHOT_MAX_CHARS = 6_000  # ~1.5K tokens


async def _accessibility_snapshot(page: Any, selector: str = "") -> str:
    """Generate a compact accessibility tree snapshot from the page."""
    try:
        if selector:
            # Short timeout for selector — fall back to :root if not found
            root = page.locator(selector)
            try:
                await root.wait_for(timeout=3_000)
            except Exception:
                log.debug("a11y_selector_not_found", selector=selector)
                root = page.locator(":root")
        else:
            root = page.locator(":root")
        result = await root.aria_snapshot()
        if not result:
            return "(empty page)"
        if len(result) > _SNAPSHOT_MAX_CHARS:
            result = result[:_SNAPSHOT_MAX_CHARS] + f"\n... (truncated, {len(result)} total chars)"
        return result

    except Exception as exc:
        log.warning("a11y_snapshot_failed", error=str(exc))
        try:
            text = await page.inner_text("body")
            return text[:_SNAPSHOT_MAX_CHARS]
        except Exception:
            return "(unable to read page content)"


# Parameter schemas
class BrowserNavigateParams(BaseModel):
    url: str = Field(description="URL to navigate to")


class BrowserOpenParams(BaseModel):
    url: str = Field(description="URL to open in visible browser")


# Capability implementations
async def browser_navigate(params: BrowserNavigateParams) -> CapabilityResult:
    """Navigate to a URL in a headless background browser for data extraction."""
    from rune.capabilities.browser.helpers import (
        dismiss_blocking_overlays,
        extract_interactive_elements,
        wait_for_dom_settle,
    )

    log.debug("browser_navigate", url=params.url)

    url_err = _validate_browser_url(params.url)
    if url_err:
        return CapabilityResult(success=False, error=url_err)

    try:
        _, page = await _get_browser("managed")

        # Attach CDP network monitor to capture XHR/fetch API calls (#P2)
        from rune.capabilities.browser.network import get_network_monitor
        monitor = get_network_monitor()
        await monitor.attach(page)

        response = await page.goto(params.url, wait_until="domcontentloaded", timeout=30_000)

        await wait_for_dom_settle(page)
        await dismiss_blocking_overlays(page)
        await extract_interactive_elements(page)

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        # Inline discovered JSON APIs — directive tone so weak models switch strategy
        json_apis = monitor.get_json_apis()
        api_section = ""
        if json_apis:
            api_lines = [
                "\n\u26a0\ufe0f SPA DETECTED \u2014 this site loads data via API. "
                "STOP using browser_act. Call web_fetch on these URLs instead:"
            ]
            for api in json_apis[:10]:
                api_lines.append(f"  web_fetch(url=\"{api.url}\")")
            api_section = "\n".join(api_lines)

        return CapabilityResult(
            success=True,
            output=f"Navigated to: {url}\nTitle: {title}\nStatus: {status}{api_section}",
            metadata={
                "url": url,
                "title": title,
                "status": status,
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        hint = _current_page_hint()
        return CapabilityResult(
            success=False,
            error=f"Navigation failed: {exc}{hint}",
        )


async def browser_open(params: BrowserOpenParams) -> CapabilityResult:
    """Open a URL in a visible browser the user can see and interact with."""
    from rune.capabilities.browser.helpers import (
        dismiss_blocking_overlays,
        extract_interactive_elements,
        wait_for_dom_settle,
    )

    log.debug("browser_open", url=params.url)

    url_err = _validate_browser_url(params.url)
    if url_err:
        return CapabilityResult(success=False, error=url_err)

    try:
        _, page = await _get_browser("relay")

        # Skip navigation if already on the same domain.
        if _is_same_domain(page.url, params.url):
            title = await page.title()
            url = page.url
            log.debug("browser_open_same_domain_skipped", current=url, requested=params.url)
            return CapabilityResult(
                success=True,
                output=(
                    f"Browser already open on this site: {url}\n"
                    f"Title: {title}\n"
                    f"Use browser_observe/browser_act to interact with the current page."
                ),
                metadata={"url": url, "title": title, "skipped_navigation": True},
            )

        response = await page.goto(params.url, wait_until="domcontentloaded", timeout=30_000)

        await wait_for_dom_settle(page)
        await dismiss_blocking_overlays(page)
        await extract_interactive_elements(page)

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        return CapabilityResult(
            success=True,
            output=f"Opened in visible browser: {url}\nTitle: {title}\nStatus: {status}",
            metadata={
                "url": url,
                "title": title,
                "status": status,
                "profile": _active_profile,
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        hint = _current_page_hint()
        return CapabilityResult(
            success=False,
            error=f"Browser open failed: {exc}{hint}",
        )
