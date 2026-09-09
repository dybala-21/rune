"""Launch and navigate browsers owned by the current run."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from rune.capabilities.browser.session import (
    browser_operation,
    close_browser_sessions,
    current_session,
)
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

async def _get_browser(profile: str | None = None) -> tuple[Any, Any]:
    session = current_session()
    async with session.init_lock:
        if profile is not None and profile not in {"managed", "visible"}:
            raise RuntimeError("Attached Chrome requires an authenticated, explicitly selected tab")
        if session.page is not None:
            if session.page.is_closed() or not session.browser.is_connected():
                if profile is None:
                    raise RuntimeError("The browser was closed. Open it again and read fresh element references")
            elif profile is None or profile == session.profile:
                return session.browser, session.page
            await session.release_resources()
        if profile is None:
            raise RuntimeError("No browser is open in this run. Use browser_navigate or browser_open first")
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError("Install rune-ai[browser] and run playwright install chromium") from None
        from rune.config.loader import get_config
        config = get_config().browser
        session.profile = profile
        try:
            session.playwright = await async_playwright().start()
            session.browser = await session.playwright.chromium.launch(
                headless=profile == "managed", chromium_sandbox=True,
            )
            context = await session.browser.new_context(
                viewport={"width": config.viewport_width, "height": config.viewport_height},
            )
            session.page = await context.new_page()
            return session.browser, session.page
        except BaseException:
            await session.release_resources()
            raise


def _current_page_hint() -> str:
    """Return a hint about the currently open page for error messages."""
    page = current_session().page
    if page is not None:
        try:
            url = page.url
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
    """Release managed browser resources when the daemon shuts down."""
    await close_browser_sessions()


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
@browser_operation
async def browser_navigate(params: BrowserNavigateParams) -> CapabilityResult:
    """Navigate to a URL in a headless background browser for data extraction."""
    from rune.capabilities.browser.helpers import (
        extract_interactive_elements,
        format_interactive_elements,
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
        elements = await extract_interactive_elements(page)

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        # Note the data APIs this page used, without telling the model to
        # abandon what it is doing: a paired A/B measured the directive
        # version steering runs off a browsing path that converges and into
        # API exploration that does not (0/3 vs 2/3). State the option, let
        # the model choose. Gated with the rest of the hybrid path.
        from rune.capabilities.browser.network import (
            format_api_recipe,
            hybrid_api_enabled,
        )

        json_apis = monitor.get_json_apis() if hybrid_api_enabled() else []
        api_section = ""
        if json_apis:
            api_lines = ["\nData APIs this page called (readable with web_fetch):"]
            for api in json_apis[:5]:
                api_lines.append(f"  {format_api_recipe(api)}")
            api_section = "\n".join(api_lines)
            monitor.mark_reported()

        # The refs are already extracted, so hand them over: an observe round
        # that only re-reads what navigate just computed costs a model
        # round-trip and buys nothing.
        element_section = format_interactive_elements(elements)

        return CapabilityResult(
            success=True,
            output=(
                f"Navigated to: {url}\nTitle: {title}\nStatus: {status}"
                f"{api_section}{element_section}"
            ),
            metadata={
                "url": url,
                "title": title,
                "status": status,
                "interactive_count": len(elements),
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


@browser_operation
async def browser_open(params: BrowserOpenParams) -> CapabilityResult:
    """Open a URL in a visible browser the user can see and interact with."""
    from rune.capabilities.browser.helpers import (
        extract_interactive_elements,
        format_interactive_elements,
        wait_for_dom_settle,
    )

    log.debug("browser_open", url=params.url)

    url_err = _validate_browser_url(params.url)
    if url_err:
        return CapabilityResult(success=False, error=url_err)

    try:
        _, page = await _get_browser("visible")

        # Preserve in-page work only when this exact URL is already open.
        if page.url == params.url:
            title = await page.title()
            url = page.url
            log.debug("browser_open_same_url", current=url, requested=params.url)
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
        elements = await extract_interactive_elements(page)

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        return CapabilityResult(
            success=True,
            output=(
                f"Opened in visible browser: {url}\nTitle: {title}\nStatus: {status}"
                f"{format_interactive_elements(elements)}"
            ),
            metadata={
                "url": url,
                "title": title,
                "status": status,
                "profile": current_session().profile,
                "interactive_count": len(elements),
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
