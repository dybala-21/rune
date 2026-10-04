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
    from rune.cloud.boundary import hosted
    if hosted() and profile == "visible":
        profile = "managed"
    session = current_session()
    async with session.init_lock:
        if profile is not None and profile not in {"managed", "visible"}:
            raise RuntimeError("Attached Chrome requires an authenticated, explicitly selected tab")
        if session.page is not None:
            if session.profile == "native" and session.browser.is_connected():
                from rune.browser.native import select_native_page
                await select_native_page(session)
                return session.browser, session.page
            if session.page.is_closed() or not session.browser.is_connected():
                if profile is None:
                    raise RuntimeError("The browser was closed. Open it again and read fresh element references")
            else:
                # Keep the current page when switching browser entry points.
                return session.browser, session.page
            await session.release_resources()
        if profile is None:
            raise RuntimeError("No browser is open in this run. Use browser_navigate or browser_open first")
        if session.native_host:
            from rune.browser.native import connect_native
            try:
                await connect_native(session)
                return session.browser, session.page
            except BaseException:
                await session.release_resources()
                raise
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
    reload: bool = Field(default=False, description="Reload even if this exact URL is already open. Discards unsaved page state; use only when a refresh is needed.")


class BrowserOpenParams(BrowserNavigateParams):
    url: str = Field(description="URL to open in the conversation's browser")


# Capability implementations
@browser_operation
async def browser_navigate(params: BrowserNavigateParams) -> CapabilityResult:
    """Navigate in the conversation's browser, preserving an already open URL."""
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

        # Attach CDP network monitor to capture XHR/fetch API calls
        from rune.capabilities.browser.network import get_network_monitor
        monitor = get_network_monitor()
        await monitor.attach(page)

        skipped = page.url == params.url and not params.reload
        response = None
        if not skipped:
            response = await page.goto(params.url, wait_until="domcontentloaded", timeout=30_000)
            await wait_for_dom_settle(page)
        elements = await extract_interactive_elements(page)
        current_session().needs_observation = False

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        # Offer captured APIs without steering the model away from an already progressing UI task.
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

        # Return extracted refs now to avoid a redundant observation round.
        element_section = format_interactive_elements(elements)

        location = f"Already open: {url}" if skipped else f"Navigated to: {url}"
        status_line = f"\nStatus: {status}" if response else ""
        return CapabilityResult(
            success=True,
            output=(
                f"{location}\nTitle: {title}{status_line}"
                f"{api_section}{element_section}"
            ),
            metadata={
                "url": url,
                "title": title,
                "status": status,
                "skipped_navigation": skipped,
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
    """Open a browser for the user, reusing the conversation's page when possible."""
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

        skipped = page.url == params.url and not params.reload
        response = None
        if not skipped:
            response = await page.goto(params.url, wait_until="domcontentloaded", timeout=30_000)
            await wait_for_dom_settle(page)
        elements = await extract_interactive_elements(page)
        current_session().needs_observation = False

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        location = f"Already open: {url}" if skipped else f"Opened in Rune browser: {url}"
        status_line = f"\nStatus: {status}" if response else ""
        return CapabilityResult(
            success=True,
            output=(
                f"{location}\nTitle: {title}{status_line}"
                f"{format_interactive_elements(elements)}"
            ),
            metadata={
                "url": url,
                "title": title,
                "status": status,
                "profile": current_session().profile,
                "skipped_navigation": skipped,
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
