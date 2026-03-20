"""Browser capabilities for RUNE (Playwright-based).

Ported from src/capabilities/browser.ts - navigate, observe, act,
screenshot, extract, and find using a headless browser instance.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Allowed URL schemes for browser navigation.
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


# Browser agent intelligence - loop detection + history compression

_action_history: list[tuple[str, str]] = []  # (action, selector) per session
_observe_history: list[str] = []  # ["Title (N elements)"] per session
_LOOP_THRESHOLD = 2  # same action+selector N times -> loop warning


def _reset_action_history_on_navigation(before_url: str, after_url: str) -> None:
    """Clear action history when the page URL changes (refs are reassigned)."""
    if before_url != after_url:
        _action_history.clear()


def _detect_action_loop(action: str, selector: str) -> str:
    """Check if the same action+selector has been repeated recently."""
    current = (action, selector)
    recent = _action_history[-4:]  # check last 4 actions
    repeats = sum(1 for h in recent if h == current)
    _action_history.append(current)
    # Keep history bounded
    if len(_action_history) > 50:
        _action_history[:] = _action_history[-30:]
    if repeats >= _LOOP_THRESHOLD:
        return (
            f"\n⚠️ LOOP DETECTED: '{action} {selector}' repeated {repeats + 1} times with no effect.\n"
            "Try a DIFFERENT approach:\n"
            "  1. browser_observe to find alternative elements\n"
            "  2. Scroll down/up to reveal hidden elements\n"
            "  3. Construct the target URL directly (browser_navigate)\n"
            "  4. Use browser_find to locate by text\n"
            "Do NOT repeat the same action."
        )
    return ""


def _compress_observe_history(current_title: str, current_count: int) -> str:
    """Compress older observe results into a 1-line summary prefix."""
    _observe_history.append(f"{current_title} ({current_count} elements)")
    # Keep bounded
    if len(_observe_history) > 20:
        _observe_history[:] = _observe_history[-15:]
    if len(_observe_history) <= 2:
        return ""
    # Summarize all but the current entry
    older = _observe_history[:-1]
    summary = " → ".join(older[-5:])  # last 5 pages
    return f"Navigation path: {summary}\n\n"


# Parameter schemas

class BrowserNavigateParams(BaseModel):
    url: str = Field(description="URL to navigate to")


class BrowserOpenParams(BaseModel):
    url: str = Field(description="URL to open in visible browser")


class BrowserObserveParams(BaseModel):
    selector: str = Field(
        default="",
        description="CSS selector to focus observation (empty for full page)",
    )
    taskHint: str = Field(
        default="",
        description="Task hint to filter relevant elements (e.g., 'hotel list', 'login form')",
    )


class BrowserActParams(BaseModel):
    action: str = Field(description="Action: click, type, scroll, select")
    selector: str = Field(description="CSS selector for target element")
    value: str = Field(default="", description="Value for type/select actions")


class BrowserScreenshotParams(BaseModel):
    path: str = Field(default="screenshot.png", description="Output file path")
    full_page: bool = Field(default=False, alias="fullPage")


class BrowserExtractParams(BaseModel):
    selector: str = Field(description="CSS selector for elements to extract")
    attribute: str = Field(
        default="",
        description="Attribute to extract (empty for text content)",
    )


class BrowserFindParams(BaseModel):
    text: str = Field(description="Text to search for on the page")


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
    from urllib.parse import urlparse
    try:
        current = urlparse(current_url).netloc.replace("www.", "")
        target = urlparse(target_url).netloc.replace("www.", "")
        if not current or not target:
            return False
        # Handle subdomains: nol.yanolja.com and www.yanolja.com -> same domain
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
                    f"\n💡 Browser is currently on: {url}\n"
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
        if _browser_instance is not None:
            with contextlib.suppress(Exception):
                await _browser_instance.close()
            _browser_instance = None
            _page_instance = None

        if _pw_instance is not None:
            with contextlib.suppress(Exception):
                await _pw_instance.stop()
            _pw_instance = None


# Accessibility snapshot

_SNAPSHOT_MAX_CHARS = 15_000  # ~4K tokens - enough for page structure


async def _accessibility_snapshot(page: Any, selector: str = "") -> str:
    """Generate a compact accessibility tree snapshot from the page.

    Uses Playwright 1.58+ ``Locator.aria_snapshot()``.  The raw YAML
    is truncated to ~4K tokens to avoid context bloat. The interactive
    element list (appended by ``browser_observe``) is the primary
    source of actionable information for the LLM.
    """
    try:
        root = page.locator(selector) if selector else page.locator(":root")
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


# Capability implementations

async def browser_navigate(params: BrowserNavigateParams) -> CapabilityResult:
    """Navigate to a URL in a headless background browser for data extraction.

    The user cannot see this browser. For visible browser interaction,
    use ``browser_open`` instead.
    """
    from rune.capabilities.browser_helpers import (
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
        response = await page.goto(params.url, wait_until="domcontentloaded", timeout=30_000)

        # Wait for SPA hydration / dynamic content.
        await wait_for_dom_settle(page)
        # Dismiss cookie banners / overlays that appear on first load.
        await dismiss_blocking_overlays(page)
        # Pre-populate element store for subsequent act/observe calls.
        await extract_interactive_elements(page)

        status = response.status if response else 0
        title = await page.title()
        url = page.url

        return CapabilityResult(
            success=True,
            output=f"Navigated to: {url}\nTitle: {title}\nStatus: {status}",
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


async def browser_observe(params: BrowserObserveParams) -> CapabilityResult:
    """Observe the current page via accessibility tree snapshot."""
    from rune.capabilities.browser_helpers import (
        extract_interactive_elements,
        wait_for_dom_settle,
    )

    log.debug("browser_observe", selector=params.selector)

    try:
        _, page = await _get_browser()

        # Ensure DOM is stable before observation.
        await wait_for_dom_settle(page)

        url = page.url
        title = await page.title()

        # Re-extract interactive elements so the store is current.
        elements = await extract_interactive_elements(page)

        snapshot = await _accessibility_snapshot(page, params.selector)

        # Filter elements by taskHint if provided.
        if params.taskHint and elements:
            hint_lower = params.taskHint.lower()
            scored = []
            for meta in elements:
                name_lower = meta.name.lower() if meta.name else ""
                # Simple relevance: hint words appear in element name/role.
                score = sum(1 for word in hint_lower.split() if word in name_lower or word in meta.role)
                scored.append((score, meta))
            scored.sort(key=lambda x: x[0], reverse=True)
            elements = [meta for _, meta in scored]

        # Append interactive element summary (capped at 50).
        max_elements = 50
        if elements:
            shown = elements[:max_elements]
            el_lines = [f"\n--- Interactive Elements ({len(shown)}/{len(elements)}) ---"]
            for meta in shown:
                parts = [f"[{meta.ref}]", meta.role]
                if meta.name:
                    parts.append(f'"{meta.name}"')
                el_lines.append(" ".join(parts))
            snapshot += "\n".join(el_lines)

        # Detect blocking overlays/dialogs.
        overlay_warning = ""
        try:
            has_overlay = await page.evaluate("""() => {
                // Check role=dialog
                const dialog = document.querySelector(
                    '[role="dialog"]:not([aria-hidden="true"]), [role="alertdialog"]'
                );
                if (dialog && dialog.offsetParent !== null) return 'dialog';
                // Check high z-index fixed overlays
                for (const el of document.querySelectorAll('div, section, aside')) {
                    const s = getComputedStyle(el);
                    const z = parseInt(s.zIndex);
                    if (z > 1000 && (s.position === 'fixed' || s.position === 'absolute')) {
                        const r = el.getBoundingClientRect();
                        if (r.width > window.innerWidth * 0.4 && r.height > window.innerHeight * 0.25) {
                            return 'overlay';
                        }
                    }
                }
                return null;
            }""")
            if has_overlay:
                overlay_warning = f"\n⚠️ BLOCKING {has_overlay.upper()} DETECTED — close it before interacting with the page."
        except Exception:
            pass

        header = f"URL: {url}\nTitle: {title}"
        if overlay_warning:
            header += overlay_warning

        # Compress older observe history to save tokens
        history_prefix = _compress_observe_history(title, len(elements))

        return CapabilityResult(
            success=True,
            output=f"{history_prefix}{header}\n\n{snapshot}",
            metadata={
                "url": url,
                "title": title,
                "selector": params.selector,
                "interactive_count": len(elements),
                "has_overlay": bool(overlay_warning),
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Observation failed: {exc}",
        )


async def browser_act(params: BrowserActParams) -> CapabilityResult:
    """Perform an action on a page element.

    Supports two selector modes:
    - Element ref (e.g. ``"e5"``) - resolved via self-healing multi-selector.
    - CSS selector (e.g. ``"#submit"``) - direct Playwright query_selector.
    """
    from rune.capabilities.browser_helpers import (
        MAX_RETRIES,
        RETRY_DELAY_MS,
        dismiss_blocking_overlays,
        extract_interactive_elements,
        get_element_store,
        self_healing_find,
        wait_for_dom_settle,
    )

    log.debug("browser_act", action=params.action, selector=params.selector)

    try:
        _, page = await _get_browser()

        action = params.action.lower()

        # Scroll doesn't need an element.
        if action == "scroll":
            direction = params.value.lower() if params.value else "down"
            delta = 500 if direction == "down" else -500
            await page.evaluate(f"window.scrollBy(0, {delta})")
            await wait_for_dom_settle(page)
            return CapabilityResult(
                success=True,
                output=f"Scrolled {direction}",
                metadata={"action": "scroll", "url": page.url},
            )

        # --- Before snapshot ---
        before_url = page.url
        before_title = await page.title()
        before_element_count = len(get_element_store().all)

        # Resolve element - try ref-based self-healing first, fall back to CSS.
        locator = None
        if params.selector.startswith("e") and params.selector[1:].isdigit():
            locator = await self_healing_find(page, params.selector)

        if locator is None:
            element = await page.query_selector(params.selector)
            if element is None:
                return CapabilityResult(
                    success=False,
                    error=f"Element not found: {params.selector}",
                )
            locator = element

        # Execute action with reactive overlay recovery.
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                if action == "click":
                    await locator.click(timeout=10_000)
                    output = f"Clicked: {params.selector}"
                elif action == "type":
                    await locator.fill(params.value)
                    output = f"Typed into {params.selector}: {params.value[:50]}"
                elif action == "select":
                    await locator.select_option(params.value)
                    output = f"Selected '{params.value}' in {params.selector}"
                else:
                    return CapabilityResult(
                        success=False,
                        error=f"Unknown action: {params.action}. "
                              f"Supported: click, type, scroll, select",
                    )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    log.debug("browser_act_retry", attempt=attempt + 1, error=str(exc))
                    dismissed = await dismiss_blocking_overlays(page)
                    if dismissed:
                        log.debug("browser_act_overlay_dismissed_on_retry", dismissed=dismissed)
                    await page.wait_for_timeout(RETRY_DELAY_MS)
                    if params.selector.startswith("e") and params.selector[1:].isdigit():
                        new_locator = await self_healing_find(page, params.selector)
                        if new_locator is not None:
                            locator = new_locator
                    else:
                        new_el = await page.query_selector(params.selector)
                        if new_el is not None:
                            locator = new_el

        if last_error is not None:
            return CapabilityResult(
                success=False,
                error=f"Action failed after {MAX_RETRIES + 1} attempts: {last_error}",
            )

        # --- Phantom click detection (click "succeeded" but nothing happened) ---
        if action == "click":
            await wait_for_dom_settle(page)
            phantom_url = page.url
            phantom_title = await page.title()
            no_change = (phantom_url == before_url and phantom_title == before_title)

            if no_change:
                log.debug("phantom_click_detected", selector=params.selector)
                click_recovered = False

                # Fallback 1: JS element.click() - bypasses overlay interception
                try:
                    el_handle = await page.query_selector(
                        f'[data-rune-ref="{params.selector}"]'
                    ) or await page.query_selector(params.selector)
                    if el_handle:
                        await page.evaluate("""(el) => {
                            el.scrollIntoView({block: 'center'});
                            el.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
                            el.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                            el.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                            el.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
                            el.click();
                        }""", el_handle)
                        await wait_for_dom_settle(page)
                        if page.url != before_url or await page.title() != before_title:
                            click_recovered = True
                            output += " (JS click fallback)"
                            log.debug("phantom_click_recovered_js")
                except Exception as exc:
                    log.debug("js_click_fallback_failed", error=str(exc))

                # Fallback 2: CDP Input.dispatchMouseEvent - raw mouse events
                if not click_recovered:
                    try:
                        el_handle = await page.query_selector(
                            f'[data-rune-ref="{params.selector}"]'
                        ) or await page.query_selector(params.selector)
                        if el_handle:
                            box = await el_handle.bounding_box()
                            if box:
                                x = box["x"] + box["width"] / 2
                                y = box["y"] + box["height"] / 2
                                cdp = await page.context.new_cdp_session(page)
                                for event_type in ("mousePressed", "mouseReleased"):
                                    await cdp.send("Input.dispatchMouseEvent", {
                                        "type": event_type,
                                        "x": x, "y": y,
                                        "button": "left",
                                        "clickCount": 1,
                                    })
                                await cdp.detach()
                                await wait_for_dom_settle(page)
                                if page.url != before_url or await page.title() != before_title:
                                    output += " (CDP click fallback)"
                                    log.debug("phantom_click_recovered_cdp")
                    except Exception as exc:
                        log.debug("cdp_click_fallback_failed", error=str(exc))

        # --- After snapshot + change detection ---
        await wait_for_dom_settle(page)
        new_elements = await extract_interactive_elements(page)

        after_url = page.url
        after_title = await page.title()
        after_element_count = len(new_elements)

        changes: list[str] = []
        page_changed = False
        if after_url != before_url:
            changes.append(f"Navigated: {before_url} → {after_url}")
            page_changed = True
        if after_title != before_title:
            changes.append(f"Title changed: {before_title} → {after_title}")
            page_changed = True
        element_diff = after_element_count - before_element_count
        if abs(element_diff) > 2:
            changes.append(f"Elements: {before_element_count} → {after_element_count} ({'+' if element_diff > 0 else ''}{element_diff})")
        # Detect dialog/overlay opened
        has_dialog = await page.evaluate(
            "() => !!document.querySelector('[role=\"dialog\"]:not([aria-hidden=\"true\"]), [role=\"alertdialog\"]')"
        )
        if has_dialog:
            changes.append("Dialog/overlay opened")

        # Build rich output with self-evaluation
        parts = [f"Action: {action} on {params.selector}"]
        parts.append(f"URL: {after_url}")
        parts.append(f"Title: {after_title}")
        if changes:
            parts.append("Changes: " + "; ".join(changes))
        if page_changed:
            parts.append("Page changed — element refs may be stale, re-observe if needed.")

        # Reset action history on page navigation (refs get reassigned)
        _reset_action_history_on_navigation(before_url, after_url)

        # Self-evaluation: warn if action had no observable effect
        if not changes and action == "click":
            parts.append(
                "\n⚠️ NO CHANGES DETECTED — this click likely had no effect.\n"
                "Evaluate: Did you achieve your intended goal? If not, try:\n"
                "  - A different element (re-observe the page)\n"
                "  - Scrolling to reveal the target\n"
                "  - Constructing the URL directly"
            )

        # Loop detection
        loop_warning = _detect_action_loop(action, params.selector)
        if loop_warning:
            parts.append(loop_warning)

        return CapabilityResult(
            success=True,
            output="\n".join(parts),
            metadata={
                "action": params.action,
                "selector": params.selector,
                "url": after_url,
                "page_changed": page_changed,
                "changes": changes,
                "elements_refreshed": after_element_count,
                "no_effect": not changes and action == "click",
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Action failed: {exc}",
        )


async def browser_screenshot(params: BrowserScreenshotParams) -> CapabilityResult:
    """Take a screenshot of the current page."""
    log.debug("browser_screenshot", path=params.path, full_page=params.full_page)

    try:
        _, page = await _get_browser()
        from pathlib import Path

        output_path = Path(params.path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        await page.screenshot(
            path=str(output_path),
            full_page=params.full_page,
        )

        return CapabilityResult(
            success=True,
            output=f"Screenshot saved: {output_path}",
            metadata={
                "path": str(output_path),
                "url": page.url,
                "full_page": params.full_page,
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Screenshot failed: {exc}",
        )


async def browser_extract(params: BrowserExtractParams) -> CapabilityResult:
    """Extract text or attributes from elements matching a CSS selector."""
    log.debug("browser_extract", selector=params.selector, attribute=params.attribute)

    try:
        _, page = await _get_browser()
        elements = await page.query_selector_all(params.selector)

        if not elements:
            return CapabilityResult(
                success=True,
                output=f"No elements found matching: {params.selector}",
                metadata={"count": 0},
            )

        values: list[str] = []
        for el in elements:
            if params.attribute:
                val = await el.get_attribute(params.attribute)
                if val is not None:
                    values.append(val)
            else:
                text = await el.inner_text()
                if text.strip():
                    values.append(text.strip())

        return CapabilityResult(
            success=True,
            output="\n".join(values),
            metadata={
                "selector": params.selector,
                "attribute": params.attribute,
                "count": len(values),
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Extraction failed: {exc}",
        )


async def browser_find(params: BrowserFindParams) -> CapabilityResult:
    """Find elements on the page containing the specified text."""
    log.debug("browser_find", text=params.text)

    try:
        _, page = await _get_browser()

        # Use XPath to find text nodes
        escaped = params.text.replace("'", "\\'")
        xpath = f"//*[contains(text(), '{escaped}')]"

        elements = await page.query_selector_all(f"xpath={xpath}")

        if not elements:
            return CapabilityResult(
                success=True,
                output=f"No elements found containing text: '{params.text}'",
                metadata={"count": 0},
            )

        results: list[dict] = []
        for el in elements[:20]:  # Limit results
            tag = await el.evaluate("el => el.tagName.toLowerCase()")
            text = await el.inner_text()
            # Get a usable selector
            el_id = await el.get_attribute("id")
            el_class = await el.get_attribute("class")

            selector = tag
            if el_id:
                selector = f"#{el_id}"
            elif el_class:
                first_class = el_class.strip().split()[0]
                selector = f"{tag}.{first_class}"

            results.append({
                "tag": tag,
                "selector": selector,
                "text": text[:200],
            })

        lines: list[str] = [f"Found {len(results)} element(s) with text '{params.text}':"]
        for r in results:
            lines.append(f"  <{r['tag']}> [{r['selector']}] {r['text'][:80]}")

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "count": len(results),
                "elements": results,
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Find failed: {exc}",
        )


# Batch, workflow, and profile capabilities

class BrowserBatchParams(BaseModel):
    """Parameters for batch browser operations."""
    actions: list[dict[str, Any]] = Field(description="List of browser actions to execute in sequence")


async def browser_batch(params: BrowserBatchParams) -> CapabilityResult:
    """Execute multiple browser actions in a single batch."""
    from rune.capabilities.registry import get_capability_registry

    reg = get_capability_registry()
    results: list[str] = []
    successes: list[bool] = []
    for i, action in enumerate(params.actions):
        action_type = action.get("type", "")
        try:
            result = await reg.execute(f"browser_{action_type}", action.get("params", {}))
            successes.append(result.success)
            results.append(f"Action {i+1} ({action_type}): {'OK' if result.success else result.error}")
        except Exception as exc:
            successes.append(False)
            results.append(f"Action {i+1} ({action_type}): Error — {exc}")
    return CapabilityResult(success=all(successes), output="\n".join(results))


class BrowserWorkflowParams(BaseModel):
    """Parameters for browser workflow automation."""
    name: str = Field(description="Workflow name")
    steps: list[dict[str, Any]] = Field(description="Workflow steps to execute")
    timeout: int = Field(default=30000, description="Workflow timeout in ms")


async def browser_workflow(params: BrowserWorkflowParams) -> CapabilityResult:
    """Execute a multi-step browser workflow."""
    from rune.capabilities.registry import get_capability_registry

    reg = get_capability_registry()
    results: list[str] = []
    successes: list[bool] = []
    for i, step in enumerate(params.steps):
        step_type = step.get("action", "")
        try:
            result = await reg.execute(f"browser_{step_type}", step.get("params", {}))
            successes.append(result.success)
            results.append(f"Step {i+1} ({step_type}): {'OK' if result.success else result.error}")
            if not result.success and not step.get("continue_on_error", False):
                break
        except Exception as exc:
            successes.append(False)
            results.append(f"Step {i+1} ({step_type}): Error — {exc}")
            break
    return CapabilityResult(
        success=all(successes),
        output=f"Workflow '{params.name}':\n" + "\n".join(results),
    )


class BrowserProfileParams(BaseModel):
    """Parameters for browser profile management."""
    action: str = Field(description="Profile action: create, load, delete")
    name: str = Field(description="Profile name")
    settings: dict[str, Any] = Field(default_factory=dict, description="Profile settings")


_browser_profiles: dict[str, dict] = {}


async def browser_profile(params: BrowserProfileParams) -> CapabilityResult:
    """Manage browser profiles for different contexts."""
    if params.action == "create":
        _browser_profiles[params.name] = {"settings": params.settings or {}}
        return CapabilityResult(success=True, output=f"Profile '{params.name}' created.")
    elif params.action == "delete":
        _browser_profiles.pop(params.name, None)
        return CapabilityResult(success=True, output=f"Profile '{params.name}' deleted.")
    elif params.action == "list":
        names = list(_browser_profiles.keys())
        return CapabilityResult(success=True, output=f"Profiles: {', '.join(names) or '(none)'}")
    elif params.action == "get":
        profile = _browser_profiles.get(params.name)
        if profile:
            return CapabilityResult(success=True, output=str(profile))
        return CapabilityResult(success=False, error=f"Profile '{params.name}' not found.")
    return CapabilityResult(success=False, error=f"Unknown action: {params.action}")


async def browser_open(params: BrowserOpenParams) -> CapabilityResult:
    """Open a URL in a visible browser the user can see and interact with.

    Tries the user's Chrome (via relay extension) first. If the extension
    is not connected, opens a visible Playwright browser as fallback.
    Use this when the user wants to watch, interact, log in, or says
    things like "open", "show me", "launch the browser".

    If the browser is already on the same domain, skips navigation and
    returns the current page state instead.
    """
    from rune.capabilities.browser_helpers import (
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


# Registration

def register_browser_capabilities(registry: CapabilityRegistry) -> None:
    """Register all browser capabilities."""
    registry.register(CapabilityDefinition(
        name="browser_navigate",
        description=(
            "Navigate to a URL in a headless background browser for data extraction. "
            "The user CANNOT see this browser. Use browser_open instead when the user "
            "wants to see, watch, or interact with the browser."
        ),
        domain=Domain.BROWSER,
        risk_level=RiskLevel.MEDIUM,
        group="browser",
        parameters_model=BrowserNavigateParams,
        execute=browser_navigate,
    ))
    registry.register(CapabilityDefinition(
        name="browser_open",
        description=(
            "Open a URL in a VISIBLE browser the user can see. "
            "Use when the user wants to watch the browser, interact with a site, "
            "log in, make a purchase, or says things like 'open', 'show me', "
            "'launch', 'pull up'. Tries the user's Chrome first, falls back to "
            "a visible Playwright browser."
        ),
        domain=Domain.BROWSER,
        risk_level=RiskLevel.MEDIUM,
        group="browser",
        parameters_model=BrowserOpenParams,
        execute=browser_open,
    ))
    registry.register(CapabilityDefinition(
        name="browser_observe",
        description="Observe page content via accessibility tree",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.LOW,
        group="browser",
        parameters_model=BrowserObserveParams,
        execute=browser_observe,
    ))
    registry.register(CapabilityDefinition(
        name="browser_act",
        description="Perform an action on a page element (click/type/scroll/select)",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.MEDIUM,
        group="browser",
        parameters_model=BrowserActParams,
        execute=browser_act,
    ))
    registry.register(CapabilityDefinition(
        name="browser_screenshot",
        description="Take a screenshot of the current page",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.LOW,
        group="browser",
        parameters_model=BrowserScreenshotParams,
        execute=browser_screenshot,
    ))
    registry.register(CapabilityDefinition(
        name="browser_extract",
        description="Extract text or attributes from page elements",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.LOW,
        group="browser",
        parameters_model=BrowserExtractParams,
        execute=browser_extract,
    ))
    registry.register(CapabilityDefinition(
        name="browser_find",
        description="Find elements on the page matching text",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.LOW,
        group="browser",
        parameters_model=BrowserFindParams,
        execute=browser_find,
    ))
    registry.register(CapabilityDefinition(
        name="browser_batch",
        description="Execute multiple browser actions in batch",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.MEDIUM,
        group="browser",
        parameters_model=BrowserBatchParams,
        execute=browser_batch,
    ))
    registry.register(CapabilityDefinition(
        name="browser_workflow",
        description="Execute multi-step browser workflow",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.MEDIUM,
        group="browser",
        parameters_model=BrowserWorkflowParams,
        execute=browser_workflow,
    ))
    registry.register(CapabilityDefinition(
        name="browser_profile",
        description="Manage browser profiles",
        domain=Domain.BROWSER,
        risk_level=RiskLevel.LOW,
        group="browser",
        parameters_model=BrowserProfileParams,
        execute=browser_profile,
    ))
