"""Browser capabilities for RUNE (Playwright-based).

Observe, act, find, extract, screenshot capabilities plus registration.
Singleton management and navigation live in ``browser_core``.
Batch/workflow/profile live in ``browser_extended``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rune.capabilities.browser.core import (
    BrowserNavigateParams,
    BrowserOpenParams,
    _accessibility_snapshot,
    _get_browser,
    browser_navigate,
    browser_open,
)
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Browser agent intelligence — loop detection + history compression
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
            f"\n\u26a0\ufe0f LOOP DETECTED: '{action} {selector}' repeated {repeats + 1} times with no effect.\n"
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
    if len(_observe_history) > 20:
        _observe_history[:] = _observe_history[-15:]
    if len(_observe_history) <= 2:
        return ""
    older = _observe_history[:-1]
    summary = " \u2192 ".join(older[-5:])
    return f"Navigation path: {summary}\n\n"


# Parameter schemas
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


class BrowserDiscoverApisParams(BaseModel):
    filter: str = Field(
        default="",
        description="URL pattern to filter (e.g., 'search', 'api/v1', 'hotel')",
    )


# Capability implementations
async def browser_observe(params: BrowserObserveParams) -> CapabilityResult:
    """Observe the current page via accessibility tree snapshot."""
    from rune.capabilities.browser.helpers import (
        extract_interactive_elements,
        wait_for_dom_settle,
    )

    log.debug("browser_observe", selector=params.selector)

    try:
        _, page = await _get_browser()

        await wait_for_dom_settle(page)

        url = page.url
        title = await page.title()

        elements = await extract_interactive_elements(page)

        snapshot = await _accessibility_snapshot(page, params.selector)

        # Filter elements by taskHint if provided.
        if params.taskHint and elements:
            hint_lower = params.taskHint.lower()
            scored = []
            for meta in elements:
                name_lower = meta.name.lower() if meta.name else ""
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
                if meta.breadcrumb:
                    parts.append(f"in({meta.breadcrumb})")
                el_lines.append(" ".join(parts))
            snapshot += "\n".join(el_lines)

        # Detect blocking overlays/dialogs.
        overlay_warning = ""
        try:
            has_overlay = await page.evaluate("""() => {
                const dialog = document.querySelector(
                    '[role="dialog"]:not([aria-hidden="true"]), [role="alertdialog"]'
                );
                if (dialog && dialog.offsetParent !== null) return 'dialog';
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
                overlay_warning = f"\n\u26a0\ufe0f BLOCKING {has_overlay.upper()} DETECTED \u2014 close it before interacting with the page."
        except Exception:
            pass

        header = f"URL: {url}\nTitle: {title}"
        if overlay_warning:
            header += overlay_warning

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
    """Perform an action on a page element."""
    from rune.capabilities.browser.helpers import (
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

        # Before snapshot
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

        # Phantom click detection
        if action == "click":
            await wait_for_dom_settle(page)
            phantom_url = page.url
            phantom_title = await page.title()
            no_change = (phantom_url == before_url and phantom_title == before_title)

            if no_change:
                log.debug("phantom_click_detected", selector=params.selector)
                click_recovered = False

                # Fallback 1: JS element.click()
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

                # Fallback 2: CDP Input.dispatchMouseEvent
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

        # After snapshot + change detection
        await wait_for_dom_settle(page)
        new_elements = await extract_interactive_elements(page)

        after_url = page.url
        after_title = await page.title()
        after_element_count = len(new_elements)

        changes: list[str] = []
        page_changed = False
        if after_url != before_url:
            changes.append(f"Navigated: {before_url} \u2192 {after_url}")
            page_changed = True
        if after_title != before_title:
            changes.append(f"Title changed: {before_title} \u2192 {after_title}")
            page_changed = True
        element_diff = after_element_count - before_element_count
        if abs(element_diff) > 2:
            changes.append(f"Elements: {before_element_count} \u2192 {after_element_count} ({'+' if element_diff > 0 else ''}{element_diff})")
        has_dialog = await page.evaluate(
            "() => !!document.querySelector('[role=\"dialog\"]:not([aria-hidden=\"true\"]), [role=\"alertdialog\"]')"
        )
        if has_dialog:
            changes.append("Dialog/overlay opened")

        parts = [f"Action: {action} on {params.selector}"]
        parts.append(f"URL: {after_url}")
        parts.append(f"Title: {after_title}")
        if changes:
            parts.append("Changes: " + "; ".join(changes))
        if page_changed:
            parts.append("Page changed \u2014 element refs may be stale, re-observe if needed.")

        _reset_action_history_on_navigation(before_url, after_url)

        if not changes and action == "click":
            parts.append(
                "\n\u26a0\ufe0f NO CHANGES DETECTED \u2014 this click likely had no effect.\n"
                "Evaluate: Did you achieve your intended goal? If not, try:\n"
                "  - A different element (re-observe the page)\n"
                "  - Scrolling to reveal the target\n"
                "  - Constructing the URL directly"
            )

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
        for el in elements[:20]:
            tag = await el.evaluate("el => el.tagName.toLowerCase()")
            text = await el.inner_text()
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


async def browser_discover_apis(params: BrowserDiscoverApisParams) -> CapabilityResult:
    """List API endpoints discovered via CDP network monitoring.

    After browser_navigate loads a page, this tool shows the XHR/fetch
    requests the site made. Use web_fetch to call these APIs directly
    instead of clicking through the UI.
    """
    from rune.capabilities.browser.network import get_network_monitor

    monitor = get_network_monitor()
    if not monitor.active:
        return CapabilityResult(
            success=True,
            output="Network monitor not active. Navigate to a page first with browser_navigate.",
            metadata={"count": 0},
        )

    apis = monitor.get_discovered_apis(params.filter)
    if not apis:
        hint = f" matching '{params.filter}'" if params.filter else ""
        return CapabilityResult(
            success=True,
            output=f"No API endpoints discovered{hint}. Try interacting with the page (scroll, click) to trigger more requests.",
            metadata={"count": 0},
        )

    lines = [f"Discovered {len(apis)} API endpoint(s):"]
    for api in apis:
        json_tag = " [JSON]" if api.has_json_response else ""
        lines.append(f"  {api.method} {api.url} [{api.status}]{json_tag}")
    lines.append("")
    lines.append("Use web_fetch(url=...) to call these APIs directly.")

    return CapabilityResult(
        success=True,
        output="\n".join(lines),
        metadata={"count": len(apis)},
    )

# Registration
def register_browser_capabilities(registry: CapabilityRegistry) -> None:
    """Register all browser capabilities."""
    from rune.capabilities.browser.extended import (
        BrowserBatchParams,
        BrowserProfileParams,
        BrowserWorkflowParams,
        browser_batch,
        browser_profile,
        browser_workflow,
    )

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
        name="browser_discover_apis",
        description=(
            "List API endpoints discovered by the network monitor. "
            "After browser_navigate, SPA sites make XHR/fetch calls — "
            "this tool shows those endpoints so you can call them "
            "directly with web_fetch instead of clicking through the UI."
        ),
        domain=Domain.BROWSER,
        risk_level=RiskLevel.LOW,
        group="browser",
        parameters_model=BrowserDiscoverApisParams,
        execute=browser_discover_apis,
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
