"""Browser capabilities for RUNE (Playwright-based).

Observe, act, find, extract, screenshot capabilities plus registration.
Navigation and run-owned resources live in ``core`` and ``session``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rune.capabilities.browser.core import (
    BrowserNavigateParams,
    BrowserOpenParams,
    _accessibility_snapshot,
    _get_browser,
    browser_navigate,
    browser_open,
)
from rune.capabilities.browser.discover import (
    BrowserDiscoverApisParams,
    browser_discover_apis,
)
from rune.capabilities.browser.session import browser_operation, current_session
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _compress_observe_history(current_title: str, current_count: int) -> str:
    """Compress older observe results into a 1-line summary prefix."""
    _observe_history = current_session().observe_history
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
    action: Literal["click", "type", "scroll", "select", "check", "uncheck"] = Field(description="Click, fill, scroll, select, or set a checkbox state")
    selector: str = Field(description="Element ref from the current observation, or a CSS selector matching exactly one element")
    value: str = Field(default="", description="Value for type/select actions")


class BrowserScreenshotParams(BaseModel):
    path: str = Field(default="", description="Optional output file path; otherwise saved as a unique session screenshot")
    full_page: bool = Field(default=False, alias="fullPage")


class BrowserExtractParams(BaseModel):
    selector: str = Field(description="CSS selector for elements to extract")
    attribute: str = Field(
        default="",
        description="Attribute to extract (empty for text content)",
    )


class BrowserFindParams(BaseModel):
    text: str = Field(description="Text to search for on the page")


# Capability implementations
@browser_operation
async def browser_observe(params: BrowserObserveParams) -> CapabilityResult:
    """Observe the current page via accessibility tree snapshot."""
    from rune.capabilities.browser.helpers import (
        extract_interactive_elements,
        format_interactive_elements,
        wait_for_dom_settle,
    )

    log.debug("browser_observe", selector=params.selector)

    try:
        _, page = await _get_browser()

        await wait_for_dom_settle(page)

        url = page.url
        title = await page.title()

        elements = await extract_interactive_elements(page)
        current_session().needs_observation = False

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

        snapshot += format_interactive_elements(elements)

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
                overlay_warning = f"\nA {has_overlay} may block the page. Inspect its purpose before interacting."
        except Exception as exc:
            log.debug("browser_overlay_observation_failed", error=str(exc))

        header = f"URL: {url}\nTitle: {title}"
        if overlay_warning:
            header += overlay_warning

        # Surface data APIs captured since the model last saw a report — the
        # schedule/booking XHRs fire on interaction, and replaying them beats
        # clicking on (hybrid API agents: arXiv:2410.16464).
        from rune.capabilities.browser.network import get_network_monitor, hybrid_api_enabled
        _new_apis = get_network_monitor().unreported_interesting_count()
        if hybrid_api_enabled() and _new_apis > 0:
            header += (
                f"\n\U0001f4e1 {_new_apis} data API call(s) captured \u2014 "
                "browser_discover_apis(readBody='<url part>') reads the "
                "fetched JSON directly."
            )

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


@browser_operation
async def browser_act(params: BrowserActParams) -> CapabilityResult:
    """Dispatch once against the observed node and report the resulting state."""
    from rune.capabilities.browser.helpers import (
        extract_interactive_elements,
        find_element_locator,
        format_interactive_elements,
        is_element_ref,
        wait_for_dom_settle,
    )

    session = current_session()
    owned_target = None
    if session.needs_observation:
        return CapabilityResult(success=False, error="Read the current page with browser_observe after user control changed.",
                                metadata={"action_status": "not_executed"})
    if session.uncertain_action:
        return CapabilityResult(success=False, error=(
            "A previous browser action has an unknown outcome. Inspect the page or ask the user "
            "to confirm its effects; further changes are paused for this run."
        ), metadata={"action_status": "not_executed"})
    try:
        _, page = await _get_browser()
        action = params.action
        target = None
        if action != "scroll":
            if is_element_ref(params.selector):
                target = await find_element_locator(page, params.selector)
                if target is None:
                    return CapabilityResult(success=False, error=(
                        f"Reference {params.selector} is stale or ambiguous. Read the page again "
                        "and choose an unambiguous target. No action was dispatched."
                    ), metadata={"action_status": "not_executed"})
            else:
                locator = page.locator(params.selector)
                count = await locator.count()
                if count != 1:
                    return CapabilityResult(success=False, error=(
                        f"Selector matched {count} elements; exactly one is required. "
                        "No action was dispatched."
                    ), metadata={"action_status": "not_executed"})
                target = await locator.element_handle(timeout=1000)
                owned_target = target
                if target is None:
                    raise RuntimeError("The target disappeared before the action")

        before_url = page.url
        before_snapshot = await _accessibility_snapshot(page)
        before_state = await _control_state(target) if target is not None else {}
        try:
            if action == "click":
                await target.click(timeout=10_000)
            elif action == "type":
                await target.fill(params.value, timeout=10_000)
            elif action == "select":
                await target.select_option(params.value, timeout=10_000)
            elif action in {"check", "uncheck"}:
                await target.set_checked(action == "check", timeout=10_000)
            else:
                if params.value not in {"", "up", "down"}:
                    return CapabilityResult(success=False, error="Scroll value must be up or down",
                                            metadata={"action_status": "not_executed"})
                await page.evaluate("dy => window.scrollBy(0, dy)", -500 if params.value == "up" else 500)
        except BaseException as exc:
            session.uncertain_action = True
            if not isinstance(exc, Exception):
                raise
            return CapabilityResult(success=False, error=(
                f"Action outcome is unknown: {exc}. The command was attempted once and was not retried. "
                "Inspect the current state before deciding what to do next."
            ), metadata={"action_status": "unknown", "action": action, "selector": params.selector})

        try:
            await wait_for_dom_settle(page)
            after_state = await _control_state(target) if target is not None else {}
            after_snapshot = await _accessibility_snapshot(page)
            elements = await extract_interactive_elements(page)
            changed = page.url != before_url or after_snapshot != before_snapshot or before_state != after_state
            summary = "Observed a state change." if changed else (
                "No visible state change was observed. Verify the intended result before another action."
            )
            api_sections: list[str] = []
            from rune.capabilities.browser.network import get_network_monitor, hybrid_api_enabled
            _monitor = get_network_monitor()
            if hybrid_api_enabled():
                _bodies = _monitor.unreported_json_bodies()
                if _bodies:
                    _api = _bodies[-1]
                    _body = _api.response_body[:6000]
                    _cut = (
                        "\n[response truncated — browser_discover_apis(readBody=…, "
                        "jsonFilter='<keyword>') returns the matching records]"
                        if len(_api.response_body) > len(_body) else ""
                    )
                    api_sections.append(
                        f"\nThis interaction loaded {_api.method} {_api.url[:120]} — "
                        f"its JSON response follows, so the answer may already be "
                        f"here:\n{_body}{_cut}"
                    )
                    _monitor.mark_reported()
                else:
                    _new_apis = _monitor.unreported_interesting_count()
                    if _new_apis > 0:
                        api_sections.append(
                            f"\U0001f4e1 {_new_apis} new data API call(s) captured "
                            "by this interaction — browser_discover_apis lists them."
                        )
            api_text = "\n".join(api_sections)
            return CapabilityResult(success=True, output=(
                f"Action dispatched: {action} on {params.selector}\nURL: {page.url}\n{summary}"
                f"\n{after_snapshot}{format_interactive_elements(elements)}{api_text}"
            ), metadata={"action_status": "dispatched", "action": action, "selector": params.selector,
                         "url": page.url, "page_changed": changed, "control_state": after_state,
                         "elements_refreshed": len(elements)})
        except Exception as exc:
            log.debug("browser_post_action_observation_failed", error=str(exc))
            return CapabilityResult(success=True, output=(
                f"Action dispatched: {action}. Reading the resulting page failed: {exc}. "
                "Observe the page again; do not repeat the action to obtain its result."
            ), metadata={"action_status": "dispatched", "observation_failed": True})
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Action was not dispatched: {exc}",
                                metadata={"action_status": "not_executed"})
    finally:
        if owned_target is not None:
            try:
                await owned_target.dispose()
            except Exception as exc:
                log.debug("browser_target_release_failed", error=str(exc))


async def _control_state(target: Any) -> dict[str, Any]:
    try:
        return await target.evaluate("""el => ({
            connected: el.isConnected, checked: el.checked, value: el.value,
            selected: el.selectedIndex, expanded: el.getAttribute('aria-expanded'),
            pressed: el.getAttribute('aria-pressed'), disabled: el.disabled
        })""")
    except Exception as exc:
        log.debug("browser_control_state_unavailable", error=str(exc))
        return {}


@browser_operation
async def browser_screenshot(params: BrowserScreenshotParams) -> CapabilityResult:
    """Take a screenshot of the current page."""
    log.debug("browser_screenshot", path=params.path, full_page=params.full_page)

    try:
        _, page = await _get_browser()
        from pathlib import Path
        from uuid import uuid4

        from rune.safety.guardian import get_guardian
        from rune.utils.paths import rune_data

        if params.path:
            from rune.agent.isolation import enforce
            if denied := enforce(params.path):
                return CapabilityResult(success=False, error=denied)
            validation = get_guardian().validate_file_path(params.path)
            if not validation.allowed:
                return CapabilityResult(success=False, error=validation.reason)
            output_path = Path(params.path).expanduser().resolve()
        else:
            output_path = rune_data() / "screenshots" / current_session().id / f"{uuid4().hex}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        await page.screenshot(path=str(output_path), full_page=params.full_page)
        if not params.path:
            output_path.chmod(0o600)

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


@browser_operation
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


@browser_operation
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
            "Navigate to a URL in a headless background browser and return a "
            "compact snapshot: page title, status, and the interactive elements "
            "with their ref IDs — there is no need to call browser_observe right "
            "after navigating. "
            "Reach for the browser only when a page must be INTERACTED with "
            "(clicking, filling forms, content that appears after a click). For "
            "plain retrieval prefer web_search or web_fetch, and for a documented "
            "API endpoint or a .json/.txt/.md URL call web_fetch directly — the "
            "browser is far slower for those. "
            "The user CANNOT see this browser. Use browser_open instead when the "
            "user wants to see, watch, or interact with the browser."
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
            "'launch', 'pull up'. Opens a Rune-managed browser with "
            "its own profile; existing Chrome tabs are not attached."
        ),
        domain=Domain.BROWSER,
        risk_level=RiskLevel.MEDIUM,
        group="browser",
        parameters_model=BrowserOpenParams,
        execute=browser_open,
    ))
    registry.register(CapabilityDefinition(
        name="browser_observe",
        description=(
            "Re-read the current page: accessibility tree plus interactive "
            "elements with ref IDs. Use it after an interaction changes the page, "
            "since browser_navigate already returns a snapshot of its own."
        ),
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
