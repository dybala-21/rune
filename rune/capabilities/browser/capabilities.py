"""Expose browser observation and actions; core and session manage navigation and ownership."""

from __future__ import annotations

import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from rune.capabilities.browser.checks import (
    BrowserCheck,
    check_identity,
    check_page,
    control_identity,
    format_checks,
    read_controls,
    readback_resolved,
)
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
from rune.capabilities.browser.effects import begin_effect, dispatched_effect, node_key
from rune.capabilities.browser.page_queries import (
    BrowserExtractParams,
    BrowserFindParams,
    BrowserScreenshotParams,
    browser_extract,
    browser_find,
    browser_screenshot,
)
from rune.capabilities.browser.readback import ResultReadback, read_result
from rune.capabilities.browser.session import browser_operation, current_session, is_manual_input
from rune.capabilities.browser.values import equivalent
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
    resultRequirement: str = Field(default="", description="Saved outcome ID or quote for this record, either the current retrieved document or readbackUrl. Omit only for one saved outcome.")
    checks: list[BrowserCheck] = Field(default_factory=list, max_length=32, description="Read requested fields without input. Page checks return a snapshot on mismatch. Use readbackUrl for stored values; page messages do not prove storage.")
    readbackUrl: str = Field(default="", max_length=2000, description="Observed same-origin saved-record URL or link ref. Fetch once without changing the tab. If field selectors are unknown, omit checks to get fields and a readbackSnapshot.")
    readbackSnapshot: str = Field(default="", max_length=100, description="Check the last fetched record once using its snapshot and field selectors. No URL or new GET needed; include the requested field checks.")
    selector: str = Field(
        default="",
        description="CSS selector to focus observation (empty for full page)",
    )
    taskHint: str = Field(
        default="",
        description="Task hint to filter relevant elements (e.g., 'hotel list', 'login form')",
    )

    @model_validator(mode="after")
    def validate_snapshot(self):
        if self.readbackSnapshot and not self.checks:
            raise ValueError("Include field checks when reusing readbackSnapshot")
        return self


class BrowserActParams(BaseModel):
    readback: ResultReadback | None = Field(default=None, description="After this action, read the saved record once without changing the tab. Include it on the save action when an observed result URL is available.")
    validation_hint: ClassVar[str] = (
        'No input was dispatched. Use {"action":"click","selector":"CURRENT_REF"}. '
        'Optional checks need their own selector: {"selector":"RESULT_SELECTOR","property":"text",'
        '"expected":"EXACT_OBSERVED_RESULT","requirement":"USER_REQUIREMENT"}. '
        "Read the page for real targets and results; do not guess a confirmation."
    )
    checks: list[BrowserCheck] = Field(default_factory=list, max_length=32, description="Expected results to read after this action; use on the final change to verify completion in the same call.")
    action: Literal["click", "type", "scroll", "select", "check", "uncheck"] = Field(description="Use type to fill text/number inputs; select is only for native <select> dropdowns. Click, scroll, or set checkbox state with check/uncheck.")
    selector: str = Field(description="Element ref from the current observation, or a CSS selector matching exactly one element")
    value: str = Field(default="", description="Required for type/select; use an explicit empty string to clear a field. For readback only use browser_observe.checks.")

    @model_validator(mode="before")
    @classmethod
    def require_input_value(cls, values):
        if isinstance(values, dict) and values.get("action") in {"type", "select"} and "value" not in values:
            raise ValueError("type/select requires value; use browser_observe(checks=...) to read without changing data")
        if not isinstance(values, dict) or not isinstance(values.get("checks"), list):
            return values
        missing = [item for item in values["checks"] if isinstance(item, dict) and not item.get("selector") and item.get("property") != "url"]
        if len(missing) == 1 and values.get("selector"):
            check = missing[0]
            action = values.get("action")
            expected_property = "value" if action in {"type", "select"} else "checked" if action in {"check", "uncheck"} else None
            expected = values.get("value") if expected_property == "value" else str(action == "check").lower()
            if check.get("property") == expected_property and equivalent(check.get("expected"), expected, check.get("value_type", "text")):
                values = {**values, "checks": [{**item, "selector": values["selector"]} if item is check else item for item in values["checks"]]}
        return values


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

        if params.readbackUrl or params.readbackSnapshot:
            return await read_result(page, params.readbackUrl, params.checks, params.resultRequirement, params.readbackSnapshot)

        # A successful readback needs no new page snapshot or element references.
        verification = None
        check_only = bool(params.checks and not params.selector and not params.taskHint
                          and not current_session().needs_observation)
        if check_only:
            verification = await check_page(page, params.checks)
            if verification["passed"]:
                if readback_resolved(verification, current_session().pending_verification, page):
                    current_session().pending_verification = None
                return CapabilityResult(success=True, output=(
                    f"URL: {page.url}\nResult checks: " + format_checks(verification)
                ), metadata={"url": page.url, "result_checks": verification})

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

        # Expose newly captured data APIs as an alternative to further UI interaction.
        from rune.capabilities.browser.network import get_network_monitor, hybrid_api_enabled
        _new_apis = get_network_monitor().unreported_interesting_count()
        if hybrid_api_enabled() and _new_apis > 0:
            header += (
                f"\n\U0001f4e1 {_new_apis} data API call(s) captured \u2014 "
                "browser_discover_apis(readBody='<url part>') reads the "
                "fetched JSON directly."
            )

        history_prefix = _compress_observe_history(title, len(elements))
        verification = await check_page(page, params.checks)
        if readback_resolved(verification, current_session().pending_verification, page):
            current_session().pending_verification = None

        checks_text = "\nResult checks: " + format_checks(verification) if verification else ""

        return CapabilityResult(
            success=True,
            output=f"{history_prefix}{header}\n\n{snapshot}{checks_text}",
            metadata={
                "url": url,
                "title": title,
                "selector": params.selector,
                "interactive_count": len(elements),
                "has_overlay": bool(overlay_warning),
                "result_checks": verification,
                "observed_controls": await read_controls(page),
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Observation failed: {exc}",
        )


async def _follow_new_tab(session, opened: list) -> Any:
    """Switch the session to the newest tab an action opened, or None if it closed or never loaded."""
    tabs = [tab for tab in opened if not tab.is_closed()]
    if not tabs:
        return None
    tab = tabs[-1]
    try:
        await tab.wait_for_load_state("domcontentloaded", timeout=10_000)
    except Exception as exc:
        log.debug("browser_new_tab_not_loaded", error=str(exc))
    if tab.is_closed():
        return None
    from rune.capabilities.browser.network import get_network_monitor

    session.page = tab
    await get_network_monitor().retarget(tab)
    return tab


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
        return CapabilityResult(success=False, error=(
            "The page changed outside this action (user control or a closed tab). "
            "Read the current page with browser_observe before acting."),
                                metadata={"action_status": "not_executed"})
    if session.uncertain_action:
        return CapabilityResult(success=False, error=(
            "A previous browser action has an unknown outcome. Inspect the page or ask the user "
            "to confirm its effects and resume from manual control; further changes remain paused."
        ), metadata={"action_status": "not_executed"})
    try:
        _, page = await _get_browser()
        action = params.action
        from rune.agent.run_control import current_control
        control = current_control()
        pending = session.pending_verification
        if (action != "scroll" and pending and pending["control"] is control
                and pending["page"] is page and pending["url"] == page.url):
            return CapabilityResult(success=False, error=(
                "The previous action was dispatched, but its result could not be read. "
                "Use browser_observe with corrected checks before further input. "
                "A failed check does not mean the write failed; do not submit it again to verify it."
            ), metadata={"action_status": "not_executed", "verification_required": True})
        if session.needs_observation:
            return CapabilityResult(success=False, error="The selected tab changed. Read it with browser_observe before acting.",
                                    metadata={"action_status": "not_executed"})
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

        from rune.capabilities.browser.network import get_network_monitor

        request_mark = get_network_monitor().request_count()
        before_url = page.url
        before_document =session.elements.document_id if session.elements is not None else ""
        before_snapshot = await _accessibility_snapshot(page)
        # Label -> value of each chosen option, read before a save can navigate away.
        selected_options = await page.evaluate("""() => [...document.querySelectorAll('select')].slice(0, 50)
            .flatMap(s => [...s.selectedOptions].map(o => [(o.label || o.text || '').trim(), o.value]))""") if action == "click" else []
        before_state = await _control_state(target) if target is not None else {}
        if action == "select" and before_state.get("tag") != "select":
            return CapabilityResult(success=False, error=(
                "select requires a native <select> dropdown. Use type for text or number inputs, "
                "or inspect the page for a custom dropdown. No action was dispatched."
            ), metadata={"action_status": "not_executed"})
        already_set = (action == "type" and before_state.get("value") == params.value
                       or action in {"check", "uncheck"} and before_state.get("checked") is (action == "check"))
        if already_set:
            verification = await check_page(page, params.checks)
            checks_text = "\nResult checks: " + format_checks(verification) if verification else ""
            return CapabilityResult(success=True, output=(
                f"The control already has the requested value. No input was sent.\nURL: {page.url}\n"
                + before_snapshot + checks_text
            ), metadata={"action_status": "not_executed", "unchanged": True, "action": action,
                         "selector": params.selector, "url": page.url, "result_checks": verification})
        key = (control.run_id if control else None, action, await node_key(target) if target else "", params.value)
        previous = session.last_input
        # A write the server rejected may be retried once with the same input.
        retry = bool(previous and previous["key"] == key and previous.get("rejected") and not previous.get("retry"))
        if (not is_manual_input() and action != "scroll" and previous and not retry
                and previous["key"] == key and previous["page"] is page
                and (previous["state"] is None or previous["state"] == (before_url, before_snapshot, before_state))):
            if params.readback:
                readback = await read_result(page, params.readback.url, params.readback.checks, params.readback.result)
                readback.metadata.update(input_replayed=True, action_status="not_executed", unchanged=True, action=action)
                return readback
            if params.checks:
                readback = await browser_observe(BrowserObserveParams(checks=params.checks))
                readback.output = "Input was already dispatched; only the requested results were read again.\n" + readback.output
                readback.metadata.update(input_replayed=True, action_status="not_executed", unchanged=True, action=action)
                return readback
            return CapabilityResult(success=True, output=(
                "This action was already dispatched in this run. "
                "It was not dispatched again. Read its result with browser_observe. "
                "If another identical submission is needed, ask the user to confirm it in a new turn.\n" + previous["output"]
            ), metadata={"replayed": True, "action_status": "not_executed"})
        # Trial failures happen before dispatch; a later failure still has an unknown outcome.
        deadline = time.monotonic() + 10
        if action == "click":
            await target.click(trial=True, timeout=10_000)
        elif action in {"check", "uncheck"}:
            await target.set_checked(action == "check", trial=True, timeout=10_000)
        if action != "scroll":
            controls = {control_identity(item): item for item in session.observed_checks} if session.checks_page is page else {}
            controls.update({control_identity(item): item for item in params.checks if item.property in {"value", "checked"}})
            preflight = await check_page(page, list(controls.values()), bind_effect=False) if action == "click" else None
            await begin_effect(page, action, preflight["checks"] if preflight else [], key[2], target)
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            return CapabilityResult(success=False, error="The target was not ready before the action deadline.",
                                    metadata={"action_status": "not_executed"})
        # A click may open a new tab; the managed browser follows it.
        opened_tabs: list = []
        followed = False
        follow_tabs = action == "click" and session.profile != "native"

        def on_tab(tab) -> None:  # Playwright needs a Python function, not a bound builtin
            opened_tabs.append(tab)

        if follow_tabs:
            page.context.on("page", on_tab)
        try:
            if action != "scroll":
                session.result_document = None
                session.readback_document = None
            if action == "click":
                await target.click(timeout=remaining)
            elif action == "type":
                await target.fill(params.value, timeout=remaining)
            elif action == "select":
                await target.select_option(params.value, timeout=remaining)
            elif action in {"check", "uncheck"}:
                await target.set_checked(action == "check", timeout=remaining)
            else:
                if params.value not in {"", "up", "down"}:
                    return CapabilityResult(success=False, error="Scroll value must be up or down",
                                            metadata={"action_status": "not_executed"})
                await page.evaluate("dy => window.scrollBy(0, dy)", -500 if params.value == "up" else 500)
        except BaseException as exc:
            session.uncertain_action = True
            if follow_tabs:
                page.context.remove_listener("page", on_tab)
            if not isinstance(exc, Exception):
                raise
            return CapabilityResult(success=False, error=(
                f"Action outcome is unknown: {exc}. The command was attempted once and was not retried. "
                "Inspect the current state before deciding what to do next."
            ), metadata={"action_status": "unknown", "action": action, "selector": params.selector})

        if action != "scroll":
            dispatched_effect()
            session.last_input = {"key": key, "page": page, "state": None, "retry": retry,
                                  "output": "Input was dispatched; observe its result before deciding whether more work is needed."}
        try:
            await wait_for_dom_settle(page)
            # Read the write outcome before the page snapshot so its rendered result is included.
            receipt = await get_network_monitor().write_receipt(request_mark) if action != "scroll" else None
            if receipt:
                await wait_for_dom_settle(page)
            after_state = await _control_state(target) if target is not None else {}
            if follow_tabs:
                page.context.remove_listener("page", on_tab)
                follow_tabs = False
                new_tab = await _follow_new_tab(session, opened_tabs)
                if new_tab is not None:
                    page, followed = new_tab, True
                    await wait_for_dom_settle(page)
            after_snapshot = await _accessibility_snapshot(page)
            elements = await extract_interactive_elements(page)
            changed = page.url != before_url or after_snapshot != before_snapshot or before_state != after_state
            summary = (f"Opened a new tab at {page.url}; continuing in that tab (the previous tab stays open)."
                       if followed else "Observed a state change." if changed else (
                "No visible state change was observed; this alone does not mean the action failed. "
                "Read back the requested values and confirmation below with browser_observe(checks=...). "
                "A separate HTTP fetch cannot read this tab's live state. Do not navigate or submit again just to check."
            ))
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
            document_changed = (page.url != before_url or bool(before_document and session.elements is not None
                                and session.elements.document_id != before_document))
            verification = await check_page(page, [] if document_changed else params.checks)
            # A known property mismatch needs corrected readback, not a pause for an unknown target.
            unread = [item for item in verification["checks"] if item["error"] and not item.get("property_hint")] if verification else []
            if unread and action != "scroll":
                unresolved = {}
                for item in unread:
                    unresolved.setdefault(check_identity(item), set()).add(item["selector"])
                session.pending_verification = {"control": control, "page": page, "url": page.url,
                                                "checks": unresolved}

            checks_text = "\nResult checks: " + format_checks(verification) if verification else ""
            if unread and action != "scroll":
                checks_text += "\nInput is paused until browser_observe reads corrected checks. Do not repeat this action to verify it."
            result = CapabilityResult(success=True, output=(
                f"Action dispatched: {action} on {params.selector}\nURL: {page.url}\n{summary}"
                f"\n{after_snapshot}{format_interactive_elements(elements)}{api_text}{checks_text}"
            ), metadata={"action_status": "dispatched", "action": action, "selector": params.selector,
                         "url": page.url, "page_changed": changed, "control_state": after_state,
                         "elements_refreshed": len(elements), "result_checks": verification})
            if params.readback:
                fetched = await read_result(page, params.readback.url, params.readback.checks, params.readback.result)
                result.metadata.update(fetched.metadata)
                result.output += "\n" + fetched.output
            if receipt:
                result.metadata["write_receipt"] = {**receipt, "selected_options": selected_options}
            if action != "scroll":
                session.last_input = {"key": key, "page": page, "retry": retry,
                                      "rejected": bool(receipt and receipt["status"] == "rejected"),
                                      "state": (page.url, after_snapshot, after_state), "output": result.output}
            return result
        except Exception as exc:
            log.debug("browser_post_action_observation_failed", error=str(exc))
            if follow_tabs:
                page.context.remove_listener("page", on_tab)
            session.needs_observation = True
            return CapabilityResult(success=True, output=(
                f"Action dispatched: {action}. Reading the resulting page failed: {exc}. "
                "Observe the page again; do not repeat the action to obtain its result."
            ), metadata={"action_status": "dispatched", "action": action, "observation_failed": True})
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
        return await target.evaluate("""element => {
            const el = element.localName === 'label' && element.control ? element.control : element;
            return {
            tag: el.localName,
            connected: el.isConnected, checked: el.checked, value: el.value,
            selected: el.selectedIndex, expanded: el.getAttribute('aria-expanded'),
            pressed: el.getAttribute('aria-pressed'), disabled: el.disabled
        }}""")
    except Exception as exc:
        log.debug("browser_control_state_unavailable", error=str(exc))
        return {}


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
            "Navigate to a URL in the conversation's browser and return a "
            "compact snapshot: page title, status, and the interactive elements "
            "with their ref IDs — there is no need to call browser_observe right "
            "after navigating. "
            "Use when page state, dynamic rendering or interaction is needed. For "
            "plain retrieval prefer web_search or web_fetch, and for a documented "
            "API endpoint or a .json/.txt/.md URL call web_fetch directly — the "
            "browser is far slower for those. "
            "Reuses the current session and preserves an already open exact URL "
            "unless reload=true. A new turn does not require navigation."
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
            "Open a URL for the user in Rune's conversation browser. "
            "Use when the user wants to watch the browser, interact with a site, "
            "log in, make a purchase, or says things like 'open', 'show me', "
            "'launch', 'pull up'. Reuses the existing session without resetting "
            "its page state. The Computer panel shows the page or its preview; "
            "without a session, launches a visible browser. Existing Chrome tabs are not attached."
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
        description="Interact with required page controls (click/type/scroll/select). Use fresh refs from the current page. For information already available through search, fetch or an API, use those tools instead.",
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
