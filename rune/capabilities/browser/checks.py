"""Read page state for explicit post-action checks."""

import json
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from rune.capabilities.browser.values import ValueType, equivalent


class BrowserCheck(BaseModel):
    selector: str = Field(default="", max_length=300, description="Current element ref or unique CSS selector; empty only for URL checks")
    property: Literal["text", "value", "checked", "url"] = Field(description="value reads input/textarea/select values; checked reads checkbox state; text reads non-input content.")
    expected: str | None = Field(max_length=1000, description="Exact requested value; null observes a message without asserting its wording. Empty text/value is valid; checked uses true or false.")
    requirement: str = Field(default="", max_length=1000, description="Source quote when requirement_ids are unavailable; action labels do not confirm results.")
    requirement_ids: list[str] = Field(default_factory=list, max_length=1, description="Optional field requirement ID. Rune binds a unique requested value automatically; supply the ID or quote when values are ambiguous. Do not use a save ID for a field.")
    value_type: ValueType = Field(default="text", description="Comparison type from the fixed requirement; identifiers and prose use text.")

    @model_validator(mode="after")
    def validate_expected(self):
        if self.expected is None and (self.property != "text" or self.value_type != "text"):
            raise ValueError("Only text observations may omit an expected value")
        if self.property != "url" and not self.selector.strip():
            raise ValueError("A result check needs a current element ref or unique CSS selector")
        if self.property == "checked" and self.expected not in {"true", "false"}:
            raise ValueError("checked expects true or false")
        if self.property == "url" and not self.expected:
            raise ValueError("A URL check needs a nonempty URL")
        if self.property in {"url", "checked"} and self.value_type != "text":
            raise ValueError("URL and checked comparisons use text")
        return self


async def read_controls(page) -> list[dict]:
    from rune.capabilities.browser.helpers import get_element_store
    from rune.utils.logger import get_logger

    store = get_element_store()
    if store.page is not page:
        return []
    targets = [handle for ref, handle in store.handles.items()
               if store.get(ref).role in {"textbox", "searchbox", "spinbutton", "combobox", "checkbox", "radio", "switch"}][:32]
    if not targets:
        return []
    try:
        return await page.evaluate("""({targets, prefix}) => {
            const state = window.__runeNodes ||= {prefix, ids: new WeakMap(), next: 0};
            return targets.flatMap(el => {
                if (!el.isConnected || !el.getClientRects().length || getComputedStyle(el).visibility !== 'visible'
                    || !el.matches('input,textarea,select') || ['password','hidden','file'].includes(el.type)) return [];
                if (!state.ids.has(el)) state.ids.set(el, state.prefix + ':' + state.next++);
                const checked = ['checkbox','radio'].includes(el.type);
                const actual = checked ? String(el.checked) : el.value;
                return actual.length <= 1000 ? [{node_id: state.ids.get(el), property: checked ? 'checked' : 'value', actual}] : [];
            });
        }""", {"targets": targets, "prefix": uuid4().hex})
    except Exception as exc:
        get_logger(__name__).debug("browser_control_baseline_unavailable", error=type(exc).__name__)
        return []


async def check_page(page, checks: list[BrowserCheck], *, bind_effect: bool = True) -> dict | None:
    if not checks:
        return None
    from rune.capabilities.browser.effects import control_snapshots, effect_for
    from rune.capabilities.browser.helpers import find_element_locator, is_element_ref
    from rune.capabilities.browser.session import current_session

    effect = effect_for(page) if bind_effect else None
    session = current_session()
    targets = session.check_targets if session.checks_page is page else {}
    results = []
    for check in checks:
        actual = None
        error = ""
        property_hint = ""
        kind = "url"
        changed = missing = input_target = False
        node_id = None
        try:
            if check.property == "url":
                actual = page.url
            else:
                target = (await find_element_locator(page, check.selector) if is_element_ref(check.selector)
                          else page.locator(check.selector))
                saved = targets.get(check.selector)
                if target is None and saved and saved["url"] == page.url:
                    # Recover a readback target by its unique ID; input actions still require fresh refs.
                    candidate = page.locator('[id=' + json.dumps(saved["id"]) + ']')
                    if await candidate.count() == 1:
                        identity = await candidate.evaluate("el => [el.localName, el.type || '', el.getAttribute('name') || '']")
                        if identity == saved["identity"]:
                            target = candidate
                if target is None or not is_element_ref(check.selector) and await target.count() != 1:
                    missing = target is None or await target.count() == 0
                    raise ValueError("A check must identify exactly one element")
                if await target.get_attribute("type") == "password":
                    raise ValueError("Password fields cannot be read back")
                state = await target.evaluate("""(el, {effect, prefix, remember}) => {
                    const nodes = window.__runeNodes ||= {prefix, ids: new WeakMap(), next: 0};
                    if (!nodes.ids.has(el)) nodes.ids.set(el, nodes.prefix + ':' + nodes.next++);
                    return {
                    action: !!el.closest('button,a,[role="button"],[role="link"],input[type="submit"],input[type="button"]'),
                    editable: el.isContentEditable || !!el.closest('input,textarea,select,option,optgroup,[role="combobox"],[role="listbox"]'),
                    valueProperty: el.matches('input,textarea') ? (['checkbox','radio'].includes(el.type) ? 'checked' : 'value') : '',
                    changed: !!effect && (window.__runeEffect?.id !== effect.id || window.__runeEffect.touched(el)),
                    inputTarget: !!effect && nodes.ids.get(el) === effect.target,
                    nodeId: nodes.ids.get(el),
                    targetId: el.id,
                    uniqueId: remember && !!el.id && document.querySelectorAll('[id="' + CSS.escape(el.id) + '"]').length === 1,
                    identity: [el.localName, el.type || '', el.getAttribute('name') || '']
                }}""", {"effect": {"id": effect["id"], "target": effect["target"]} if effect else None, "prefix": uuid4().hex, "remember": bind_effect and is_element_ref(check.selector)})
                if state["uniqueId"]:
                    targets[check.selector] = {"id": state["targetId"], "identity": state["identity"], "url": page.url}
                changed = state["changed"]
                input_target = state["inputTarget"]
                node_id = state["nodeId"]
                kind = "control" if state["editable"] or check.property in {"value", "checked"} else "result"
                if state["action"] and check.property == "text":
                    raise ValueError("An action label does not confirm its result")
                if check.property == "text" and state["valueProperty"]:
                    property_hint = state["valueProperty"]
                    raise ValueError(f"Read this control with property='{property_hint}', not 'text'.")
                if check.property == "text":
                    actual = (await target.inner_text()).strip()
                elif check.property == "checked":
                    actual = str(await target.is_checked()).lower()
                else:
                    actual = await target.input_value()
        except Exception as exc:
            error = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        observed = check.expected is None
        passed = (isinstance(actual, str) if observed else equivalent(actual, check.expected, check.value_type)) and not error
        actual = actual[:1000] if isinstance(actual, str) else None
        results.append({**check.model_dump(), "actual": actual,
                        **({"expected": actual, "observed": True} if observed else {}),
                        "kind": kind, "passed": passed, "error": error,
                        "changed": changed, "missing": missing, "input_target": input_target, "node_id": node_id,
                        **({"property_hint": property_hint} if property_hint else {})})
    confirmed = bool(effect and effect["action"] == "click"
                     and any(item["kind"] == "result" and item["changed"] and item["passed"] for item in results))
    if confirmed:
        inputs = {(*check_identity(item), item["selector"]): item for item in effect["inputs"] if item["passed"]}
        removed = await page.evaluate("""({id, inputs}) => {
            const state = window.__runeEffect;
            if (state?.id !== id) return inputs;
            return state.inputs.filter(el => !el.isConnected).map(el => window.__runeNodes.ids.get(el));
        }""", {"id": effect["id"], "inputs": [item.get("node_id") for item in effect["inputs"]]}) if any(item["missing"] for item in results) else []
        for item in results:
            # A removed control can use its pre-submit value only alongside a fresh result.
            before = inputs.get((*check_identity(item), item["selector"]))
            if item["missing"] and before and before["kind"] == "control" and before.get("node_id") in removed:
                item.update(actual=before["actual"], kind="submitted_control", passed=True, error="", source="before_action")
    if bind_effect:
        controls = {control_identity(item): item for item in session.observed_checks} if session.checks_page is page else {}
        session.checks_page = page
        session.check_targets = dict(list(targets.items())[-64:])
        controls.update({control_identity(item): item for item in checks if item.property in {"value", "checked"}})
        session.observed_checks = list(controls.values())[-32:]
    controls = None
    if effect and effect["action"] == "click":
        controls, submitted = await control_snapshots(page, effect["id"])
        if submitted is not None:
            effect["controls"] = submitted
    return {"passed": all(result["passed"] for result in results), "checks": results, "effect_observed": confirmed,
            "effect": {"id": effect["id"], "target": effect["target"], "controls": effect.get("controls")} if effect else None,
            "controls": controls}


def check_identity(item: dict) -> tuple[str, str, str | None]:
    return item["property"], item.get("requirement", ""), None if item.get("observed") else item["expected"]


def control_identity(item: BrowserCheck) -> tuple[str, str, str | None, str]:
    return item.property, item.requirement, item.expected, item.selector


def format_checks(verification: dict) -> str:
    return json.dumps({**{key: value for key, value in verification.items() if key not in {"effect", "controls"}}, "checks": [
        {key: value for key, value in item.items() if key not in {"changed", "missing", "input_target", "node_id"}}
        for item in verification["checks"]
    ]}, ensure_ascii=False)


def readback_resolved(verification: dict | None, pending: dict | None, page) -> bool:
    if not verification or not pending or pending["page"] is not page or pending["url"] != page.url:
        return False
    resolved = set()
    for item in verification["checks"]:
        key = check_identity(item)
        if (key not in pending["checks"] and item["property"] == "text"
                and (key[0], key[1], None) in pending["checks"]):
            key = (key[0], key[1], None)
        selectors = pending["checks"].get(key, set())
        if not item["error"] and (item["selector"] in selectors or item.get("input_target")
                                  or verification.get("effect_observed") and item.get("changed") and item.get("kind") == "result"):
            resolved.add(key)
    return set(pending["checks"]) <= resolved
