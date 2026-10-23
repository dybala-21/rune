"""Bind page observations to the input that preceded them."""

import hashlib
import json
from uuid import uuid4

from rune.capabilities.browser.session import current_session

_CONTROL_SNAPSHOT = """root => {
    const nodes = window.__runeNodes;
    if (!nodes) return {complete: false, values: {}};
    const values = {};
    let complete = true, count = 0;
    const selector = 'input,textarea,select,[contenteditable="true"]';
    const fields = new Set(root.querySelectorAll(selector));
    if (root.localName === 'form') {
        for (const el of root.elements) if (el.matches(selector)) fields.add(el);
    }
    for (const el of fields) {
        if (!el.getClientRects().length || getComputedStyle(el).visibility !== 'visible'
            || ['password','hidden','file','submit','button','reset','image'].includes(el.type)) continue;
        if (++count > 128) { complete = false; break; }
        const property = ['checkbox','radio'].includes(el.type) ? 'checked' : el.isContentEditable ? 'text' : 'value';
        const value = String(property === 'checked' ? el.checked : property === 'text' ? el.innerText : el.value);
        if (value.length > 1000) { complete = false; continue; }
        if (!nodes.ids.has(el)) nodes.ids.set(el, nodes.prefix + ':' + nodes.next++);
        values[nodes.ids.get(el)] = [property, value];
    }
    return {complete, values};
}"""


def _fingerprints(snapshot: dict) -> dict:
    return {**snapshot, "values": {key: hashlib.sha256(json.dumps(value).encode()).hexdigest()
                                  for key, value in snapshot["values"].items()}}


async def control_snapshots(page, effect_id: str | None = None) -> tuple[dict, dict | None]:
    state = await page.evaluate("""id => ({
        current: (CONTROL_SNAPSHOT)(document),
        submitted: id && window.__runeEffect?.id === id ? window.__runeEffect.controls : null
    })""".replace("CONTROL_SNAPSHOT", _CONTROL_SNAPSHOT), effect_id)
    return _fingerprints(state["current"]), _fingerprints(state["submitted"]) if state["submitted"] else None


async def node_key(target) -> str:
    return await target.evaluate("""(el, prefix) => {
        if (el.localName === 'label' && el.control) el = el.control;
        const state = window.__runeNodes ||= {prefix, ids: new WeakMap(), next: 0};
        if (!state.ids.has(el)) state.ids.set(el, state.prefix + ':' + state.next++);
        return state.ids.get(el);
    }""", uuid4().hex)


async def begin_effect(page, action: str, inputs: list[dict], target: str, element) -> None:
    session = current_session()
    session.last_effect = None
    identity = uuid4().hex
    snapshot = await element.evaluate("""(el, {id, inputs, action}) => {
        window.__runeEffect?.observer.disconnect();
        window.__runeEffect?.clickListener?.abort();
        const changed = new WeakSet(), added = new WeakSet();
        const collect = records => {
            for (const record of records) {
                for (let node = record.target; node; node = node.parentNode) changed.add(node);
                for (const node of record.addedNodes) added.add(node);
            }
        };
        const observer = new MutationObserver(collect);
        observer.observe(document, {childList: true, characterData: true, subtree: true});
        const wanted = new Set(inputs);
        const fields = wanted.size ? [...document.querySelectorAll('input,textarea,select,[contenteditable],[role="combobox"],[role="listbox"]')]
            .filter(el => wanted.has(window.__runeNodes?.ids.get(el))) : [];
        const target = el.localName === 'label' && el.control ? el.control : el;
        const scope = target.form || target.closest('form') || document;
        const snapshot = CONTROL_SNAPSHOT;
        const clickListener = new AbortController();
        const state = window.__runeEffect = {id, observer, clickListener,
            controls: action === 'click' ? snapshot(scope) : null, inputs: fields, touched(el) {
            collect(observer.takeRecords());
            if (changed.has(el)) return true;
            for (let node = el; node; node = node.parentNode) if (added.has(node)) return true;
            return false;
        }};
        // Checkboxes and radios update before their click handlers run.
        if (action === 'click') document.addEventListener('click', event => {
            if (!event.composedPath().includes(target)) return;
            state.controls = snapshot(scope);
            clickListener.abort();
        }, {capture: true, signal: clickListener.signal});
        return state.controls;
    }""".replace("CONTROL_SNAPSHOT", _CONTROL_SNAPSHOT),
        {"id": identity, "inputs": [item.get("node_id") for item in inputs], "action": action})
    session.last_effect = {"page": page, "id": identity, "action": action, "target": target,
                           "inputs": inputs, "controls": _fingerprints(snapshot) if snapshot else None, "dispatched": False}


def dispatched_effect() -> None:
    if effect := current_session().last_effect:
        effect["dispatched"] = True


def effect_for(page) -> dict | None:
    effect = current_session().last_effect
    return effect if effect and effect["page"] is page and effect["dispatched"] else None
