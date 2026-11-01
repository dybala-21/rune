"""Bind observed controls to actual nodes, including repeated accessible names."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.capabilities.browser.helpers import ElementMeta, ElementStore

log = get_logger(__name__)

_CONTEXT = """el => {
    const own = (el.innerText || '').trim();
    let parent = el.parentElement;
    let text = '';
    for (let depth = 0; parent && depth < 5; depth++, parent = parent.parentElement) {
        const candidate = (parent.innerText || '').replace(/\\s+/g, ' ').trim();
        if (candidate.length > 1200) break;
        if (candidate !== own) text = candidate;
        if (parent.matches('article, li, tr, section, [role="group"], [role="listitem"]')) break;
    }
    return text.slice(0, 300);
}"""


async def bind_nodes(page: Any, root: Any, elements: list[ElementMeta], store: ElementStore,
                     previous: list[tuple[ElementMeta, Any]]) -> list[ElementMeta]:
    # Repeated AX labels do not establish node order; bind the matching DOM nodes.
    templates: dict[tuple[str, str], ElementMeta] = {}
    for meta in elements:
        templates.setdefault((meta.role, meta.name), meta)
    store.clear()
    bound = []
    for key, template in templates.items():
        handles = []
        try:
            locator = root.get_by_role(template.role, name=template.name, exact=True)
            handles = await locator.element_handles()
            count = await page.get_by_role(template.role, name=template.name, exact=True).count()
            old = [(meta, handle) for meta, handle in previous if (meta.role, meta.name) == key]
            for handle in handles:
                if not await handle.evaluate("el => el.isConnected && el.ownerDocument === document"):
                    continue
                matched = await handle.evaluate(
                    "(node, old) => old.findIndex(candidate => candidate === node)",
                    [item[1] for item in old],
                ) if old else -1
                ref = old[matched][0].ref if matched >= 0 else store.next_ref()
                meta = replace(template, ref=ref, match_count=count)
                if count > 1:
                    # Derive context from this node rather than a shared AX template.
                    meta.breadcrumb = ""
                    meta.context = await handle.evaluate(_CONTEXT)
                if meta.role == "link":
                    meta.href = await handle.evaluate("el => el.href || ''")
                store.put(meta)
                store.handles[ref] = handle
                bound.append(meta)
        except Exception as exc:
            log.debug("browser_node_binding_failed", role=key[0], error=str(exc))
        finally:
            retained = set(store.handles.values())
            for handle in handles:
                if handle not in retained:
                    await handle.dispose()
    return bound
