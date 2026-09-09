"""Browser automation helpers for RUNE.

Ported from src/capabilities/browser.ts - DOM settlement detection,
overlay dismissal, multi-selector element location, and self-healing
element find with Playwright-native locator strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Configuration constants (mirrored from TS CONFIG)

DOM_SETTLE_TIMEOUT_MS = 2000
NETWORK_IDLE_TIMEOUT_MS = 5000
MUTATION_QUIET_PERIOD_MS = 500
MUTATION_MAX_WAIT_MS = 3000
ACTION_TIMEOUT_MS = 10_000
RETRY_DELAY_MS = 500
MAX_RETRIES = 2

# Interactive ARIA roles that represent actionable elements.
INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "searchbox", "combobox",
    "listbox", "option", "menuitem", "menuitemcheckbox", "menuitemradio",
    "tab", "checkbox", "radio", "switch", "slider", "spinbutton",
    "menu", "treeitem",
})


# 1. DOM settle detection

# JavaScript injected into the page to wait for DOM stability.
_DOM_SETTLE_JS = """
(cfg) => new Promise(resolve => {
    let quietTimer;
    const maxTimer = setTimeout(() => {
        observer.disconnect();
        resolve();
    }, cfg.maxWait);

    const observer = new MutationObserver(() => {
        clearTimeout(quietTimer);
        quietTimer = setTimeout(() => {
            observer.disconnect();
            clearTimeout(maxTimer);
            resolve();
        }, cfg.quietPeriod);
    });

    observer.observe(document.body || document.documentElement, {
        childList: true, subtree: true, attributes: true,
    });

    // Initial quiet timer (no mutations yet -> already settled).
    quietTimer = setTimeout(() => {
        observer.disconnect();
        clearTimeout(maxTimer);
        resolve();
    }, cfg.quietPeriod);
})
"""


async def wait_for_dom_settle(page: Any) -> None:
    """Wait for the page DOM to stabilise after navigation or action.

    Two-phase strategy matching the TypeScript implementation:
    1. Race between ``networkidle`` and a hard timeout.
    2. MutationObserver quiet-period detection for SPA hydration.
    """
    # Phase 1: network settle (best-effort).
    try:
        await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
    except Exception:
        pass  # Timeout or navigation - continue to phase 2.

    # Phase 2: MutationObserver quiet period.
    try:
        await page.evaluate(
            _DOM_SETTLE_JS,
            {"quietPeriod": MUTATION_QUIET_PERIOD_MS, "maxWait": MUTATION_MAX_WAIT_MS},
        )
    except Exception as exc:
        log.debug("dom_settle_mutation_failed", error=str(exc))


@dataclass(slots=True)
class ElementMeta:
    """Metadata for an interactive page element."""
    ref: str
    role: str
    name: str
    tag: str = ""
    input_type: str = ""
    is_disabled: bool = False
    selectors: list[dict[str, Any]] = field(default_factory=list)
    # selectors: [{"type": "role"|"text"|"label"|"placeholder"|"testid"|"css",
    #              "value": str, "confidence": float}]
    breadcrumb: str = ""  # ancestor context path (e.g. "main > list > listitem")


class ElementStore:
    """In-memory store of interactive elements observed on a page."""

    def __init__(self) -> None:
        self._elements: dict[str, ElementMeta] = {}
        self._last_url: str = ""
        self._last_observe_time: float = 0.0
        self._ref_counter: int = 0
        self._ref_prefix = uuid4().hex
        self.page: Any = None
        self.document: Any = None
        self.handles: dict[str, Any] = {}

    def clear(self) -> None:
        """Clear metadata without ever reusing a retired reference."""
        self._elements.clear()

    async def release(self) -> None:
        handles = [*self.handles.values(), self.document]
        self.handles.clear()
        self.document = self.page = None
        self.clear()
        for handle in handles:
            if handle is not None:
                try:
                    await handle.dispose()
                except Exception as exc:
                    log.debug("browser_reference_release_failed", error=str(exc))

    def get(self, ref: str) -> ElementMeta | None:
        return self._elements.get(ref)

    def put(self, meta: ElementMeta) -> None:
        self._elements[meta.ref] = meta

    @property
    def last_url(self) -> str:
        return self._last_url

    @last_url.setter
    def last_url(self, url: str) -> None:
        self._last_url = url

    @property
    def last_observe_time(self) -> float:
        return self._last_observe_time

    @last_observe_time.setter
    def last_observe_time(self, t: float) -> None:
        self._last_observe_time = t

    def next_ref(self) -> str:
        ref = f"e{self._ref_prefix}_{self._ref_counter}"
        self._ref_counter += 1
        return ref

    @property
    def all(self) -> dict[str, ElementMeta]:
        return self._elements


def get_element_store() -> ElementStore:
    from rune.capabilities.browser.session import current_session

    session = current_session()
    if session.elements is None:
        session.elements = ElementStore()
    return session.elements


async def extract_interactive_elements(page: Any, root_selector: str = "") -> list[ElementMeta]:
    """Extract interactive elements from the page using the accessibility tree.

    Populates the element store and returns the list of discovered elements.
    Uses Playwright 1.58+ ``aria_snapshot()`` which returns a YAML-formatted
    accessibility tree.  Interactive roles are parsed from the YAML lines.
    """
    store = get_element_store()
    previous = {(meta.role, meta.name): (meta, store.handles[meta.ref])
                for meta in store.all.values() if meta.ref in store.handles}
    previous_handles = [*store.handles.values(), store.document]
    previous_document = store.document
    same_page = store.page is page and store.last_url == page.url
    store.clear()
    store.handles = {}
    store.document = None
    try:
        same_document = False
        if same_page and previous_document is not None:
            try:
                same_document = await previous_document.evaluate("doc => doc === document")
            except Exception as exc:
                log.debug("browser_document_changed", error=str(exc))
        store.page = page
        store.document = await page.evaluate_handle("document")
        root = page.locator(root_selector) if root_selector else page.locator(":root")
        snapshot_text = await root.aria_snapshot()
        elements: list[ElementMeta] = []
        _parse_aria_snapshot(snapshot_text or "", elements, store)
        for meta in elements:
            locator = page.get_by_role(meta.role, name=meta.name, exact=True)
            try:
                if await locator.count() != 1:
                    continue
                handle = await locator.element_handle(timeout=1000)
                if handle is None:
                    continue
                store.handles[meta.ref] = handle
                observed = previous.get((meta.role, meta.name)) if same_document else None
                if observed and await handle.evaluate("(node, old) => node === old", observed[1]):
                    # A fresh observation does not invalidate an unchanged node.
                    # Replacements with the same label still receive a new ref.
                    store.handles.pop(meta.ref)
                    store.all.pop(meta.ref)
                    meta.ref = observed[0].ref
                    store.handles[meta.ref] = handle
                    store.put(meta)
            except Exception as exc:
                log.debug("browser_reference_unavailable", ref=meta.ref, error=str(exc))
        store.last_url = page.url
        store.last_observe_time = time.monotonic()
        return elements
    except Exception as exc:
        log.warning("ax_snapshot_failed", error=str(exc))
        await store.release()
        return []
    finally:
        for handle in previous_handles:
            if handle is not None:
                try:
                    await handle.dispose()
                except Exception as exc:
                    log.debug("browser_reference_release_failed", error=str(exc))


# Regex to parse aria_snapshot YAML lines at ANY indent level.
# Matches patterns like:
#   - button "Submit"
#       - link "Home":
#           - textbox "Search" [value=hello]
#   - cell "Product A"
import re

_ARIA_LINE_RE = re.compile(
    r"^\s*-\s+(\w+)(?:\s+\"([^\"]*)\")?"
)

# Semantic roles worth showing in breadcrumb context (#P3 DOM Distillation).
_CONTEXT_ROLES = frozenset({
    "navigation", "main", "complementary", "banner", "contentinfo",
    "region", "section", "article", "dialog", "form", "list",
    "listitem", "table", "row", "group", "tabpanel", "menu",
    "heading", "cell",
})


def _build_breadcrumb(ancestor_stack: list[tuple[int, str, str]]) -> str:
    """Build a compact ancestor path from the indentation stack.

    Only includes semantic roles (navigation, main, list, etc.) to
    keep breadcrumbs concise. Shows last 3 ancestors maximum.
    """
    parts: list[str] = []
    for _, role, name in ancestor_stack:
        if role not in _CONTEXT_ROLES:
            continue
        if name:
            parts.append(f'{role}"{name}"')
        else:
            parts.append(role)
    if not parts:
        return ""
    return " > ".join(parts[-3:])


def _parse_aria_snapshot(
    text: str,
    out: list[ElementMeta],
    store: ElementStore,
) -> None:
    """Parse Playwright aria_snapshot YAML output into ElementMeta entries.

    Uses indentation-based stack to track ancestor context, producing
    breadcrumb paths like ``main > list > listitem"Hotel Name"`` for
    each interactive element (#P3 DOM Distillation / AgentOccam).
    """
    ancestor_stack: list[tuple[int, str, str]] = []  # (indent, role, name)

    for line in text.splitlines():
        m = _ARIA_LINE_RE.match(line)
        if not m:
            continue

        indent = len(line) - len(line.lstrip())
        role = m.group(1)
        name = m.group(2) or ""

        # Pop ancestors at same or deeper indent level
        while ancestor_stack and ancestor_stack[-1][0] >= indent:
            ancestor_stack.pop()

        if role in INTERACTIVE_ROLES:
            breadcrumb = _build_breadcrumb(ancestor_stack)
            ref = store.next_ref()
            selectors = _build_selectors(role, name, {})
            meta = ElementMeta(
                ref=ref, role=role, name=name,
                selectors=selectors, breadcrumb=breadcrumb,
            )
            store.put(meta)
            out.append(meta)

        ancestor_stack.append((indent, role, name))


def _build_selectors(role: str, name: str, node: dict) -> list[dict[str, Any]]:
    """Build a priority-ordered list of locator selectors for an element."""
    selectors: list[dict[str, Any]] = []

    # 1. role + name (most reliable with Playwright)
    if role and name:
        selectors.append({"type": "role", "value": name, "role": role, "confidence": 0.90})

    # 2. text content (for buttons / links)
    if role in ("button", "link", "tab", "menuitem") and name:
        selectors.append({"type": "text", "value": name, "confidence": 0.75})

    # 3. label (for inputs)
    if role in ("textbox", "searchbox", "combobox", "spinbutton", "slider") and name:
        selectors.append({"type": "label", "value": name, "confidence": 0.85})

    # 4. placeholder (also for inputs, lower confidence)
    if role in ("textbox", "searchbox") and name:
        selectors.append({"type": "placeholder", "value": name, "confidence": 0.80})

    return selectors


MAX_LISTED_ELEMENTS = 50


def format_interactive_elements(
    elements: list[ElementMeta], max_elements: int = MAX_LISTED_ELEMENTS
) -> str:
    """Render the ref-tagged element list the model acts on.

    Shared by navigate and observe so a page reads the same either way — and
    so navigate can return one, which spares the model an observe round just
    to learn the refs.
    """
    if not elements:
        return ""
    shown = elements[:max_elements]
    lines = [f"\n--- Interactive Elements ({len(shown)}/{len(elements)}) ---"]
    for meta in shown:
        parts = [f"[{meta.ref}]", meta.role]
        if meta.name:
            parts.append(f'"{meta.name}"')
        if meta.breadcrumb:
            parts.append(f"in({meta.breadcrumb})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def is_element_ref(value: str) -> bool:
    return re.fullmatch(r"e(?:[0-9a-f]{32}_)?\d+", value) is not None


async def find_element_locator(page: Any, ref: str) -> Any | None:
    """Return the observed node, never a replacement selected by a similar name."""
    store = get_element_store()
    meta, handle = store.get(ref), store.handles.get(ref)
    if meta is None or handle is None or store.page is not page or store.last_url != page.url:
        return None
    try:
        if not await store.document.evaluate("doc => doc === document"):
            return None
        if not await handle.evaluate("el => el.isConnected && el.ownerDocument === document"):
            return None
        locator = page.get_by_role(meta.role, name=meta.name, exact=True)
        if await locator.count() != 1:
            return None
        if await locator.evaluate("(el, observed) => el === observed", handle, timeout=1000):
            return handle
    except Exception as exc:
        log.debug("browser_reference_stale", ref=ref, error=str(exc))
    return None


async def self_healing_find(page: Any, ref: str) -> Any | None:
    # Kept for callers using the old helper name. Recovery requires a new observation.
    return await find_element_locator(page, ref)


# 5. Scroll helpers

_SCROLL_INFO_JS = """
() => ({
    scrollY: window.scrollY,
    scrollHeight: document.documentElement.scrollHeight,
    viewportHeight: window.innerHeight,
    scrollPercent: document.documentElement.scrollHeight <= window.innerHeight
        ? 100
        : Math.round(window.scrollY / (document.documentElement.scrollHeight - window.innerHeight) * 100),
})
"""


async def get_scroll_info(page: Any) -> dict[str, int]:
    """Return current scroll position and page dimensions."""
    try:
        return await page.evaluate(_SCROLL_INFO_JS)
    except Exception:
        return {"scrollY": 0, "scrollHeight": 0, "viewportHeight": 0, "scrollPercent": 0}
