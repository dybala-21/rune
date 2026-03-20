"""Browser automation helpers for RUNE.

Ported from src/capabilities/browser.ts - DOM settlement detection,
overlay dismissal, multi-selector element location, and self-healing
element find with Playwright-native locator strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

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


# 2. Overlay / modal dismissal

_DISMISS_OVERLAYS_JS = """
() => {
    const results = [];

    // --- Tier 1: CSS selector targeting ---
    const closeSelectors = [
        '[class*="cookie"] button[class*="accept"]',
        '[class*="consent"] button[class*="agree"]',
        '[class*="modal"] [class*="close"]',
        '[role="dialog"] button[aria-label*="close"]',
        '[role="dialog"] button[aria-label*="닫기"]',
        '[class*="app-banner"] [class*="close"]',
        'button[class*="dismiss"]',
        'button.close',
    ];

    for (const sel of closeSelectors) {
        try {
            const btn = document.querySelector(sel);
            if (btn && btn.offsetParent !== null) {
                btn.click();
                results.push('css:' + sel);
            }
        } catch {}
    }

    // --- Tier 2: text-based detection ---
    const closeTexts = [
        '닫기', '확인', '동의', '다음에', '나중에',
        'Close', 'Dismiss', 'Accept', 'Got it', 'No thanks',
        '×', '✕', '✖', '╳',
    ];
    const overlayParents = '[class*="modal"],[class*="overlay"],[role="dialog"],[role="alertdialog"]';

    function isFixed(el) {
        let cur = el;
        while (cur && cur !== document.body) {
            const s = getComputedStyle(cur);
            if (s.position === 'fixed' || s.position === 'sticky') return true;
            cur = cur.parentElement;
        }
        return false;
    }

    for (const btn of document.querySelectorAll('button, [role="button"]')) {
        const text = (btn.textContent || '').trim();
        const ariaLabel = btn.getAttribute('aria-label') || '';
        const match = closeTexts.some(t => t === text || t === ariaLabel);
        if (match) {
            const inOverlay = btn.closest(overlayParents) || isFixed(btn);
            if (inOverlay && btn.offsetParent !== null) {
                try { btn.click(); results.push('text:' + text); } catch {}
            }
        }
    }

    // --- Tier 3: high z-index blocking overlays ---
    for (const el of document.querySelectorAll('div, section, aside')) {
        try {
            const s = getComputedStyle(el);
            const z = parseInt(s.zIndex);
            if (z > 1000 && (s.position === 'fixed' || s.position === 'absolute')) {
                const r = el.getBoundingClientRect();
                if (r.width > window.innerWidth * 0.5 && r.height > window.innerHeight * 0.3) {
                    const closeBtn = el.querySelector(
                        '[class*="close"], [aria-label*="close"], [aria-label*="닫기"]'
                    );
                    if (closeBtn) {
                        closeBtn.click();
                        results.push('zindex:' + z);
                    }
                }
            }
        } catch {}
    }

    // --- Tier 4: force-hide if nothing else worked ---
    if (results.length === 0) {
        for (const el of document.querySelectorAll('div, section, aside')) {
            try {
                const s = getComputedStyle(el);
                const z = parseInt(s.zIndex);
                if (z > 1000 && (s.position === 'fixed' || s.position === 'absolute')) {
                    const r = el.getBoundingClientRect();
                    if (r.width > window.innerWidth * 0.8 && r.height > window.innerHeight * 0.5) {
                        el.style.display = 'none';
                        results.push('force-hidden:zindex-' + z);
                    }
                }
            } catch {}
        }
        // Also hide any dimmed/backdrop overlays
        for (const el of document.querySelectorAll('[class*="dim"], [class*="backdrop"], [class*="mask"]')) {
            try {
                const s = getComputedStyle(el);
                if (s.position === 'fixed' || s.position === 'absolute') {
                    el.style.display = 'none';
                    results.push('force-hidden:backdrop');
                }
            } catch {}
        }
    }

    return results;
}
"""


async def dismiss_blocking_overlays(page: Any) -> list[str]:
    """Attempt to dismiss blocking overlays, modals, and cookie banners.

    Returns a list of identifiers for dismissed elements.
    """
    try:
        dismissed: list[str] = await page.evaluate(_DISMISS_OVERLAYS_JS)
        if dismissed:
            log.debug("overlays_dismissed", count=len(dismissed), details=dismissed)
            await page.wait_for_timeout(300)  # animation settle
        return dismissed
    except Exception as exc:
        log.debug("dismiss_overlays_failed", error=str(exc))
        return []


# 3. Element store & multi-selector location

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


class ElementStore:
    """In-memory store of interactive elements observed on a page."""

    def __init__(self) -> None:
        self._elements: dict[str, ElementMeta] = {}
        self._last_url: str = ""
        self._last_observe_time: float = 0.0
        self._ref_counter: int = 0

    def clear(self) -> None:
        self._elements.clear()
        self._ref_counter = 0

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
        ref = f"e{self._ref_counter}"
        self._ref_counter += 1
        return ref

    def find_similar(self, role: str, name: str) -> ElementMeta | None:
        """Find an element with the same role and overlapping name."""
        prefix = name[:20] if name else ""
        if not prefix:
            return None
        for meta in self._elements.values():
            if meta.role == role and prefix in meta.name:
                return meta
        return None

    @property
    def all(self) -> dict[str, ElementMeta]:
        return self._elements


# Module-level element store (shared across calls within a session).
_element_store = ElementStore()


def get_element_store() -> ElementStore:
    return _element_store


async def extract_interactive_elements(page: Any, root_selector: str = "") -> list[ElementMeta]:
    """Extract interactive elements from the page using the accessibility tree.

    Populates the element store and returns the list of discovered elements.
    Uses Playwright 1.58+ ``aria_snapshot()`` which returns a YAML-formatted
    accessibility tree.  Interactive roles are parsed from the YAML lines.
    """
    store = _element_store
    store.clear()

    try:
        root = page.locator(root_selector) if root_selector else page.locator(":root")
        snapshot_text = await root.aria_snapshot()
    except Exception as exc:
        log.warning("ax_snapshot_failed", error=str(exc))
        return []

    if not snapshot_text:
        return []

    elements: list[ElementMeta] = []
    _parse_aria_snapshot(snapshot_text, elements, store)

    store.last_url = page.url
    store.last_observe_time = time.monotonic()
    return elements


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


def _parse_aria_snapshot(
    text: str,
    out: list[ElementMeta],
    store: ElementStore,
) -> None:
    """Parse Playwright aria_snapshot YAML output into ElementMeta entries.

    Scans ALL lines regardless of indent depth, catching nested elements
    inside tables, comboboxes, dialogs, etc.
    """
    for line in text.splitlines():
        m = _ARIA_LINE_RE.match(line)
        if not m:
            continue
        role = m.group(1)
        name = m.group(2) or ""
        if role in INTERACTIVE_ROLES:
            ref = store.next_ref()
            selectors = _build_selectors(role, name, {})
            meta = ElementMeta(ref=ref, role=role, name=name, selectors=selectors)
            store.put(meta)
            out.append(meta)


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


async def find_element_locator(page: Any, ref: str) -> Any | None:
    """Resolve an element ref to a Playwright Locator using multi-selector strategy.

    Tries selectors in confidence-descending order. Returns the first
    locator that resolves to exactly one visible element, or ``None``.
    """
    store = _element_store
    meta = store.get(ref)
    if meta is None:
        return None

    sorted_selectors = sorted(meta.selectors, key=lambda s: s["confidence"], reverse=True)

    for sel in sorted_selectors:
        try:
            locator = _selector_to_locator(page, sel)
            if locator is None:
                continue
            count = await locator.count()
            if count == 1:
                return locator
            if count > 1:
                first = locator.first
                try:
                    if await first.is_visible():
                        return first
                except Exception:
                    pass
        except Exception:
            continue

    return None


def _selector_to_locator(page: Any, sel: dict[str, Any]) -> Any | None:
    """Convert a selector dict to a Playwright Locator."""
    sel_type = sel["type"]
    value = sel["value"]

    if sel_type == "role":
        return page.get_by_role(sel["role"], name=value)
    if sel_type == "text":
        return page.get_by_text(value, exact=False)
    if sel_type == "label":
        return page.get_by_label(value)
    if sel_type == "placeholder":
        return page.get_by_placeholder(value)
    if sel_type == "testid":
        return page.get_by_test_id(value)
    if sel_type == "css":
        return page.locator(value)
    return None


# 4. Self-healing element find

async def self_healing_find(page: Any, ref: str) -> Any | None:
    """Find an element with automatic recovery on stale references.

    Three-phase strategy (matching TS selfHealingFind):
    1. Try current element store.
    2. If URL changed or >5s since last observe, re-extract and retry.
    3. Similarity match - find element with same role and partial name.
    """
    store = _element_store

    # Phase 1: direct lookup in current store.
    locator = await find_element_locator(page, ref)
    if locator is not None:
        return locator

    original_meta = store.get(ref)

    # Phase 2: re-extract if page likely changed.
    current_url = page.url
    elapsed = time.monotonic() - store.last_observe_time
    if current_url != store.last_url or elapsed > 5.0:
        log.debug("self_heal_re_extract", reason="url_change" if current_url != store.last_url else "stale")
        await wait_for_dom_settle(page)
        await extract_interactive_elements(page)
        locator = await find_element_locator(page, ref)
        if locator is not None:
            return locator

    # Phase 3: similarity match.
    if original_meta and original_meta.name:
        similar = store.find_similar(original_meta.role, original_meta.name)
        if similar and similar.ref != ref:
            log.debug("self_heal_similar_match", original=ref, matched=similar.ref)
            locator = await find_element_locator(page, similar.ref)
            if locator is not None:
                return locator

    log.warning("self_healing_find_failed", ref=ref)
    return None


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
