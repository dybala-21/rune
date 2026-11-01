"""Wait for observed NOL placeholders, independently of network idleness."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

_PLACEHOLDERS_GONE = """() => ![...document.querySelectorAll('.animate-pulse')].some(el =>
    el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden')"""


async def wait_for_stay_content(page: Any, *, timeout_ms: int = 12_000) -> None:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    parts = urlsplit(page.url)
    if parts.hostname != "nol.yanolja.com" or not parts.path.startswith("/stay/"):
        return
    # NOL's room skeleton remained visible after DOM quiet; the live probe loaded at ~6s.
    # This only establishes readability. The ledger still checks every required quote.
    try:
        handle = await page.wait_for_function(_PLACEHOLDERS_GONE, timeout=timeout_ms)
        await handle.dispose()
    except PlaywrightTimeoutError as exc:
        raise ValueError("NOL content is still loading or unavailable; no evidence was captured") from exc
