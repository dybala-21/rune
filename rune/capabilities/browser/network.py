"""CDP network monitoring for RUNE.

Captures XHR/fetch API requests made by SPA sites via Chrome DevTools
Protocol. When the browser navigates to a JavaScript-rendered site,
this monitor records the API endpoints the site calls, so the LLM can
use ``web_fetch`` to call them directly instead of clicking through UI.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_MAX_ENTRIES = 100

# URL patterns to ignore (analytics, ads, tracking).
_NOISE_RE = re.compile(
    r"google-analytics|googletagmanager|hotjar|sentry|facebook|doubleclick"
    r"|ads\.|analytics\.|tracking\.|pixel\.|beacon\."
    r"|\.(js|css|woff2?|png|jpe?g|gif|svg|ico)(\?|$)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ApiRequest:
    """A captured API request."""
    url: str
    method: str
    resource_type: str  # XHR, Fetch
    status: int = 0
    content_type: str = ""
    has_json_response: bool = False
    # Request body (truncated) — without it a discovered POST endpoint (the
    # schedule/booking class) cannot be replayed via web_fetch.
    post_data: str = ""
    request_content_type: str = ""
    # Captured response body (JSON responses only, truncated). Reading the
    # body the site already fetched needs no replay and no parameters — the
    # page's own click flow supplied them (network-level scraping).
    response_body: str = ""


def hybrid_api_enabled() -> bool:
    """Gate for the hybrid API path (POST replay, recipes, late-API hints)."""
    import os

    return os.environ.get("RUNE_HYBRID_API", "1") != "0"


def format_api_recipe(api: ApiRequest) -> str:
    """One replayable line for a captured API call — method, URL, body."""
    if api.method != "GET" and api.post_data:
        ct = api.request_content_type or "application/x-www-form-urlencoded"
        return (
            f'web_fetch(url="{api.url}", method="{api.method}", '
            f'body="{api.post_data[:200]}", content_type="{ct}")'
        )
    return f'web_fetch(url="{api.url}")'


class NetworkMonitor:
    """CDP-based network request monitor for API endpoint discovery."""

    def __init__(self) -> None:
        self._cdp: Any = None
        self._requests: deque[ApiRequest] = deque(maxlen=_MAX_ENTRIES)
        self._pending: dict[str, ApiRequest] = {}  # requestId -> ApiRequest
        self._active: bool = False
        self._reported_count: int = 0
        # requestId → ApiRequest, awaiting body capture on loadingFinished.
        self._await_body: dict[str, ApiRequest] = {}

    @property
    def active(self) -> bool:
        return self._active

    async def attach(self, page: Any) -> None:
        """Enable CDP Network domain and start capturing requests."""
        if self._active:
            return
        try:
            self._cdp = await page.context.new_cdp_session(page)
            self._cdp.on("Network.requestWillBeSent", self._on_request)
            self._cdp.on("Network.responseReceived", self._on_response)
            self._cdp.on("Network.loadingFinished", self._on_loading_finished)
            await self._cdp.send("Network.enable")
            self._active = True
            log.debug("network_monitor_attached")
        except Exception as exc:
            log.debug("network_monitor_attach_failed", error=str(exc))
            self._cdp = None

    async def detach(self) -> None:
        """Stop monitoring and clean up CDP session."""
        if not self._active or self._cdp is None:
            return
        try:
            await self._cdp.send("Network.disable")
            await self._cdp.detach()
        except Exception:
            pass
        self._cdp = None
        self._active = False
        log.debug("network_monitor_detached")

    def _on_request(self, params: dict) -> None:
        """Handle Network.requestWillBeSent event."""
        resource_type = params.get("type", "")
        if resource_type not in ("XHR", "Fetch"):
            return

        request = params.get("request", {})
        url = request.get("url", "")

        if not url or _NOISE_RE.search(url):
            return

        request_id = params.get("requestId", "")
        req_headers = request.get("headers", {}) or {}
        api = ApiRequest(
            url=url,
            method=request.get("method", "GET"),
            resource_type=resource_type,
            post_data=str(request.get("postData", "") or "")[:500],
            request_content_type=str(
                req_headers.get("Content-Type") or req_headers.get("content-type") or ""
            ),
        )
        self._pending[request_id] = api

    def _on_response(self, params: dict) -> None:
        """Handle Network.responseReceived event."""
        request_id = params.get("requestId", "")
        api = self._pending.pop(request_id, None)
        if api is None:
            return

        response = params.get("response", {})
        api.status = response.get("status", 0)
        api.content_type = response.get("mimeType", "")
        api.has_json_response = "json" in api.content_type.lower()

        # Body becomes readable at loadingFinished; only JSON payloads are
        # worth the fetch (the data the page's own click flow requested).
        if api.has_json_response and hybrid_api_enabled():
            self._await_body[request_id] = api

        self._requests.append(api)
        log.debug(
            "network_api_captured",
            method=api.method, url=api.url[:100],
            status=api.status, json=api.has_json_response,
        )

    _BODY_MAX_CHARS = 30_000

    def _on_loading_finished(self, params: dict) -> None:
        """Schedule the async body read for a finished JSON response."""
        api = self._await_body.pop(params.get("requestId", ""), None)
        if api is None or self._cdp is None:
            return
        import asyncio

        try:
            asyncio.get_running_loop().create_task(
                self._fetch_body(params.get("requestId", ""), api)
            )
        except RuntimeError:
            pass  # no running loop (unit tests) — bodies just stay empty

    async def _fetch_body(self, request_id: str, api: ApiRequest) -> None:
        """Read the response body via CDP; failures leave the body empty."""
        try:
            result = await self._cdp.send(
                "Network.getResponseBody", {"requestId": request_id}
            )
            body = result.get("body", "")
            if result.get("base64Encoded"):
                import base64

                body = base64.b64decode(body).decode("utf-8", errors="replace")
            api.response_body = body[: self._BODY_MAX_CHARS]
            log.debug(
                "network_body_captured", url=api.url[:100], chars=len(api.response_body)
            )
        except Exception as exc:
            log.debug("network_body_fetch_failed", error=str(exc)[:100])

    def get_discovered_apis(self, filter_pattern: str = "") -> list[ApiRequest]:
        """Return captured API requests, optionally filtered by URL pattern."""
        results = list(self._requests)
        if filter_pattern:
            pattern = filter_pattern.lower()
            results = [r for r in results if pattern in r.url.lower()]
        return results

    def get_json_apis(self) -> list[ApiRequest]:
        """Return only requests that returned JSON responses."""
        return [r for r in self._requests if r.has_json_response]

    def mark_reported(self) -> None:
        """Remember how many requests the model has already been shown."""
        self._reported_count = len(self._requests)

    def unreported_interesting_count(self) -> int:
        """JSON/POST calls captured since the last report — data-loading XHRs
        (the schedule/booking class) fire on interaction, after the navigate-
        time snapshot the model saw."""
        skip = min(self._reported_count, len(self._requests))
        return sum(
            1 for i, r in enumerate(self._requests)
            if i >= skip and (r.has_json_response or r.method != "GET")
        )

    def clear(self) -> None:
        """Clear captured requests."""
        self._requests.clear()
        self._pending.clear()
        self._await_body.clear()
        self._reported_count = 0


# Module-level singleton
_monitor: NetworkMonitor | None = None


def get_network_monitor() -> NetworkMonitor:
    global _monitor
    if _monitor is None:
        _monitor = NetworkMonitor()
    return _monitor
