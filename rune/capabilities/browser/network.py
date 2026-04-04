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


class NetworkMonitor:
    """CDP-based network request monitor for API endpoint discovery."""

    def __init__(self) -> None:
        self._cdp: Any = None
        self._requests: deque[ApiRequest] = deque(maxlen=_MAX_ENTRIES)
        self._pending: dict[str, ApiRequest] = {}  # requestId -> ApiRequest
        self._active: bool = False

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
        api = ApiRequest(
            url=url,
            method=request.get("method", "GET"),
            resource_type=resource_type,
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

        self._requests.append(api)
        log.debug(
            "network_api_captured",
            method=api.method, url=api.url[:100],
            status=api.status, json=api.has_json_response,
        )

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

    def clear(self) -> None:
        """Clear captured requests."""
        self._requests.clear()
        self._pending.clear()


# Module-level singleton
_monitor: NetworkMonitor | None = None


def get_network_monitor() -> NetworkMonitor:
    global _monitor
    if _monitor is None:
        _monitor = NetworkMonitor()
    return _monitor
