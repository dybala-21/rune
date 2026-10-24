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
_MAX_PAYLOAD = 64 * 1024
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# 503 is refused before taking effect; other 5xx can follow a committed write.
_REFUSED = frozenset({503})


def _refused(status: int) -> bool:
    return 400 <= status < 500 or status in _REFUSED


def _reports_failure(body: str) -> bool:
    """A JSON answer that declares failure despite a 2xx status."""
    import json

    try:
        value = json.loads(body)
    except ValueError:
        return False
    if not isinstance(value, dict):
        return False
    errors = value.get("errors")
    return bool(errors) and isinstance(errors, list) or value.get("success") is False or value.get("ok") is False


def _leaves(value) -> list[str]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaves(item)]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    return [] if value is None else [str(value)]

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
    response_truncated: bool = False
    # First redirect status, 0 when not redirected.
    redirect_status: int = 0
    # Send order; receipts only consider requests sent after an action began.
    seq: int = 0
    # Used to match submitted fields; never included in receipts.
    payload: str = ""
    # "" when no body is needed, else "awaiting" until a write's JSON body is read, then "done".
    body_state: str = ""


def hybrid_api_enabled() -> bool:
    """Gate for the hybrid API path (POST replay, recipes, late-API hints).

    OPT-IN (RUNE_HYBRID_API=1): the paired A/B on the discriminative
    booking-SPA cell measured the directive hints steering the model into
    API exploration that fails to converge within the round budget (ON 0/3
    vs OFF 2/3), so the default stays off until the design converts.
    """
    import os

    return os.environ.get("RUNE_HYBRID_API", "0") == "1"


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
        self.document_sequence = 0
        self.documents: list[dict] = []
        self._cached_requests: deque[str] = deque(maxlen=64)
        self._sent = 0

    def _on_cached(self, params: dict) -> None:
        request_id = params.get("requestId", "")
        self._cached_requests.append(request_id)
        for document in self.documents:
            if document["id"] == request_id:
                document["cached"] = True

    def fresh_document(self, url: str, after: int) -> bool:
        matches = [document for document in self.documents if document["url"] == url and document["sequence"] > after]
        return len(matches) == 1 and not matches[0]["cached"]

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
            self._cdp.on("Network.requestServedFromCache", self._on_cached)
            self._cdp.on("Network.loadingFinished", self._on_loading_finished)
            self._cdp.on("Network.loadingFailed", self._on_loading_failed)
            await self._cdp.send("Network.enable")
            self._active = True
            log.debug("network_monitor_attached")
        except Exception as exc:
            log.debug("network_monitor_attach_failed", error=str(exc))
            self._cdp = None

    async def retarget(self, page: Any) -> None:
        """Follow the browser to another tab; requests in flight on the old tab end as unknown."""
        await self.detach()
        for api in self._pending.values():
            api.status = 0
            self._requests.append(api)
        self._pending.clear()
        for api in self._await_body.values():
            api.body_state = "done" if api.body_state else ""
        self._await_body.clear()
        await self.attach(page)

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
        request = params.get("request", {})
        tracked = self._pending.get(params.get("requestId", ""))
        if tracked is not None and isinstance(params.get("redirectResponse"), dict):
            # A redirect reuses the request id; keep the original request.
            tracked.redirect_status = tracked.redirect_status or int(params["redirectResponse"].get("status") or 0)
            return
        # A full-page form submission is a write too; plain page loads are not.
        if resource_type not in ("XHR", "Fetch") and not (
                resource_type == "Document" and request.get("method", "GET") != "GET"):
            return

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
        self._sent += 1
        api.seq = self._sent
        if api.method not in ("GET", "HEAD", "OPTIONS"):
            api.payload = str(request.get("postData", "") or "")[:_MAX_PAYLOAD]
        self._pending[request_id] = api

    def _on_response(self, params: dict) -> None:
        """Handle Network.responseReceived event."""
        request_id = params.get("requestId", "")
        if params.get("type") == "Document":
            response = params.get("response", {})
            self.document_sequence += 1
            self.documents.append({"id": request_id, "url": response.get("url"), "sequence": self.document_sequence,
                                   "cached": request_id in self._cached_requests or any(response.get(key) for key in
                                       ("fromDiskCache", "fromServiceWorker", "fromPrefetchCache"))})
            self.documents[:] = self.documents[-32:]
        api = self._pending.pop(request_id, None)
        if api is None:
            return

        response = params.get("response", {})
        api.status = response.get("status", 0)
        api.content_type = response.get("mimeType", "")
        api.has_json_response = "json" in api.content_type.lower()

        # Body becomes readable at loadingFinished; only JSON payloads are
        # worth the fetch (the data the page's own click flow requested).
        if api.has_json_response and (hybrid_api_enabled() or api.method not in _SAFE_METHODS):
            # A write's JSON body can report failure under a 2xx status.
            if api.method not in _SAFE_METHODS:
                api.body_state = "awaiting"
            self._await_body[request_id] = api

        self._requests.append(api)
        log.debug(
            "network_api_captured",
            method=api.method, url=api.url[:100],
            status=api.status, json=api.has_json_response,
        )

    _BODY_MAX_CHARS = 30_000

    def _on_loading_failed(self, params: dict) -> None:
        """Finish a request that got no response with an unknown outcome."""
        request_id = params.get("requestId", "")
        self._await_body.pop(request_id, None)
        api = self._pending.pop(request_id, None)
        if api is not None:
            api.status = 0
            self._requests.append(api)

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
            # No running loop (unit tests) — bodies just stay empty.
            log.debug("body_fetch_skipped_no_loop",
                      request_id=params.get("requestId", ""))

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
            api.response_truncated = len(body) > self._BODY_MAX_CHARS
            log.debug(
                "network_body_captured", url=api.url[:100], chars=len(api.response_body)
            )
        except Exception as exc:
            log.debug("network_body_fetch_failed", error=str(exc)[:100])
        finally:
            if api.body_state:
                api.body_state = "done"

    def get_discovered_apis(self, filter_pattern: str = "") -> list[ApiRequest]:
        """Return captured API requests, optionally filtered by URL pattern."""
        results = [r for r in self._requests if r.resource_type != "Document"]
        if filter_pattern:
            pattern = filter_pattern.lower()
            results = [r for r in results if pattern in r.url.lower()]
        return results

    def get_json_apis(self) -> list[ApiRequest]:
        """Return only requests that returned JSON responses."""
        return [r for r in self._requests if r.has_json_response]

    def request_count(self) -> int:
        """A mark for write_receipt: requests seen after it belong to the next action."""
        return self._sent

    async def write_receipt(self, mark: int, wait: float = 3.0) -> dict | None:
        """Describe the writes sent since *mark*: accepted, rejected (all 4xx or 503) or unknown; None if none."""
        import asyncio

        safe = _SAFE_METHODS
        deadline = asyncio.get_running_loop().time() + wait

        def pending() -> bool:
            return (any(r.seq > mark and r.method not in safe for r in self._pending.values())
                    or any(r.seq > mark and r.body_state == "awaiting" for r in self._requests))

        while pending() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(.05)
        writes = [r for r in self._requests if r.seq > mark and r.method not in safe]
        still_pending = pending()
        if not writes and not still_pending:
            return None

        def outcome(r: ApiRequest) -> str:
            failed = r.redirect_status if r.redirect_status >= 400 else r.status
            if failed >= 400:
                return "rejected" if _refused(failed) else "unknown"
            # Of redirects only 303 confirms a write; 202 only queues it.
            if 200 <= r.status < 300 and r.status != 202 and r.redirect_status in (0, 303):
                if r.body_state == "awaiting" or r.body_state == "done" and _reports_failure(r.response_body):
                    return "unknown"
                return "accepted"
            return "unknown"

        outcomes = {outcome(r) for r in writes}
        status = "unknown" if still_pending or not writes or len(outcomes) != 1 else outcomes.pop()
        return {"status": status, "seqs": [r.seq for r in writes], "requests": [{"method": r.method, "url": r.url[:200], "status": r.status,
                                                **({"redirect": r.redirect_status} if r.redirect_status else {})}
                                               for r in writes[-8:]]}

    def submitted_fields(self, seqs: list[int]) -> list[list[str]]:
        """Field values each given write sent, parsed from its JSON or form body; unparsable bodies give none."""
        import json
        from urllib.parse import parse_qsl

        wanted = set(seqs)
        sent = []
        for request in self._requests:
            if request.seq not in wanted or not request.payload:
                continue
            body = request.payload
            try:
                fields = _leaves(json.loads(body))
            except ValueError:
                fields = [value for _, value in parse_qsl(body, keep_blank_values=True)] if "=" in body else []
            # Browsers submit textarea line breaks as CRLF.
            sent.append([value.replace("\r\n", "\n") for value in fields])
        return sent

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
            if i >= skip and r.resource_type != "Document" and (r.has_json_response or r.method != "GET")
        )

    def unreported_json_bodies(self) -> list[ApiRequest]:
        """Captured JSON responses the model has not been shown yet.

        The data the page loaded in response to its own click — the answer
        itself, not a pointer to where the answer might be.
        """
        skip = min(self._reported_count, len(self._requests))
        return [
            r for i, r in enumerate(self._requests)
            if i >= skip and r.response_body
        ]

    def clear(self) -> None:
        """Clear captured requests."""
        self._requests.clear()
        self._pending.clear()
        self._await_body.clear()
        self._reported_count = 0


# Module-level singleton
def get_network_monitor() -> NetworkMonitor:
    from rune.capabilities.browser.session import current_session

    session = current_session()
    if session.monitor is None:
        session.monitor = NetworkMonitor()
    return session.monitor
