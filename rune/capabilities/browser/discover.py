"""API discovery over the CDP network log.

Split from capabilities.py, which this tool's arrival had pushed past the
project's module-size line. Everything here is about one job: after a page
loads, list the XHR/fetch calls it made and read their captured bodies, so
data can come off the wire instead of out of the UI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


class BrowserDiscoverApisParams(BaseModel):
    filter: str = Field(
        default="",
        description="URL pattern to filter (e.g., 'search', 'api/v1', 'hotel')",
    )
    read_body: str = Field(
        default="",
        alias="readBody",
        description=(
            "URL substring of a captured call whose JSON response body to "
            "read — no replay or parameters needed, the page's own request "
            "already carried them"
        ),
    )
    json_filter: str = Field(
        default="",
        alias="jsonFilter",
        description=(
            "With readBody: keep only JSON subtrees containing this keyword "
            "(e.g. a branch name) so large payloads stay readable"
        ),
    )


async def browser_discover_apis(params: BrowserDiscoverApisParams) -> CapabilityResult:
    """List API endpoints discovered via CDP network monitoring.

    After browser_navigate loads a page, this tool shows the XHR/fetch
    requests the site made. Use web_fetch to call these APIs directly
    instead of clicking through the UI.
    """
    from rune.capabilities.browser.network import get_network_monitor

    monitor = get_network_monitor()
    if not monitor.active:
        return CapabilityResult(
            success=True,
            output="Network monitor not active. Navigate to a page first with browser_navigate.",
            metadata={"count": 0},
        )

    from rune.capabilities.browser.network import format_api_recipe, hybrid_api_enabled

    # Read a captured response body — the highest-value path: the site's own
    # click flow already supplied the parameters, so there is nothing to
    # reconstruct or replay (network-level scraping).
    if params.read_body and hybrid_api_enabled():
        matches = [
            a for a in monitor.get_discovered_apis(params.read_body)
            if a.response_body
        ]
        if not matches:
            return CapabilityResult(
                success=False,
                error=(
                    f"No captured body matches '{params.read_body}'. "
                    "List calls first (browser_discover_apis) — bodies exist "
                    "only for JSON responses seen after the monitor attached."
                ),
            )
        api = matches[-1]  # newest matching call
        body = api.response_body
        if params.json_filter:
            from rune.capabilities.web import _prune_json_by_term

            pruned = _prune_json_by_term(body, params.json_filter)
            if pruned is not None:
                body = f"[filtered by '{params.json_filter}'] {pruned}"
        return CapabilityResult(
            success=True,
            output=f"{api.method} {api.url} [{api.status}]\n{body[:30_000]}",
            metadata={"url": api.url, "chars": len(body)},
        )

    apis = monitor.get_discovered_apis(params.filter)
    if not apis:
        hint = f" matching '{params.filter}'" if params.filter else ""
        return CapabilityResult(
            success=True,
            output=f"No API endpoints discovered{hint}. Try interacting with the page (scroll, click) to trigger more requests.",
            metadata={"count": 0},
        )

    # Data-carrying calls first: JSON responses, then POSTs (booking/schedule
    # endpoints), then the rest — bounded so the recipe list stays cheap.
    ranked = sorted(
        apis, key=lambda a: (not a.has_json_response, a.method == "GET"),
    )[:8]
    lines = [f"Discovered {len(apis)} API endpoint(s) (top {len(ranked)} shown):"]
    for api in ranked:
        json_tag = " [JSON]" if api.has_json_response else ""
        body_tag = (
            f" [body captured: {len(api.response_body)} chars]"
            if api.response_body else ""
        )
        lines.append(f"  {api.method} {api.url} [{api.status}]{json_tag}{body_tag}")
        if hybrid_api_enabled() and not api.response_body:
            lines.append(f"    replay: {format_api_recipe(api)}")
    lines.append("")
    lines.append(
        "PREFER captured bodies: browser_discover_apis(readBody='<url part>', "
        "jsonFilter='<keyword>') reads the data the page already fetched — "
        "no parameters needed. Replay with web_fetch only when no body was "
        "captured."
        if hybrid_api_enabled()
        else "Use web_fetch(url=...) to call these APIs directly."
    )
    monitor.mark_reported()

    return CapabilityResult(
        success=True,
        output="\n".join(lines),
        metadata={"count": len(apis)},
    )

# Registration
