"""List captured API requests and read their response bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rune.capabilities.browser.session import browser_operation
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


@browser_operation
async def browser_discover_apis(params: BrowserDiscoverApisParams) -> CapabilityResult:
    """List API requests captured after navigation or read a captured response."""
    from rune.capabilities.browser.network import get_network_monitor

    monitor = get_network_monitor()
    if not monitor.active:
        return CapabilityResult(
            success=True,
            output="Network monitor not active. Navigate to a page first with browser_navigate.",
            metadata={"count": 0},
        )

    from rune.capabilities.browser.network import format_api_recipe, hybrid_api_enabled

    # Reuse the captured response without replaying the request.
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
        from rune.capabilities.web import MAX_JSON_OUTPUT_CHARS, _prune_json_by_term

        pruned = _prune_json_by_term(body, params.json_filter)
        if not pruned:
            return CapabilityResult(success=False,
                                    error="No complete JSON records matched. Change jsonFilter or inspect another captured response.",
                                    metadata={"url": api.url, "filter_no_match": pruned == "", "invalid_json": pruned is None})
        if params.json_filter:
            body = f"[filtered by '{params.json_filter}'] {pruned}"
        output = f"{api.method} {api.url} [{api.status}]\n{body}"
        if len(output) > MAX_JSON_OUTPUT_CHARS:
            return CapabilityResult(success=False, error="The captured response is too large. Narrow jsonFilter to read complete records.",
                                    metadata={"url": api.url, "delivery_truncated": True})
        return CapabilityResult(
            success=True,
            output=output,
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

    # Show up to eight requests, prioritizing JSON responses and POSTs.
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
