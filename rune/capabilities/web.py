"""Web capabilities for RUNE.

Ported from src/capabilities/web.ts - web search and fetch with
pluggable search providers (DuckDuckGo default, Brave opt-in) and
HTML-to-text conversion.
"""

from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.search.duckduckgo import DuckDuckGoSearchProvider
from rune.capabilities.search.provider import SearchOptions, SearchProvider, SearchResult
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Search provider chain

_MAX_CONSECUTIVE_FAILURES = 3


class _AutoSearchProvider(SearchProvider):
    """Fallback chain: tries providers in order, skips after repeated failures."""

    def __init__(self, providers: list[SearchProvider]) -> None:
        self._providers = providers
        self._failure_counts: dict[str, int] = {}

    @property
    def name(self) -> str:  # noqa: D102
        return "auto"

    async def search(self, options: SearchOptions) -> list[SearchResult]:
        last_error: Exception | None = None
        for provider in self._providers:
            if self._failure_counts.get(provider.name, 0) >= _MAX_CONSECUTIVE_FAILURES:
                log.debug("provider_skipped_consecutive_failures", provider=provider.name)
                continue
            try:
                results = await provider.search(options)
                if results:
                    self._failure_counts[provider.name] = 0
                    return results
                # Empty results - not a failure, but try next provider
            except Exception as exc:
                count = self._failure_counts.get(provider.name, 0) + 1
                self._failure_counts[provider.name] = count
                log.debug(
                    "search_provider_failed",
                    provider=provider.name,
                    error=str(exc),
                    consecutive_failures=count,
                )
                last_error = exc
        if last_error:
            log.debug("all_search_providers_failed", last_error=str(last_error))
        return []


def build_search_provider(
    provider_config: str = "auto",
    page_pool: Any | None = None,
) -> SearchProvider:
    """Build a search provider based on configuration.

    Args:
        provider_config: "auto", "duckduckgo", "brave", or "browser".
        page_pool: Optional browser page pool for browser-based fallback.
    """
    if provider_config == "duckduckgo":
        return DuckDuckGoSearchProvider()

    if provider_config == "brave":
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            log.warning("brave_api_key_missing_falling_back_to_duckduckgo")
            return DuckDuckGoSearchProvider()
        from rune.capabilities.search.brave import BraveSearchProvider
        return BraveSearchProvider(api_key)

    if provider_config == "browser":
        if page_pool is None:
            log.warning("page_pool_missing_falling_back_to_duckduckgo")
            return DuckDuckGoSearchProvider()
        from rune.capabilities.search.browser_search import BrowserSearchProvider
        return BrowserSearchProvider(page_pool)

    # "auto" - build fallback chain: DDG -> Brave (if key) -> Browser (if pool)
    providers: list[SearchProvider] = [DuckDuckGoSearchProvider()]

    api_key = os.environ.get("BRAVE_API_KEY", "")
    if api_key:
        from rune.capabilities.search.brave import BraveSearchProvider
        providers.insert(0, BraveSearchProvider(api_key))

    if page_pool is not None:
        from rune.capabilities.search.browser_search import BrowserSearchProvider
        providers.append(BrowserSearchProvider(page_pool))

    if len(providers) == 1:
        return providers[0]
    return _AutoSearchProvider(providers)


# Module-level provider instance - lazily initialised on first search.
_search_provider: SearchProvider | None = None


def get_search_provider() -> SearchProvider:
    """Return the module-level search provider, creating it on first use."""
    global _search_provider  # noqa: PLW0603
    if _search_provider is None:
        _search_provider = build_search_provider()
    return _search_provider


def set_search_provider(provider: SearchProvider) -> None:
    """Override the module-level search provider (called during init)."""
    global _search_provider  # noqa: PLW0603
    _search_provider = provider


# Parameter schemas

class WebSearchParams(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=10, alias="maxResults")
    language: str = Field(default="en")
    freshness: str = Field(default="", description="Time filter: day, week, month, year")
    site: str = Field(default="", description="Restrict search to a specific site/domain")


class WebFetchParams(BaseModel):
    url: str = Field(description="URL to fetch")
    selector: str | None = Field(default=None, description="CSS selector to extract")
    max_length: int = Field(default=50_000, alias="maxLength")
    # Replay support for APIs discovered via browser network monitoring —
    # data-loading endpoints on dynamic sites are frequently POST.
    method: str = Field(default="GET", description="HTTP method (GET or POST)")
    body: str | None = Field(
        default=None,
        description="Request body for POST (as captured by browser_discover_apis)",
    )
    content_type: str | None = Field(
        default=None,
        alias="contentType",
        description="Request body content type (default form-urlencoded)",
    )
    json_filter: str | None = Field(
        default=None,
        alias="jsonFilter",
        description=(
            "For large JSON responses: keep only array elements / subtrees "
            "containing this keyword (e.g. a branch or product name), so the "
            "relevant records survive truncation"
        ),
    )


# HTML helpers

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Convert HTML to lightweight Markdown, preserving structure.

    Keeps headings, links, and list items as Markdown instead of
    stripping all tags to plain text.  This helps the LLM understand
    page structure (navigation, sections, links).
    """
    text = _SCRIPT_STYLE_RE.sub("", html)
    # Structural tags → Markdown (before generic tag strip)
    text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h[3-6][^>]*>(.*?)</h[3-6]>", r"\n### \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", text, flags=re.DOTALL)
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    # Strip remaining tags
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_by_selector(html: str, selector: str) -> str:
    """Basic CSS selector extraction supporting tag, .class, and #id.

    Falls back to full text if no match is found.
    """
    selector = selector.strip()
    pattern: str | None = None

    if selector.startswith("#"):
        # ID selector: #my-id
        id_val = re.escape(selector[1:])
        pattern = (
            rf'<[^>]+id\s*=\s*["\']?{id_val}["\']?[^>]*>'
            r"(.*?)</[^>]+>"
        )
    elif selector.startswith("."):
        # Class selector: .my-class
        cls_val = re.escape(selector[1:])
        pattern = (
            rf'<[^>]+class\s*=\s*["\'][^"\']*\b{cls_val}\b[^"\']*["\'][^>]*>'
            r"(.*?)</[^>]+>"
        )
    else:
        # Tag selector: div, p, article, etc.
        tag = re.escape(selector)
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"

    if pattern:
        matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
        if matches:
            combined = "\n".join(matches)
            return _html_to_text(combined)

    return _html_to_text(html)


# Capability implementations

async def web_search(params: WebSearchParams) -> CapabilityResult:
    """Search the web using the configured provider chain."""
    log.debug("web_search", query=params.query, max_results=params.max_results)

    options = SearchOptions(
        query=params.query,
        max_results=params.max_results,
        freshness=params.freshness or None,
        site=params.site or None,
    )

    provider = get_search_provider()
    results = await provider.search(options)

    if not results:
        return CapabilityResult(
            success=True,
            output="No search results found.",
            metadata={"query": params.query, "count": 0},
        )

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.url}")
        if r.description:
            lines.append(f"   {r.description}")
        lines.append("")

    return CapabilityResult(
        success=True,
        output="\n".join(lines).strip(),
        metadata={"query": params.query, "count": len(results)},
    )


# Session-level fetch failure tracking (reset per process)
_fetch_failures: dict[str, int] = {}


def _get_domain(url: str) -> str:
    """Extract domain from URL for failure tracking."""
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_path_wipe_redirect(original_url: str, final_url: str) -> bool:
    """Return True when the server redirected a deep path to a shallower one
    that does not share the requested prefix.

    Common case: invalid resource ID lands on the homepage or a search page.
    Canonicalization redirects (https upgrade, trailing slash, case change)
    are allowed because the path prefix is preserved.
    """
    if original_url == final_url:
        return False
    from urllib.parse import urlparse
    try:
        orig = urlparse(original_url)
        final = urlparse(final_url)
    except Exception:
        return False

    orig_segments = [p for p in orig.path.split("/") if p]
    final_segments = [p for p in final.path.split("/") if p]

    if len(orig_segments) < 2:
        return False
    if len(final_segments) >= len(orig_segments):
        return False
    # Path prefix preserved means canonicalization, not wipe.
    if orig.path.rstrip("/").startswith(final.path.rstrip("/").rstrip()):
        if final_segments and orig_segments[: len(final_segments)] == final_segments:
            return False
    return True


def _json_contains(node: Any, term: str) -> bool:
    if isinstance(node, dict):
        return any(_json_contains(v, term) for v in node.values())
    if isinstance(node, list):
        return any(_json_contains(x, term) for x in node)
    return term in str(node).lower()


def _prune_json_by_term(text: str, term: str) -> str | None:
    """Keep only the JSON subtrees containing *term* (case-insensitive).

    List elements that match are kept WHOLE — the point is preserving all
    fields of the matching records (e.g. a branch's showtimes), not just the
    matching strings. Returns None when the text isn't valid JSON.
    """
    import json as _json

    try:
        data = _json.loads(text)
    except Exception:
        return None
    t = term.lower()

    def prune(node: Any) -> Any:
        if isinstance(node, list):
            return [x for x in node if _json_contains(x, t)]
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    if _json_contains(v, t):
                        out[k] = prune(v)
                else:
                    out[k] = v
            return out
        return node

    return _json.dumps(prune(data), ensure_ascii=False)


async def web_fetch(params: WebFetchParams) -> CapabilityResult:
    """Fetch a URL and convert HTML to clean text.

    Tracks per-domain failures within a session. If a domain has failed
    2+ times, returns early with a guidance message to use search snippets.
    """
    import httpx

    domain = _get_domain(params.url)
    log.debug("web_fetch", url=params.url, selector=params.selector)

    # Check session failure history for this domain
    if domain and _fetch_failures.get(domain, 0) >= 2:
        return CapabilityResult(
            success=False,
            error=(
                f"Skipped: {domain} has failed {_fetch_failures[domain]} times this session "
                f"(likely bot-blocked or paywalled). Use search snippets for this source."
            ),
            metadata={"status_code": 0, "skipped": True},
        )

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RUNE/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            method = (params.method or "GET").upper()
            if method != "GET":
                # Never downgrade a write to a read: the caller would believe
                # it submitted something it did not.
                if os.environ.get("RUNE_HYBRID_API", "0") != "1":
                    return CapabilityResult(
                        success=False,
                        error=(
                            f"{method} requests are disabled "
                            "(set RUNE_HYBRID_API=1 to enable API replay). "
                            "Use a GET, or interact with the page directly."
                        ),
                        metadata={"status_code": 0, "skipped": True},
                    )
                headers["Content-Type"] = (
                    params.content_type or "application/x-www-form-urlencoded"
                )
                resp = await client.request(
                    method, params.url, headers=headers, content=params.body or ""
                )
            else:
                resp = await client.get(params.url, headers=headers)
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        html = resp.text
        final_url = str(resp.url)

        # Path-wipe redirect detection. Sites commonly serve an invalid
        # deep path by redirecting to the homepage or a search page,
        # which returns 200 OK with boilerplate nav content. That looks
        # like success but is not the requested resource.
        if _is_path_wipe_redirect(params.url, final_url):
            if domain:
                _fetch_failures[domain] = _fetch_failures.get(domain, 0) + 1
            return CapabilityResult(
                success=False,
                error=(
                    f"URL {params.url} redirected to {final_url}. "
                    "The requested path was lost, target likely does not exist."
                ),
                metadata={
                    "url": params.url,
                    "final_url": final_url,
                    "redirected": True,
                    "path_wiped": True,
                    "status_code": resp.status_code,
                },
            )

        # Extract by selector if provided
        if params.selector:
            text = _extract_by_selector(html, params.selector)
        else:
            text = _html_to_text(html)

        # JS-rendered site detection: HTTP 200 but minimal useful text.
        # Sites like yanolja.com return valid HTML with JS bundles but
        # no readable content after tag stripping.
        if len(text.strip()) < 500:
            if domain:
                _fetch_failures[domain] = _fetch_failures.get(domain, 0) + 1
            return CapabilityResult(
                success=False,
                error=(
                    f"Page returned minimal content ({len(text.strip())} chars) — "
                    f"likely JavaScript-rendered or empty. "
                    f"Use web_search snippets or browser_navigate instead."
                ),
                metadata={"status_code": resp.status_code, "js_rendered": True},
            )

        # Large JSON: blind truncation drops the relevant records (a schedule
        # API can return every branch in one 500KB payload). Prune to the
        # subtrees matching the caller's keyword before truncating.
        filter_note = ""
        if params.json_filter and "json" in content_type.lower():
            pruned = _prune_json_by_term(text, params.json_filter)
            if pruned is not None:
                text = pruned
                filter_note = f"[filtered by '{params.json_filter}'] "

        # Truncate if too long
        truncated = False
        if len(text) > params.max_length:
            text = text[: params.max_length]
            truncated = True
            if "json" in content_type.lower() and not params.json_filter:
                text += (
                    "\n\n[TRUNCATED JSON — retry with "
                    'jsonFilter="<keyword>" to keep only the relevant records]'
                )
        if filter_note:
            text = filter_note + text

        # Success: reset failure count for this domain
        if domain:
            _fetch_failures.pop(domain, None)

        return CapabilityResult(
            success=True,
            output=text,
            metadata={
                "url": params.url,
                "final_url": final_url,
                "redirected": final_url != params.url,
                "content_type": content_type,
                "length": len(text),
                "truncated": truncated,
                "status_code": resp.status_code,
            },
        )

    except httpx.HTTPStatusError as exc:
        if domain:
            _fetch_failures[domain] = _fetch_failures.get(domain, 0) + 1
        return CapabilityResult(
            success=False,
            error=f"HTTP {exc.response.status_code}: {params.url}",
            metadata={"status_code": exc.response.status_code},
        )
    except httpx.TimeoutException:
        if domain:
            _fetch_failures[domain] = _fetch_failures.get(domain, 0) + 1
        return CapabilityResult(
            success=False,
            error=f"Timeout fetching {params.url}",
        )
    except Exception as exc:
        if domain:
            _fetch_failures[domain] = _fetch_failures.get(domain, 0) + 1
        return CapabilityResult(
            success=False,
            error=f"Fetch failed: {exc}",
        )


# Registration

def register_web_capabilities(registry: CapabilityRegistry) -> None:
    """Register all web capabilities."""
    registry.register(CapabilityDefinition(
        name="web_search",
        description="Search the web",
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=WebSearchParams,
        execute=web_search,
    ))
    registry.register(CapabilityDefinition(
        name="web_fetch",
        description=(
            "Fetch a URL and convert it to text. The first choice for reading a "
            "page, a documented API endpoint, or a .json/.txt/.md file — reach "
            "for the browser only when the page must be interacted with. "
            "Supports method='POST' with body/contentType to replay API calls "
            "found by browser_discover_apis, and jsonFilter to keep only the "
            "matching records of a large JSON response."
        ),
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=WebFetchParams,
        execute=web_fetch,
    ))
