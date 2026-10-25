"""Search the web and fetch page content or API responses."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from html import unescape
from typing import Any, Literal
from urllib.parse import urldefrag, urljoin, urlsplit

from pydantic import BaseModel, Field

from rune.capabilities.fetch_state import current_fetch_state
from rune.capabilities.json_records import RecordQuery, query_records, record_paths
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
    freshness: Literal["", "day", "week", "month", "year"] = Field(default="", description="Filter publication dates: day, week, month, year. Omit when finding live data pages; check their observation times instead.")
    site: str = Field(default="", description="Restrict search to a specific site/domain")


class WebFetchParams(BaseModel):
    url: str = Field(description="URL to fetch")
    selector: str | None = Field(default=None, description="CSS selector; defaults to main content when available. Use body for the full page.")
    max_length: int = Field(default=50_000, alias="maxLength", ge=1)
    # Some discovered APIs use POST to load data.
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
    record_query: RecordQuery | None = Field(default=None, alias="recordQuery",
        description="JSON record selection/aggregation; cannot combine with jsonFilter.")


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
    # Ignore markup without readable text.
    if not unescape(_TAG_RE.sub(" ", text)).strip():
        return ""
    # Structural tags → Markdown (before generic tag strip)
    text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h[3-6][^>]*>(.*?)</h[3-6]>", r"\n### \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", text, flags=re.DOTALL)
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    text = _WS_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_by_selector(html: str, selector: str) -> str:
    """Return text from the complete subtrees matched by a CSS selector."""
    from cssselect import SelectorError
    from lxml import etree
    from lxml import html as document

    try:
        tree = document.fromstring(
            html.encode("utf-8"), parser=document.HTMLParser(encoding="utf-8", no_network=True),
        )
        matches = tree.cssselect(selector.strip())
    except (SelectorError, etree.ParserError) as exc:
        raise ValueError(f"Cannot extract the CSS selector: {exc}") from exc
    if not matches:
        raise ValueError("No elements match the CSS selector.")
    return _html_to_text("\n".join(
        document.tostring(node, encoding="unicode", with_tail=False) for node in matches
    ))


def _extract_page_content(html: str) -> tuple[str, str]:
    """Prefer main content so navigation does not fill the text limit."""
    from lxml import etree
    from lxml import html as document

    if not html.strip():
        return "", "body"
    try:
        tree = document.document_fromstring(
            html.encode("utf-8"), parser=document.HTMLParser(encoding="utf-8", no_network=True),
        )
    except etree.ParserError:
        return _html_to_text(html), "body"
    for node in tree.xpath('//head | //script | //style | //noscript | //noframes | //template | //*[@hidden or @aria-hidden="true"]'):
        if node.getparent() is not None:
            node.drop_tree()
    main_nodes = set(tree.xpath('//main | //*[@role="main"]'))
    mode = "main"
    if not main_nodes:
        main_nodes = set(tree.xpath('//article[not(ancestor::article)]'))
        mode = "article"
    roots = [node for node in tree.iter() if node in main_nodes and not any(
        ancestor in main_nodes for ancestor in node.iterancestors()
    )]
    if roots:
        text = _html_to_text("\n".join(
            document.tostring(node, encoding="unicode", with_tail=False) for node in roots
        ))
        if text:
            return text, mode
    for node in tree.xpath('//nav | //header | //footer | //aside'):
        if node.getparent() is not None:
            node.drop_tree()
    return _html_to_text(document.tostring(tree, encoding="unicode")), "body"


def _embedded_page_urls(html: str, base_url: str) -> list[str]:
    """Resolve frame URLs without executing page scripts."""
    from lxml import etree
    from lxml import html as document

    try:
        tree = document.document_fromstring(html.encode("utf-8"),
            parser=document.HTMLParser(encoding="utf-8", no_network=True))
    except etree.ParserError:
        return []
    try:
        base = urljoin(base_url, next(iter(tree.xpath('//base/@href')), ""))
    except ValueError:
        log.debug("invalid_embedded_page_base")
        base = base_url
    urls = []
    for src in tree.xpath('//frame/@src | //iframe[not(@srcdoc)]/@src'):
        if not src.strip():
            continue
        try:
            url = urldefrag(urljoin(base, src.strip()))[0]
            parsed = urlsplit(url)
        except ValueError:
            continue
        if (parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
                and url != urldefrag(base_url)[0] and len(url) <= 2048 and url not in urls):
            urls.append(url)
        if len(urls) == 20:
            break
    return urls


def _origin(url: str) -> tuple | None:
    try:
        parsed = urlsplit(url)
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return None


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

    lines = [
        "Search excerpts are indexed snapshots, not live readings. Confirm current values "
        "and their observation time on the source page or API.", "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.url}")
        if r.age:
            lines.append(f"   Page date reported by search: {r.age} (not necessarily the date of the facts)")
        if r.description:
            lines.append(f"   {r.description}")
        lines.append("")

    return CapabilityResult(
        success=True,
        output="\n".join(lines).strip(),
        metadata={
            "query": params.query, "count": len(results), "freshness": params.freshness,
            "source_kind": "search_excerpt",
        },
    )


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


MAX_JSON_OUTPUT_CHARS = 29_000


class _JSONNumber(str):
    """Keep a JSON number's original precision and notation."""


def _dump_json(node: Any) -> str:
    if isinstance(node, _JSONNumber):
        return str(node)
    if isinstance(node, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _dump_json(v) for k, v in node.items()) + "}"
    if isinstance(node, list):
        return "[" + ",".join(_dump_json(v) for v in node) + "]"
    return json.dumps(node, ensure_ascii=False)


def _json_contains(node: Any, term: str) -> bool:
    if isinstance(node, dict):
        return any(term in k.casefold() or _json_contains(v, term) for k, v in node.items())
    if isinstance(node, list):
        return any(_json_contains(x, term) for x in node)
    return term in str(node).casefold()


def _prune_json_by_term(text: str, term: str) -> str | None:
    """Keep matching records intact; return "" for no match or None for invalid JSON."""
    try:
        data = json.loads(text, parse_float=_JSONNumber, parse_int=_JSONNumber)
    except (ValueError, RecursionError):
        return None
    t = term.casefold().strip()
    if not t:
        return text
    if not _json_contains(data, t):
        return ""

    def prune(node: Any) -> Any:
        if isinstance(node, list):
            return [x for x in node if _json_contains(x, t)]
        if isinstance(node, dict):
            # Keep nested values and context when the record itself matches.
            if any(not isinstance(v, (dict, list)) and _json_contains(v, t) for v in node.values()):
                return node
            out: dict[str, Any] = {}
            for k, v in node.items():
                if t in k.casefold():
                    out[k] = v
                elif isinstance(v, dict):
                    out[k] = prune(v) if _json_contains(v, t) else v
                elif isinstance(v, list):
                    if _json_contains(v, t):
                        out[k] = prune(v)
                else:
                    out[k] = v
            return out
        return node

    return _dump_json(prune(data))


async def web_fetch(params: WebFetchParams) -> CapabilityResult:
    attempt = {"started": False}
    try:
        async with asyncio.timeout(30):
            result = await _fetch_response(params, attempt)
            if params.method.upper() == "GET" and not (params.selector or params.json_filter or params.record_query):
                origin = _origin((result.metadata or {}).get("final_url", params.url))
                path = [params.url]
                # Follow a sole same-origin frame without another model call.
                for _ in range(2):
                    metadata = result.metadata or {}
                    frames = metadata.get("embedded_pages", [])
                    if (not metadata.get("empty_content") or len(frames) != 1 or not origin
                            or _origin(frames[0]) != origin or frames[0] in path):
                        break
                    path.append(frames[0])
                    result = await _fetch_response(params.model_copy(update={"url": frames[0]}), attempt,
                                                   allowed_origin=origin)
                    result.metadata = {**(result.metadata or {}), "requested_url": params.url, "frame_path": list(path)}
    except TimeoutError:
        result = CapabilityResult(success=False, error="The page retrieval deadline was reached.")
    if (params.method or "GET").upper() not in {"GET", "HEAD"}:
        metadata = result.metadata or {}
        if metadata.get("status_code") == 202:
            result.success = False
            result.error = "The request was accepted for processing; its final result is not confirmed. Check its status before resubmitting."
        status = "completed" if result.success else "unknown" if attempt["started"] else "not_executed"
        result.metadata = {**metadata, "action_status": status}
    return result


async def _fetch_response(params: WebFetchParams, attempt: dict, *, allowed_origin: tuple | None = None) -> CapabilityResult:
    """Fetch a URL; preserve short responses and isolate transient failures."""
    import httpx

    if params.record_query and params.json_filter:
        return CapabilityResult(success=False, error="recordQuery and jsonFilter cannot be combined. Use recordQuery.where for exact field matches.")
    state = current_fetch_state() if params.method.upper() == "GET" else None
    if state is not None and (delay := state.retry_after(params.url)) > 0:
        return CapabilityResult(
            success=False,
            error=f"This URL recently failed twice. Retry in {delay:.0f} seconds or use another source.",
            metadata={"status_code": 0, "skipped": True, "retry_after_seconds": round(delay, 1)},
        )
    log.debug("web_fetch", url=params.url, selector=params.selector)

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RUNE/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=allowed_origin is None and (params.method or "GET").upper() in {"GET", "HEAD"},
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
                attempt["started"] = True
                resp = await client.request(
                    method, params.url, headers=headers, content=params.body or ""
                )
            else:
                resp = await client.get(params.url, headers=headers)
                if allowed_origin is not None:
                    for _ in range(5):
                        if not resp.is_redirect:
                            break
                        target = urljoin(str(resp.url), resp.headers.get("location", ""))
                        if _origin(target) != allowed_origin:
                            return CapabilityResult(success=False,
                                error="The embedded page redirects to another origin. Inspect that URL separately.",
                                metadata={"url": params.url, "redirect_blocked": True})
                        resp = await client.get(target, headers=headers)
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        html = resp.text
        final_url = str(resp.url)
        source = {"url": params.url, "final_url": final_url,
                  "retrieved_at": datetime.now(UTC).isoformat(), "status_code": resp.status_code,
                  "content_type": content_type}

        # A redirect to the homepage or search page can return 200 without the requested content.
        if _is_path_wipe_redirect(params.url, final_url):
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

        extraction = "selector"
        embedded_pages = []
        if params.selector:
            try:
                text = _extract_by_selector(html, params.selector)
                if not text.strip():
                    raise ValueError("The selected elements have no readable text.")
            except ValueError as exc:
                return CapabilityResult(
                    success=False,
                    error=f"{exc} Adjust the selector or omit it to read the full page.",
                    metadata={"status_code": resp.status_code, "selector_error": True},
                )
        elif "html" in content_type.lower() or (
            not content_type and html.lstrip().startswith("<")
        ):
            text, extraction = _extract_page_content(html)
            embedded_pages = _embedded_page_urls(html, final_url)
        else:
            text, extraction = html, "text"

        frame_limit = min(4000, params.max_length)
        if text.strip():
            # Frame hints must not crowd out the page body.
            spare = params.max_length - len(text) if len(text) <= params.max_length else params.max_length
            frame_limit = min(frame_limit, params.max_length // 4, spare)
        frame_note = ""
        for url in embedded_pages:
            note = (frame_note or "Embedded pages (not yet read); use web_fetch for the relevant content:\n") + url + "\n"
            if len(note) + 1 > frame_limit:
                continue
            frame_note = note
        if frame_note:
            frame_note += "\n"
        if not text.strip():
            return CapabilityResult(
                success=False,
                error=("The parent page has no readable text. " + frame_note if frame_note else
                       "The response has no readable content. Check the URL or inspect the page in a browser if needed."),
                metadata={**source, "empty_content": True,
                          **({"embedded_pages": embedded_pages} if embedded_pages else {})},
            )

        json_data = None
        if "json" in content_type.lower():
            try:
                json_data = json.loads(text)
            except (ValueError, RecursionError):
                return CapabilityResult(success=False, error="The response is not valid JSON.",
                                        metadata={**source, "invalid_json": True})
        filter_note = frame_note
        record_query = None
        if params.record_query:
            if "json" not in content_type.lower():
                return CapabilityResult(success=False, error="recordQuery requires JSON.")
            try:
                import hashlib
                original_hash = hashlib.sha256(text.encode()).hexdigest()
                text, record_query = query_records(text, params.record_query)
                record_query["input_hash"] = original_hash
                filter_note = (f"[recordQuery {params.record_query.operation}: {record_query['matched']} of "
                               f"{record_query['records']} records in this response; pagination not checked] ")
            except ValueError as exc:
                return CapabilityResult(success=False, error=str(exc) + record_paths(json_data),
                                        metadata={**source, "record_query_failed": True})
        if params.json_filter and "json" in content_type.lower():
            pruned = _prune_json_by_term(text, params.json_filter)
            if not pruned:
                return CapabilityResult(
                    success=False,
                    error=("The JSON filter matched no records. Change jsonFilter or read the unfiltered response."
                           if pruned == "" else "The response is not valid JSON; the filter could not be applied."),
                    metadata={**source, "filter_no_match": pruned == "", "invalid_json": pruned is None},
                )
            text = pruned
            filter_note = f"[filtered by '{params.json_filter}'] "

        truncated = False
        if "json" in content_type.lower():
            required_length = len(text) + len(filter_note)
            if required_length > min(params.max_length, MAX_JSON_OUTPUT_CHARS):
                hint = ("Use jsonFilter alone to inspect matching records, or recordQuery with the array path."
                        if required_length > MAX_JSON_OUTPUT_CHARS else
                        f"Use jsonFilter, recordQuery, or increase maxLength to at least {required_length}.")
                return CapabilityResult(
                    success=False,
                    error=f"The JSON response needs {required_length} characters. {hint}" + record_paths(json_data),
                    metadata={**source, "truncated": True, "required_length": required_length},
                )
        elif len(text) + len(filter_note) > params.max_length:
            text = text[: max(0, params.max_length - len(filter_note))]
            truncated = True
        if not text.strip():
            return CapabilityResult(
                success=False,
                error="No readable content fits maxLength. Increase maxLength or select the relevant content.",
                metadata={**source, "empty_content": True, "truncated": truncated},
            )
        if filter_note:
            text = filter_note + text

        if state is not None:
            state.succeeded(params.url)

        return CapabilityResult(
            success=True,
            output=text,
            metadata={
                **source,
                "redirected": final_url != params.url,
                "extraction": extraction,
                "length": len(text),
                "body_offset": len(filter_note),
                "truncated": truncated,
                **({"embedded_pages": embedded_pages} if embedded_pages else {}),
                **({"record_query": record_query} if record_query is not None else {}),
            },
        )

    except httpx.HTTPStatusError as exc:
        if state is not None and (exc.response.status_code == 429 or exc.response.status_code >= 500):
            state.failed(params.url)
        return CapabilityResult(
            success=False,
            error=f"HTTP {exc.response.status_code}: {params.url}",
            metadata={"status_code": exc.response.status_code},
        )
    except httpx.TimeoutException:
        if state is not None:
            state.failed(params.url)
        return CapabilityResult(
            success=False,
            error=f"Timeout fetching {params.url}",
        )
    except Exception as exc:
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
            "matching records of a large JSON response. recordQuery selects latest records or computes aggregates "
            "within the retrieved response; it does not prove that the response covers all pages or is current."
        ),
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=WebFetchParams,
        execute=web_fetch,
    ))
