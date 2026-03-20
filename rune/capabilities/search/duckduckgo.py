"""DuckDuckGo Lite search provider for RUNE.

Ported from src/capabilities/search/duckduckgo.ts - no API key required.
Parses DuckDuckGo Lite HTML to extract search results.
"""

from __future__ import annotations

import re

import httpx

from rune.capabilities.search.provider import SearchOptions, SearchProvider, SearchResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Freshness parameter mapping
_FRESHNESS_MAP: dict[str, str] = {"day": "d", "week": "w", "month": "m", "year": "y"}

# HTML parsing helpers

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES: list[tuple[str, str]] = [
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&nbsp;", " "),
]


def strip_html_tags(html: str) -> str:
    """Strip HTML tags and decode common entities."""
    text = _HTML_TAG_RE.sub("", html)
    for entity, char in _ENTITIES:
        text = text.replace(entity, char)
    return text


# Regex to extract result links (class-first and href-first variants)
_LINK_PATTERN = re.compile(
    r'<a[^>]*class="result-link"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)</a>'
    r"|"
    r'<a[^>]*href="([^"]*)"[^>]*class="result-link"[^>]*>([\s\S]*?)</a>',
    re.IGNORECASE,
)
_SNIPPET_PATTERN = re.compile(
    r'<td[^>]*class="result-snippet"[^>]*>([\s\S]*?)</td>',
    re.IGNORECASE,
)


def parse_ddg_lite_results(html: str, max_results: int) -> list[SearchResult]:
    """Parse DuckDuckGo Lite HTML into search results."""
    if "result-link" not in html and "result-snippet" not in html:
        log.debug("ddg_lite_no_markers")
        return []

    links: list[tuple[str, str]] = []
    for m in _LINK_PATTERN.finditer(html):
        url = m.group(1) or m.group(3)
        title = m.group(2) or m.group(4)
        if url and title:
            links.append((url, strip_html_tags(title).strip()))

    snippets: list[str] = [
        strip_html_tags(m.group(1)).strip() for m in _SNIPPET_PATTERN.finditer(html)
    ]

    results: list[SearchResult] = []
    for i, (url, title) in enumerate(links):
        if i >= max_results:
            break
        if not url.startswith("http"):
            continue
        results.append(SearchResult(
            title=title or "Untitled",
            url=url,
            description=snippets[i] if i < len(snippets) else None,
        ))

    return results


# Provider class

class DuckDuckGoSearchProvider(SearchProvider):
    """Search provider using DuckDuckGo Lite (no API key required)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "duckduckgo"

    async def search(self, options: SearchOptions) -> list[SearchResult]:
        """Execute a DuckDuckGo Lite search."""
        effective_query = f"site:{options.site} {options.query}" if options.site else options.query

        log.debug(
            "duckduckgo_search",
            query=effective_query,
            max_results=options.max_results,
        )

        body_params: dict[str, str] = {"q": effective_query}
        if options.freshness:
            df = _FRESHNESS_MAP.get(options.freshness)
            if df:
                body_params["df"] = df

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data=body_params,
                headers=headers,
            )
            if not resp.is_success:
                raise RuntimeError(f"DuckDuckGo Lite error ({resp.status_code})")
            html = resp.text

        return parse_ddg_lite_results(html, options.max_results)
