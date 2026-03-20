"""Brave Search provider for RUNE.

Ported from src/capabilities/search/brave.ts - uses the Brave Search
API (requires BRAVE_API_KEY).
"""

from __future__ import annotations

import httpx

from rune.capabilities.search.provider import SearchOptions, SearchProvider, SearchResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


class BraveSearchProvider(SearchProvider):
    """Search provider backed by the Brave Search API."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:  # noqa: D102
        return "brave"

    async def search(self, options: SearchOptions) -> list[SearchResult]:
        """Execute a Brave web search."""
        effective_query = f"site:{options.site} {options.query}" if options.site else options.query

        log.debug(
            "brave_search",
            query=effective_query,
            max_results=options.max_results,
        )

        params: dict[str, str | int] = {
            "q": effective_query,
            "count": min(options.max_results, 20),
        }
        if options.freshness:
            params["freshness"] = options.freshness

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers=headers,
            )
            if not resp.is_success:
                error_text = resp.text[:200]
                raise RuntimeError(
                    f"Brave Search API error ({resp.status_code}): {error_text}"
                )
            data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            results.append(SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                description=item.get("description"),
                age=item.get("age"),
            ))

        return results
