"""Search providers must preserve the requested date filter and result dates."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from rune.capabilities.search.brave import BraveSearchProvider
from rune.capabilities.search.browser_search import BrowserSearchProvider
from rune.capabilities.search.provider import SearchOptions, SearchResult
from rune.capabilities.web import WebSearchParams, web_search


@pytest.mark.parametrize("freshness,expected", [(None, None), ("day", "pd"), ("week", "pw"), ("month", "pm"), ("year", "py")])
async def test_brave_uses_documented_date_filters(monkeypatch, freshness, expected):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"web": {"results": [{
            "title": "Report", "url": "https://example.org/report", "age": "October 15, 2026",
        }]}})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    results = await BraveSearchProvider("test-key").search(SearchOptions(query="report", freshness=freshness))
    assert requests[0].url.params.get("freshness") == expected
    assert results[0].age == "October 15, 2026"


@pytest.mark.parametrize("freshness,expected", [(None, None), ("day", "d"), ("week", "w"), ("month", "m"), ("year", "y")])
async def test_browser_fallback_keeps_the_filter_on_retry(monkeypatch, freshness, expected):
    html = '<a class="result-link" href="https://example.org">Result</a>'
    page = SimpleNamespace(goto=AsyncMock(side_effect=[SimpleNamespace(status=429), SimpleNamespace(status=200)]),
                           content=AsyncMock(return_value=html))
    pool = SimpleNamespace(acquire=AsyncMock(return_value=page), release=Mock())
    monkeypatch.setattr("rune.capabilities.search.browser_search.asyncio.sleep", AsyncMock())
    results = await BrowserSearchProvider(pool).search(SearchOptions(query="report", freshness=freshness))
    assert len(results) == 1 and page.goto.call_count == 2
    for call in page.goto.call_args_list:
        assert parse_qs(urlsplit(call.args[0]).query).get("df") == ([expected] if expected else None)
    pool.release.assert_called_once_with(page)


async def test_search_exposes_page_dates_without_claiming_live_observations(monkeypatch):
    provider = SimpleNamespace(search=AsyncMock(return_value=[
        SearchResult(title="Report", url="https://example.org", description="Value: 267,250", age="September 9, 2026"),
    ]))
    monkeypatch.setattr("rune.capabilities.web.get_search_provider", lambda: provider)
    result = await web_search(WebSearchParams(query="current value", freshness="day"))
    assert "September 9, 2026" in result.output and "not live readings" in result.output
    assert "not necessarily the date of the facts" in result.output
    assert result.metadata["source_kind"] == "search_excerpt"
    assert result.metadata["freshness"] == "day"
    assert provider.search.call_args.args[0].freshness == "day"


def test_unknown_date_filters_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WebSearchParams(query="report", freshness="yesterday")
