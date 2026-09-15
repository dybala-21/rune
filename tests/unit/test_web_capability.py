"""Tests for web capabilities (mock httpx)."""

from __future__ import annotations

import httpx
import pytest

from rune.capabilities.web import (
    WebFetchParams,
    WebSearchParams,
    _extract_by_selector,
    _html_to_text,
    _is_path_wipe_redirect,
    web_fetch,
)


def test_html_to_text():
    """Strips tags and normalises whitespace."""
    html = "<html><body><script>alert(1)</script><p>Hello   World</p></body></html>"
    result = _html_to_text(html)
    assert "Hello" in result
    assert "World" in result
    assert "<" not in result
    assert "alert" not in result
    # Whitespace normalised (no multiple spaces)
    assert "   " not in result


def test_extract_by_selector():
    """Basic CSS extraction: tag, class, and id selectors."""
    html = '<div id="main"><p class="content">Target text</p><p>Other</p></div>'

    # Tag selector
    result = _extract_by_selector(html, "p")
    assert "Target text" in result

    # ID selector
    result_id = _extract_by_selector(html, "#main")
    assert "Target text" in result_id

    # Class selector
    result_cls = _extract_by_selector(html, ".content")
    assert "Target text" in result_cls


@pytest.mark.parametrize("selector", ["#decimal-objects", r"#decimal\.Decimal + dd", '[id="decimal.Decimal"] + dd > p'])
def test_nested_document_sections_and_compound_selectors(selector):
    html = '<section id="decimal-objects"><span></span><dl><dt id="decimal.Decimal">Decimal</dt><dd><p>Exact <code>0.1</code> from strings.</p><p>Float conversion retains the approximation.</p></dd></dl></section><p>Unrelated footer</p>'
    output = _extract_by_selector(html, selector)
    assert "Exact" in output and "approximation" in output
    assert "Unrelated footer" not in output


@pytest.mark.parametrize("selector", ["#missing", "[broken", "#empty"])
async def test_selector_errors_do_not_mark_a_domain_as_js_only(monkeypatch, selector):
    client = httpx.AsyncClient
    html = '<div id="empty"></div><p id="price">Price: 12,000 KRW</p>'
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=html, request=req))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    for _ in range(3):
        result = await web_fetch(WebFetchParams(url="https://example.org", selector=selector))
        assert not result.success
        assert result.metadata.get("selector_error") is True
        assert not result.metadata.get("js_rendered")
    result = await web_fetch(WebFetchParams(url="https://example.org", selector="#price"))
    assert result.success and "12,000" in result.output


@pytest.mark.parametrize("content_type,text", [
    ("text/html", "<p>Available: 2 seats</p>"),
    ("application/json", '{"available": 2}'),
    ("text/plain", "Available: 2 seats"),
])
async def test_short_valid_response_does_not_require_a_browser(monkeypatch, content_type, text):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=text, headers={"content-type": content_type}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/status"))
    assert result.success and "2" in result.output
    assert not result.metadata.get("js_rendered")


async def test_failed_page_does_not_block_other_pages_or_later_recovery(monkeypatch):
    client = httpx.AsyncClient
    responses = iter([503, 503, 200, 200])
    transport = httpx.MockTransport(lambda req: httpx.Response(next(responses), text="<p>Recovered</p>"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    for _ in range(2):
        assert not (await web_fetch(WebFetchParams(url="https://example.org/down"))).success
    assert (await web_fetch(WebFetchParams(url="https://example.org/healthy"))).success
    assert (await web_fetch(WebFetchParams(url="https://example.org/down"))).success


async def test_fetch_cooldown_is_per_run_per_url_and_expires(monkeypatch):
    from rune.capabilities.fetch_state import fetch_scope

    now = [10.0]
    monkeypatch.setattr("rune.capabilities.fetch_state.time.monotonic", lambda: now[0])
    requests = []

    def respond(req):
        requests.append(str(req.url))
        status = 503 if len(requests) <= 2 else 200
        return httpx.Response(status, text="<p>Ready</p>")

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    params = WebFetchParams(url="https://example.org/down")
    with fetch_scope():
        assert not (await web_fetch(params)).success
        assert not (await web_fetch(params)).success
        assert (await web_fetch(params)).metadata["retry_after_seconds"] == 30
        assert len(requests) == 2
        assert (await web_fetch(WebFetchParams(url="https://example.org/other"))).success
        with fetch_scope():
            assert (await web_fetch(params)).success
        assert (await web_fetch(params)).metadata["skipped"]
        now[0] += 31
        assert (await web_fetch(params)).success
        assert (await web_fetch(params)).success
    assert (await web_fetch(params)).success


def test_web_search_params():
    """Pydantic model validation for WebSearchParams."""
    params = WebSearchParams(query="python asyncio")
    assert params.query == "python asyncio"
    assert params.max_results == 10
    assert params.language == "en"

    # With aliases
    params2 = WebSearchParams(query="test", maxResults=5, language="ko")
    assert params2.max_results == 5
    assert params2.language == "ko"


def test_web_fetch_params():
    """Pydantic model validation for WebFetchParams."""
    params = WebFetchParams(url="https://example.com")
    assert params.url == "https://example.com"
    assert params.selector is None
    assert params.max_length == 50_000

    params2 = WebFetchParams(url="https://example.com", selector=".main", maxLength=1000)
    assert params2.selector == ".main"
    assert params2.max_length == 1000


class TestIsPathWipeRedirect:
    def test_no_redirect_returns_false(self):
        url = "https://x.com/a/b"
        assert _is_path_wipe_redirect(url, url) is False

    def test_https_upgrade_is_not_wipe(self):
        assert _is_path_wipe_redirect(
            "http://x.com/a/b", "https://x.com/a/b",
        ) is False

    def test_trailing_slash_is_not_wipe(self):
        assert _is_path_wipe_redirect(
            "https://x.com/a/b", "https://x.com/a/b/",
        ) is False

    def test_canonical_path_prefix_preserved(self):
        assert _is_path_wipe_redirect(
            "https://x.com/a/b/c", "https://x.com/a/b",
        ) is False

    def test_deep_path_to_root_is_wipe(self):
        assert _is_path_wipe_redirect(
            "https://yeogi.com/domestic-accommodations/6128",
            "https://yeogi.com/",
        ) is True

    def test_deep_path_to_search_is_wipe(self):
        assert _is_path_wipe_redirect(
            "https://yeogi.com/domestic-accommodations/6128",
            "https://yeogi.com/search",
        ) is True

    def test_shallow_original_not_wipe(self):
        assert _is_path_wipe_redirect(
            "https://x.com/a", "https://x.com/",
        ) is False

    def test_different_domain_deep_to_shallow_is_wipe(self):
        assert _is_path_wipe_redirect(
            "https://a.com/x/y/z", "https://b.com/",
        ) is True
