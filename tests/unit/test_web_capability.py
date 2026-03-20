"""Tests for web capabilities (mock httpx)."""

from __future__ import annotations

from rune.capabilities.web import (
    WebFetchParams,
    WebSearchParams,
    _extract_by_selector,
    _html_to_text,
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
