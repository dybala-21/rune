"""Tests for web capabilities (mock httpx)."""

from __future__ import annotations

import json

import httpx
import pytest

from rune.capabilities.web import (
    WebFetchParams,
    WebSearchParams,
    _extract_by_selector,
    _html_to_text,
    _is_path_wipe_redirect,
    _prune_json_by_term,
    web_fetch,
)


@pytest.mark.parametrize("case", ["timeout", "invalid_json", "accepted", "redirect", "disabled"])
async def test_write_response_failures_keep_dispatch_status(monkeypatch, case):
    monkeypatch.setenv("RUNE_HYBRID_API", "0" if case == "disabled" else "1")
    calls = []

    def respond(request):
        calls.append(request)
        if case == "timeout":
            raise httpx.ReadTimeout("Response lost", request=request)
        if case == "redirect":
            return httpx.Response(307, headers={"location": "https://example.org/other"})
        if case == "accepted":
            return httpx.Response(202, json={"queued": True})
        return httpx.Response(200, text='{"broken":', headers={"content-type": "application/json"})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    result = await web_fetch(WebFetchParams(url="https://example.org/orders", method="POST", body="{}"))
    assert not result.success
    assert result.metadata["action_status"] == ("not_executed" if case == "disabled" else "unknown")
    assert len(calls) == (0 if case == "disabled" else 1)


async def test_record_query_selects_latest_before_delivery_limit(monkeypatch):
    payload = {"unit": "USD", "rows": [{"entity": "ACME", "value": i, "time": f"2026-10-{i:02d}"} for i in range(1, 17)]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/records", maxLength=300,
        recordQuery={"path": "/rows", "operation": "latest", "field": "time", "where": {"entity": "ACME"}}))
    assert result.success, result.error
    assert json.loads(result.output[result.metadata["body_offset"]:]) == {"unit": "USD", **payload["rows"][-1]}
    assert result.metadata["record_query"]["matched"] == 16
    assert "pagination not checked" in result.output


async def test_record_query_rejects_pre_filtered_inputs_and_cannot_claim_a_partial_sum(monkeypatch):
    reads = []
    def respond(req):
        reads.append(req)
        return httpx.Response(200, json=[{"v": 1}, {}])
    transport = httpx.MockTransport(respond)
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    query = {"operation": "sum", "field": "v"}
    result = await web_fetch(WebFetchParams(url="https://example.org/records", recordQuery=query))
    assert not result.success and result.metadata["record_query_failed"]
    result = await web_fetch(WebFetchParams(url="https://example.org/records", jsonFilter="1", recordQuery=query))
    assert not result.success and "cannot be combined" in result.error
    assert len(reads) == 1


async def test_large_json_and_wrong_path_return_structure_for_recovery(monkeypatch):
    payload = {"observations": [{"station": "North", "value": 17.125}] * 1000}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    for options in ({}, {"recordQuery": {"operation":"sum", "field":"value"}}):
        result = await web_fetch(WebFetchParams(url="https://example.org/data", **options))
        assert not result.success and '"path":"/observations"' in result.error
        assert '"sampleFields":["station","value"]' in result.error and "17.125" not in result.error


@pytest.mark.parametrize("records", [
    {"symbol": "ACME", "quote": {"value": 120, "unit": "USD", "observed_at": "2026-10-16T09:00:00Z"}},
    {"ACME": {"value": 120, "unit": "USD", "observed_at": "2026-10-16T09:00:00Z"}},
    [{"symbol": "ACME", "quote": {"value": 120, "unit": "USD", "observed_at": "2026-10-16T09:00:00Z"}}],
])
def test_json_filter_preserves_record_identity_and_nested_observation(records):
    assert json.loads(_prune_json_by_term(json.dumps(records), "acme")) == records
    assert _prune_json_by_term(json.dumps(records), "missing") == ""


def test_json_filter_retains_parent_scope_without_other_records():
    record = {"symbol": "ACME", "value": 120, "observed_at": "2026-10-16T09:00:00Z"}
    payload = {"unit": "USD", "metadata": {"venue": "X", "timezone": "UTC"},
               "data": [record, {"symbol": "OTHER", "value": 999}]}
    assert json.loads(_prune_json_by_term(json.dumps(payload), "ACME")) == {**payload, "data": [record]}


def test_json_filter_does_not_round_numbers_or_change_numeric_strings():
    text = '{"symbol":"ACME","value":0.123456789012345678901,"count":12345678901234567890,"rate":1.20e-9,"code":"005930"}'
    assert _prune_json_by_term(text, "ACME") == text


@pytest.mark.parametrize("body,kwargs,flag", [
    ('{"symbol":"OTHER","value":999}', {"jsonFilter": "ACME"}, "filter_no_match"),
    ('[{"symbol":"OTHER","value":999}]', {"jsonFilter": "ACME"}, "filter_no_match"),
    ('{"symbol":"ACME","value":120,"unit":"USD","observed_at":"2020-01-01"}', {"maxLength": 40}, "truncated"),
    ('{"value":120', {}, "invalid_json"),
])
async def test_missing_or_incomplete_json_is_not_source_evidence(monkeypatch, body, kwargs, flag):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body, headers={"content-type": "application/json"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/quotes", **kwargs))
    assert not result.success and result.metadata[flag]
    assert not result.output
    from rune.agent.web_grounding import WebGrounding

    check = WebGrounding(required=True, requires_page=True)
    assert not check.observe("web_fetch", {}, result)
    assert check.blocker()


@pytest.mark.parametrize("body", ['[]', '{}', 'null', '0', 'false'])
async def test_valid_empty_or_zero_json_is_returned_without_inventing_a_failure(monkeypatch, body):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body, headers={"content-type": "application/json"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/data"))
    assert result.success and result.output == body


async def test_filter_recovery_keeps_the_observation_time_separate_from_fetch_time(monkeypatch):
    client = httpx.AsyncClient
    payload = {"symbol": "ACME", "value": 120, "unit": "USD", "observed_at": "2020-01-01"}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    failed = await web_fetch(WebFetchParams(url="https://example.org/quotes", jsonFilter="missing"))
    result = await web_fetch(WebFetchParams(url="https://example.org/quotes", jsonFilter="ACME"))
    assert not failed.success and result.success
    assert '"observed_at":"2020-01-01"' in result.output
    assert result.metadata["retrieved_at"] != payload["observed_at"]


async def test_large_json_points_to_filtering_when_a_larger_limit_cannot_help(monkeypatch):
    client = httpx.AsyncClient
    payload = {"unit": "celsius", "records": [{"station": "Other", "data": "x" * 30_000},
                                              {"station": "North", "value": 17.125}]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    for limit in (8000, 100_000):
        result = await web_fetch(WebFetchParams(url="https://example.org/data", maxLength=limit))
        assert not result.success and "jsonFilter" in result.error and "increase maxLength" not in result.error
    filtered = await web_fetch(WebFetchParams(url="https://example.org/data", maxLength=8000, jsonFilter="North"))
    assert filtered.success and '"value":17.125' in filtered.output and '"unit":"celsius"' in filtered.output


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


@pytest.mark.parametrize("body", [
    '<table><tr><td><iframe src="/news"></iframe></td></tr></table>',
    '<ul><li><a href="javascript:openNews()"><img src="banner.png"></a></li></ul>',
    '<main><table><tr><td> </td></tr></table></main>',
])
async def test_layout_only_html_is_not_source_content(monkeypatch, body):
    from rune.agent.web_grounding import WebGrounding

    async def get(self, url, **kwargs):
        return httpx.Response(200, text=body, headers={"content-type": "text/html"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    result = await web_fetch(WebFetchParams(url="https://example.org/"))
    assert not result.success and result.metadata["empty_content"]
    grounding = WebGrounding(required=True, requires_page=True)
    assert not grounding.observe("web_fetch", {"url": "https://example.org/"}, result)
    assert grounding.pages == 0


@pytest.mark.parametrize("body,expected", [('<p>|</p>', '|'), ('<p>0</p>', '0'),
    ('<p>✓</p>', '✓'), ('<a href="/news">News</a>', '[News](/news)')])
def test_html_reading_preserves_source_symbols_and_named_links(body, expected):
    assert _html_to_text(body) == expected


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


async def test_empty_main_does_not_return_navigation_as_source_content(monkeypatch):
    client = httpx.AsyncClient
    page = '<nav>Home Quotes Help</nav><main><script>loadQuotes()</script></main><footer>Privacy</footer>'
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/quotes"))
    assert not result.success and result.metadata.get("empty_content")


@pytest.mark.parametrize("element", ["main", 'div role="main"', "article"])
async def test_main_content_survives_a_small_fetch_limit(monkeypatch, element):
    page = ('<html><head><title>Quote</title></head><body><nav>' + 'Menu ' * 2000
            + f'</nav><{element}><h1>Example</h1><p>213,000 KRW at 09:02 GMT+9</p>'
            + '<table><tr><td>2026-10-15</td><td>210,000</td></tr>'
            + '<tr><td>2026-10-16</td><td>213,000</td></tr></table>'
            + f'</{element.split()[0]}></body></html>')
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/quote", maxLength=300))
    assert result.success and result.metadata["extraction"] == ("article" if element == "article" else "main")
    assert "213,000 KRW at 09:02 GMT+9" in result.output
    assert "2026-10-15 | 210,000 |\n" in result.output
    assert not result.metadata["truncated"] and "Menu" not in result.output
    full = await web_fetch(WebFetchParams(url="https://example.org/quote", selector="body"))
    assert "Menu" in full.output and "213,000" in full.output


@pytest.mark.parametrize("body", ["", '<div id="root"></div><script>loadData()</script>'])
async def test_title_only_html_does_not_count_as_readable_content(monkeypatch, body):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, text=f"<html><head><title>Finance</title></head><body>{body}</body></html>",
        headers={"content-type": "text/html"},
    ))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/quote"))
    assert not result.success and result.metadata["empty_content"]


@pytest.mark.parametrize("content_type,body", [
    ("application/json", '{"comparison": "a < b > c", "html": "<p>2</p>"}'),
    ("text/plain", "a < b > c\n  keep spacing &amp; symbols"),
])
async def test_non_html_responses_are_preserved(monkeypatch, content_type, body):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body, headers={"content-type": content_type}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/data"))
    assert result.success and result.output == body


async def test_frameset_exposes_content_urls_without_counting_the_shell_as_evidence(monkeypatch):
    from rune.agent.web_grounding import WebGrounding

    body = '''<html><head><title>Example game</title></head>
    <frameset rows="0,*"><frame src="/session"><frame src="/main/index"></frameset>
    <noframes><body></body></noframes></html>'''
    client = httpx.AsyncClient
    reads = []

    def respond(req):
        reads.append(str(req.url))
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/"))
    assert not result.success and result.metadata["empty_content"]
    assert result.metadata["embedded_pages"] == ["https://example.org/session", "https://example.org/main/index"]
    assert "https://example.org/main/index" in result.error
    assert reads == ["https://example.org/"]
    check = WebGrounding(required=True, requires_page=True)
    assert not check.observe("web_fetch", {}, result) and check.pages == 0


async def test_iframe_links_preserve_parent_text_and_ignore_non_web_sources(monkeypatch):
    body = '''<html><body><p>Published in 2024.</p><iframe src="../news"></iframe>
    <iframe src="../news#latest"></iframe><iframe src="javascript:alert(1)"></iframe>
    <iframe src="file:///etc/passwd"></iframe><iframe src="https://user:secret@example.org/private"></iframe>
    </body></html>'''
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body, headers={"content-type": "text/html"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/main/index"))
    assert result.success
    assert result.metadata["embedded_pages"] == ["https://example.org/news"]
    assert "https://example.org/news" in result.output
    assert result.output[result.metadata["body_offset"]:] == "Published in 2024."
    limited = await web_fetch(WebFetchParams(url="https://example.org/main/index", maxLength=120))
    assert limited.success and len(limited.output) <= 120


@pytest.mark.parametrize("limit", [105, 300, 1000])
async def test_frame_hints_cannot_displace_a_fitting_source_body(monkeypatch, limit):
    text = "Important announcement: service will end tomorrow."
    frames = "".join(f'<iframe src="https://ads.example.org/widget/{i}"></iframe>' for i in range(20))
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, text=f"<main>{text}</main>{frames}", headers={"content-type": "text/html"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/", maxLength=limit))
    assert result.success and len(result.output) <= limit
    assert result.output[result.metadata["body_offset"]:] == text
    assert not result.metadata["truncated"]


async def test_long_parent_keeps_body_and_frame_discovery_within_the_limit(monkeypatch):
    text = "Official notice details. " * 200
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, text=f'<main>{text}</main><iframe src="/ads/{"x" * 500}"></iframe><iframe src="/news"></iframe>',
        headers={"content-type": "text/html"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/", maxLength=1000))
    assert result.success and result.metadata["truncated"]
    assert len(result.output) <= 1000 and "https://example.org/news" in result.output
    assert len(result.output[result.metadata["body_offset"]:]) >= 750


async def test_truncating_plain_text_before_its_content_does_not_count_as_a_read(monkeypatch):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, text=" " * 300 + "Actual content", headers={"content-type": "text/plain"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    result = await web_fetch(WebFetchParams(url="https://example.org/", maxLength=100))
    assert not result.success and result.metadata["empty_content"]


def test_inline_frame_does_not_advertise_its_inactive_src():
    from rune.capabilities.web import _embedded_page_urls

    assert _embedded_page_urls('<iframe src="/old" srcdoc="New notice"></iframe>', "https://example.org/") == []


@pytest.mark.parametrize("case", ["chain", "external", "multiple", "parent", "cycle", "depth", "selector", "filter", "redirect", "external_redirect"])
async def test_frame_reads_are_bounded_and_keep_the_actual_source(monkeypatch, case):
    from rune.agent.tool_adapter import _format_tool_output
    from rune.agent.web_grounding import WebGrounding

    pages = {"/": '<iframe src="/main"></iframe>', "/main": '<iframe src="/news"></iframe>',
             "/news": '<main>Service reopened.</main>'}
    options = {}
    expected = ["/", "/main", "/news"]
    if case == "external":
        pages["/"] = '<iframe src="https://other.example/news"></iframe>'
        expected = ["/"]
    elif case == "multiple":
        pages["/"] += '<iframe src="/other"></iframe>'
        expected = ["/"]
    elif case == "parent":
        pages["/"] += '<main>Parent notice.</main>'
        expected = ["/"]
    elif case == "cycle":
        pages["/main"] = '<iframe src="/"></iframe>'
        expected = ["/", "/main"]
    elif case == "depth":
        pages["/news"] = '<iframe src="/further"></iframe>'
    elif case in {"selector", "filter"}:
        options = {"selector": "body"} if case == "selector" else {"jsonFilter": "service"}
        expected = ["/"]
    elif case in {"redirect", "external_redirect"}:
        expected = ["/", "/main", "/news"] if case == "redirect" else ["/", "/main"]
    calls = []

    async def send(self, request, **kwargs):
        assert request.url.host == "example.org"
        calls.append(request.url.path)
        if request.url.path == "/main" and case in {"redirect", "external_redirect"}:
            target = "/news" if case == "redirect" else "https://other.example/news"
            return httpx.Response(302, headers={"location": target}, request=request)
        return httpx.Response(200, text=pages[request.url.path], headers={"content-type": "text/html"}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await web_fetch(WebFetchParams(url="https://example.org/", **options))
    assert calls == expected
    readable = case in {"chain", "parent", "redirect"}
    assert result.success is readable
    grounding = WebGrounding(required=True, requires_page=True)
    assert grounding.observe("web_fetch", {}, result) is readable
    if case in {"chain", "redirect"}:
        assert result.metadata["requested_url"] == "https://example.org/"
        assert result.metadata["final_url"] == "https://example.org/news"
        assert "[SOURCE_URL: https://example.org/news]" in _format_tool_output("web_fetch", {}, result)
    if case == "external_redirect":
        assert result.metadata["redirect_blocked"]


async def test_frame_chain_shares_one_request_deadline(monkeypatch):
    import asyncio

    timeout = asyncio.timeout
    deadlines, calls = [], []

    def deadline(seconds):
        deadlines.append(seconds)
        return timeout(0.01)

    async def send(self, request, **kwargs):
        calls.append(request.url.path)
        if request.url.path == "/main":
            await asyncio.sleep(1)
        return httpx.Response(200, text='<iframe src="/main"></iframe>',
                              headers={"content-type": "text/html"}, request=request)

    monkeypatch.setattr("rune.capabilities.web.asyncio.timeout", deadline)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await web_fetch(WebFetchParams(url="https://example.org/"))
    assert deadlines == [30] and calls == ["/", "/main"]
    assert not result.success and "deadline" in result.error


@pytest.mark.parametrize("base,expected", [
    ('<base href="https://example.net/content/">', "https://example.net/content/news"),
    ('<base href="https://[broken">', "https://example.org/news"),
])
def test_embedded_page_urls_resolve_base_without_fetching_it(base, expected):
    from rune.capabilities.web import _embedded_page_urls

    assert _embedded_page_urls(f"<html><head>{base}</head><body><iframe src='news'></iframe></body></html>",
                               "https://example.org/") == [expected]


def test_nested_main_is_not_duplicated_and_entities_are_readable():
    from rune.capabilities.web import _extract_page_content

    text, mode = _extract_page_content('<div role="main"><main><p>A &amp; B &lt; 3</p></main></div>')
    assert mode == "main" and text == "A & B < 3"
    text, mode = _extract_page_content('<main></main><p>Available: 2</p>')
    assert mode == "body" and text == "Available: 2"
    text, mode = _extract_page_content('<main hidden>Old value</main><main>Current value</main>')
    assert mode == "main" and text == "Current value"


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
