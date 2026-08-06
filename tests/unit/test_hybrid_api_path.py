"""Hybrid API path: POST capture/replay recipes and late-API accounting.

The path is opt-in (measured harmful as a default); tests enable it.
"""

import os

os.environ["RUNE_HYBRID_API"] = "1"

from rune.capabilities.browser.network import (
    ApiRequest,
    NetworkMonitor,
    format_api_recipe,
)


def _send(monitor, url, method="GET", post_data="", json_response=True):
    rid = f"r{len(monitor._requests)}{url[-8:]}"
    monitor._on_request({
        "type": "XHR",
        "requestId": rid,
        "request": {
            "url": url,
            "method": method,
            "postData": post_data,
            "headers": {"Content-Type": "application/x-www-form-urlencoded"} if post_data else {},
        },
    })
    monitor._on_response({
        "requestId": rid,
        "response": {"status": 200, "mimeType": "application/json" if json_response else "text/html"},
    })


def test_post_body_is_captured_and_truncated():
    m = NetworkMonitor()
    _send(m, "https://megabox.co.kr/on/oh/ohc/Brch/schedulePage.do",
          method="POST", post_data="brchNo=4062&playDe=20260806" + "x" * 600)
    api = m.get_discovered_apis()[0]
    assert api.method == "POST"
    assert api.post_data.startswith("brchNo=4062&playDe=20260806")
    assert len(api.post_data) <= 500
    assert api.request_content_type == "application/x-www-form-urlencoded"


def test_recipe_includes_method_and_body_for_post():
    api = ApiRequest(
        url="https://x.test/a.do", method="POST", resource_type="XHR",
        post_data="a=1&b=2", request_content_type="application/x-www-form-urlencoded",
    )
    r = format_api_recipe(api)
    assert 'method="POST"' in r and 'body="a=1&b=2"' in r

    get_api = ApiRequest(url="https://x.test/list.json", method="GET", resource_type="XHR")
    assert format_api_recipe(get_api) == 'web_fetch(url="https://x.test/list.json")'


def test_unreported_counts_only_new_interesting_calls():
    m = NetworkMonitor()
    _send(m, "https://x.test/first.json")
    assert m.unreported_interesting_count() == 1
    m.mark_reported()
    assert m.unreported_interesting_count() == 0
    # A plain-HTML GET is not interesting; a POST is even without JSON.
    _send(m, "https://x.test/page.html", json_response=False)
    assert m.unreported_interesting_count() == 0
    _send(m, "https://x.test/submit.do", method="POST",
          post_data="q=1", json_response=False)
    assert m.unreported_interesting_count() == 1
    m.clear()
    assert m.unreported_interesting_count() == 0


def test_web_fetch_params_accept_post_fields():
    from rune.capabilities.web import WebFetchParams

    p = WebFetchParams(
        url="https://x.test/a.do", method="POST", body="a=1",
        contentType="application/json",
    )
    assert p.method == "POST" and p.body == "a=1" and p.content_type == "application/json"
    # GET default unchanged
    assert WebFetchParams(url="https://x.test").method == "GET"


def test_json_prune_keeps_matching_records_whole():
    from rune.capabilities.web import _prune_json_by_term

    text = (
        '{"statCd":0,"megaMap":{"movieFormList":['
        '{"brchNm":"코엑스","playSchdlList":[{"time":"19:30","seat":"12"}]},'
        '{"brchNm":"송도","playSchdlList":[{"time":"20:00","seat":"5"}]}'
        ']},"note":"meta"}'
    )
    out = _prune_json_by_term(text, "코엑스")
    assert out is not None
    assert "코엑스" in out and "19:30" in out          # matching record intact
    assert "송도" not in out and "20:00" not in out    # non-matching pruned
    assert '"note": "meta"' in out or '"note":"meta"' in out.replace(" ", "")

    assert _prune_json_by_term("not json", "x") is None


def test_response_body_capture_via_cdp(monkeypatch):
    import asyncio

    from rune.capabilities.browser import network as net

    m = net.NetworkMonitor()

    class _FakeCdp:
        async def send(self, method, params=None):
            assert method == "Network.getResponseBody"
            return {"body": '{"schedule":[{"time":"19:30"}]}', "base64Encoded": False}

    m._cdp = _FakeCdp()
    _send(m, "https://x.test/schedulePage.do", method="POST", post_data="brchNo=1")
    api = m.get_discovered_apis()[0]
    rid = list(m._await_body.keys())[0]

    async def run():
        m._on_loading_finished({"requestId": rid})
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert api.response_body == '{"schedule":[{"time":"19:30"}]}'
    assert m._await_body == {}


def test_discover_read_body_returns_pruned_json():
    import asyncio

    from rune.capabilities.browser import network as net
    from rune.capabilities.browser.capabilities import (
        BrowserDiscoverApisParams,
        browser_discover_apis,
    )

    m = net.get_network_monitor()
    m.clear()
    m._active = True
    try:
        _send(m, "https://x.test/on/oh/schedulePage.do", method="POST", post_data="p=1")
        m.get_discovered_apis()[0].response_body = (
            '{"movieFormList":[{"brchNm":"코엑스","time":"19:30"},'
            '{"brchNm":"송도","time":"20:00"}]}'
        )
        res = asyncio.run(browser_discover_apis(
            BrowserDiscoverApisParams(readBody="schedulePage", jsonFilter="코엑스")
        ))
        assert res.success
        assert "19:30" in res.output and "송도" not in res.output

        missing = asyncio.run(browser_discover_apis(
            BrowserDiscoverApisParams(readBody="nosuch")
        ))
        assert not missing.success
    finally:
        m.clear()
        m._active = False
