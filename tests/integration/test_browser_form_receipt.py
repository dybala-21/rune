"""A full-page form submission leaves a receipt graded by HTTP semantics, not by the page it lands on."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    browser_act,
    browser_observe,
)
from rune.capabilities.browser.core import BrowserNavigateParams, browser_navigate
from tests.integration.test_browser_foundation import screen as screen


@pytest.fixture
def site():
    state = {"mode": "see_other", "posts": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _page(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/form":
                self._page(b'''<form method="post" action="/save"><input name="title" value="Release review">
                    <button id="save">Save</button></form>
                    <button id="api" onclick="fetch('/api',{method:'POST'})">Save via API</button>''')
            else:
                # Every landing page claims success; only the HTTP exchange tells them apart.
                self._page(b"<h1>Saved</h1>")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["posts"] += 1
            if self.path == "/api":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            mode = state["mode"]
            if mode == "drop":
                self.connection.close()
                return
            status, location = {"see_other": (303, "/done"), "login": (302, "/login"), "ok": (200, None),
                                "error": (500, None), "invalid": (422, None)}[mode]
            self.send_response(status)
            if location:
                self.send_header("Location", location)
            body = b"<h1>Saved</h1>" if location is None else b""
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/form", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def submit(url, selector="#save"):
    assert (await browser_navigate(BrowserNavigateParams(url=url))).success
    assert (await browser_observe(BrowserObserveParams())).success
    result = await browser_act(BrowserActParams(action="click", selector=selector))
    assert result.success, result.error
    return result.metadata.get("write_receipt")


@pytest.mark.parametrize("mode,status", [("see_other", "accepted"), ("ok", "accepted"),
                                         ("login", "unknown"), ("invalid", "rejected"),
                                         # A 500 can follow a committed write, so it does not prove nothing changed.
                                         ("error", "unknown"), ("drop", "unknown")])
async def test_form_post_receipt_follows_http_semantics(screen, site, mode, status):
    url, state = site
    state["mode"] = mode
    receipt = await submit(url)
    assert state["posts"] == 1
    assert receipt and receipt["status"] == status, receipt
    assert receipt["requests"][0]["method"] == "POST"


async def test_a_dropped_submission_does_not_stall_later_receipts(screen, site):
    url, state = site
    state["mode"] = "drop"
    assert (await submit(url))["status"] == "unknown"
    # The next action's own write is judged on its own response.
    assert (await submit(url, "#api"))["status"] == "accepted"
