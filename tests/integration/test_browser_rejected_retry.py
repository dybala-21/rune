"""A save the server rejected may be retried once; a successful save is never repeated."""

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
    state = {"statuses": [], "posts": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            data = b'''<input id="title" value="Release review"><button id="save" onclick="fetch('/save',{method:'POST',body:title.value})
                .then(r=>status.textContent=r.ok?'Saved':'Service unavailable. Nothing was saved.')">Save draft</button>
                <p id="status" role="status">Not saved</p>'''
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status = state["statuses"][state["posts"]] if state["posts"] < len(state["statuses"]) else 200
            state["posts"] += 1
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/form", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("statuses,posts", [
    ([200], 1),             # A successful save is never sent again.
    ([503], 2),             # A rejected save is retried once.
    ([503, 503], 2),        # A rejected retry is not retried again.
    ([400, 200], 2),        # The successful retry is not repeated.
    ([500], 1),             # A 500 may follow a committed write, so it is never sent again.
    ([504], 1),
])
async def test_identical_save_is_dispatched_again_only_after_a_rejection(screen, site, statuses, posts):
    url, state = site
    state["statuses"] = statuses
    assert (await browser_navigate(BrowserNavigateParams(url=url))).success
    assert (await browser_observe(BrowserObserveParams())).success
    for _ in range(3):
        result = await browser_act(BrowserActParams(action="click", selector="#save"))
        assert result.success and "Reading the resulting page failed" not in result.output, result.output
    assert state["posts"] == posts
