"""The managed browser follows a tab an action opens and returns when a followed tab closes."""

import json
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
from rune.capabilities.browser.session import current_session
from tests.integration.test_browser_foundation import screen as screen

PAGES = {
    "/form": b'''<h1>Bookings</h1><a id="edit" href="/edit" target="_blank">Edit in new tab</a>
        <button id="auth" onclick="window.open('/auth')">Sign in</button>
        <button id="stay" onclick="document.querySelector('h1').textContent='Still here'">Stay</button>''',
    "/edit": b'''<h1>Edit booking</h1><label>Title <input id="title" value=""></label>
        <button id="save" onclick="fetch('/save',{method:'POST',body:JSON.stringify({title:document.querySelector('#title').value})})
          .then(()=>document.querySelector('h1').textContent='Saved')">Save</button>''',
    "/auth": b'<h1>Signing in</h1><script>setTimeout(() => window.close(), 50)</script>',
}


@pytest.fixture
def site():
    state = {"saves": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            body = PAGES.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            state["saves"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def open_form(url):
    assert (await browser_navigate(BrowserNavigateParams(url=url + "/form"))).success
    assert (await browser_observe(BrowserObserveParams())).success


async def test_a_link_that_opens_a_new_tab_is_followed_and_its_save_is_seen(screen, site):
    url, state = site
    await open_form(url)
    opened = await browser_act(BrowserActParams(action="click", selector="#edit"))
    assert opened.success, opened.error
    assert "Opened a new tab" in opened.output and "Edit booking" in opened.output
    assert current_session().page.url == url + "/edit"

    assert (await browser_act(BrowserActParams(action="type", selector="#title", value="Release review"))).success
    saved = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert state["saves"] == [{"title": "Release review"}]
    # The network monitor moved with the tab, so the save in the new tab still has a receipt.
    assert saved.metadata["write_receipt"]["status"] == "accepted"


async def test_a_popup_that_closes_returns_to_the_tab_that_opened_it(screen, site):
    url, _ = site
    await open_form(url)
    clicked = await browser_act(BrowserActParams(action="click", selector="#auth"))
    assert clicked.success, clicked.error
    await screen.wait_for_timeout(300)  # the popup closes itself
    observed = await browser_observe(BrowserObserveParams())
    assert observed.success, observed.error
    assert current_session().page.url == url + "/form" and "Bookings" in observed.output


async def test_an_ordinary_click_stays_in_the_same_tab(screen, site):
    url, _ = site
    await open_form(url)
    page = current_session().page
    clicked = await browser_act(BrowserActParams(action="click", selector="#stay"))
    assert clicked.success and "Opened a new tab" not in clicked.output
    assert current_session().page is page


def test_browser_tools_keep_their_session_guard():
    from rune.capabilities.browser import capabilities

    for name in ("browser_act", "browser_observe"):
        assert hasattr(getattr(capabilities, name), "__wrapped__"), name
    assert not hasattr(capabilities._follow_new_tab, "__wrapped__")
