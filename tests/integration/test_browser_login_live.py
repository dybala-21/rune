"""In a real browser: the owner types into the page while a run waits, and a kept session's cookies
reach the site's first page."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from rune.api.computer import Computers, computer_router
from rune.capabilities.browser import login as login_module
from rune.capabilities.browser.core import _get_browser
from rune.connectors.store import authority_key

SEALED = {"key_id": "arn:aws:kms:test", "data_key": "k", "nonce": "n", "ciphertext": "c"}


async def test_the_owner_types_into_the_page_while_the_run_waits_and_the_run_looks_again_after(tmp_path):
    computers = Computers()
    entry = await computers.claim("signin", "run1")
    app = FastAPI()
    app.include_router(computer_router(computers, lambda: None))
    try:
        async with computers.bind(entry), httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                                            base_url="http://test") as client:
            _, page = await _get_browser("managed")
            await page.set_content("<title>Sign in</title><input id=password type=password>")
            assert entry.browser.hand_to_owner is not None
            state = (await client.get("/api/computer/state", params={"sessionId": "signin"})).json()
            assert state["state"] == "running"
            refused = await client.post("/api/computer/input", json={
                "sessionId": "signin", "lease": state["lease"], "frameId": state.get("frameId", ""),
                "action": "click", "x": 5, "y": 5})
            assert refused.status_code == 409, "not before the browser is handed over"

            await entry.browser.hand_to_owner(True)
            state = (await client.get("/api/computer/state", params={"sessionId": "signin"})).json()
            assert state["state"] == "manual" and state["runId"] == "run1", "the run waits, unpaused"
            box = await page.locator("#password").bounding_box()
            for action, values in (("click", {"x": box["x"] + 5, "y": box["y"] + 5}), ("text", {"text": "hunter2"})):
                answer = await client.post("/api/computer/input", json={
                    "sessionId": "signin", "lease": state["lease"], "frameId": state["frameId"],
                    "action": action, **values})
                assert answer.status_code == 200, answer.text
                state = answer.json()
            assert await page.locator("#password").input_value() == "hunter2"

            await entry.browser.hand_to_owner(False)
            state = (await client.get("/api/computer/state", params={"sessionId": "signin"})).json()
            assert state["state"] == "running"
            assert entry.browser.needs_observation, "the run observes the page again before acting"
        assert entry.browser.hand_to_owner is None, "only while the run is bound"
    finally:
        await computers.close()


class Echo(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server's name
        body = f"<title>shop</title><pre id=cookie>{self.headers.get('cookie', '')}</pre>".encode()
        if self.path == "/go":
            self.send_response(302)
            self.send_header("location", "/")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def test_a_kept_session_s_cookies_reach_the_site_s_first_page(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://10.207.0.1:4100")
    authority_key(tmp_path / "broker", create=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    login_module._write("127.0.0.1", ["127.0.0.1"], SEALED)
    state = {"cookies": [{"name": "sid", "value": "s3cret", "domain": "127.0.0.1", "path": "/", "expires": -1,
                          "httpOnly": True, "secure": False, "sameSite": "Lax"}],
             "origins": [{"origin": f"http://127.0.0.1:{port}", "localStorage": [{"name": "token", "value": "t"}]}]}
    broker = AsyncMock(return_value={"kept": True, "state": json.dumps(state)})
    monkeypatch.setattr("rune.connectors.client.call_broker", broker)
    computers = Computers()
    entry = await computers.claim("shop", "run1")
    try:
        async with computers.bind(entry):
            _, page = await _get_browser("managed")
            assert broker.await_count == 0, "nothing opened until the site loads"
            await page.goto(f"http://127.0.0.1:{port}/go", wait_until="domcontentloaded")
            assert page.url == f"http://127.0.0.1:{port}/", "the redirect is followed"
            assert await page.locator("#cookie").inner_text() == "sid=s3cret"
            assert await page.evaluate("localStorage.getItem('token')") == "t"
            await page.goto(f"http://127.0.0.1:{port}/again", wait_until="domcontentloaded")
            assert await page.locator("#cookie").inner_text() == "sid=s3cret"
            assert broker.await_count == 1, "opened once per browser"
    finally:
        await computers.close()
        server.shutdown()
        await asyncio.sleep(0)
