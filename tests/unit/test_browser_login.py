"""The owner signs in to a site from a card in the chat, and a hosted machine keeps it only sealed."""

import asyncio
import json
import stat
from unittest.mock import AsyncMock

import pytest

from rune.api.approvals import approval_callback
from rune.capabilities.browser import login as login_module
from rune.capabilities.browser.login import (
    BrowserLoginParams,
    attach,
    browser_login,
    keep,
    saved,
    site_of,
)
from rune.capabilities.browser.session import BrowserSession, _current
from rune.connectors.store import authority_key
from rune.safety.approval_context import approval_channel_scope, parking_scope

SEALED = {"key_id": "arn:aws:kms:test", "data_key": "k", "nonce": "n", "ciphertext": "c"}
STATE = {"cookies": [
    {"name": "sid", "value": "s3cret", "domain": ".shop.example", "path": "/"},
    {"name": "cdn", "value": "x", "domain": "img.shop.example", "path": "/"},
    {"name": "track", "value": "y", "domain": ".tracker.example", "path": "/"},
], "origins": [
    {"origin": "https://www.shop.example", "localStorage": [{"name": "token", "value": "t"}]},
    {"origin": "https://tracker.example", "localStorage": [{"name": "id", "value": "z"}]},
]}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://10.207.0.1:4100")
    authority_key(tmp_path / "broker", create=True)
    monkeypatch.setattr(login_module, "KEEP_POLL_SECONDS", 0.01)
    return tmp_path


class Context:
    def __init__(self):
        self.listeners, self.routes, self.cookies, self.scripts = {}, [], [], []

    async def storage_state(self):
        return STATE

    def on(self, event, handler):
        self.listeners[event] = handler

    def remove_listener(self, event, handler):
        assert self.listeners.pop(event) is handler

    async def route(self, matcher, handler):
        self.routes.append((matcher, handler))

    async def add_cookies(self, cookies):
        self.cookies.extend(cookies)

    async def add_init_script(self, script):
        self.scripts.append(script)


class Page:
    def __init__(self):
        self.context, self.url, self.viewport_size, self.sizes = Context(), "about:blank", {"width": 1280, "height": 800}, []

    async def goto(self, url, **_):
        self.url = url
        # The sign-in page loads from the site, its CDN and a tracker.
        for host in ("www.shop.example", "img.shop.example", "tracker.example"):
            request = type("Request", (), {"url": f"https://{host}/x"})()
            if "request" in self.context.listeners:
                self.context.listeners["request"](request)

    async def set_viewport_size(self, size):
        self.sizes.append(size)
        self.viewport_size = size


def broker(monkeypatch, *, kept_after=None, kept_at_first=False, private_fails=False):
    """A broker that keeps the site once `kept_after` status calls have gone by (never if None)."""
    calls = []

    async def call(operation, params):
        calls.append((operation, params))
        if operation == "private":
            if private_fails:
                raise OSError("broker down")
            return {"private": True, "closed": True}
        if operation == "login":
            return {"kept": kept_at_first, "approval_url": "https://approve.example.com/"}
        if operation == "status":
            done = kept_after is not None and sum(op == "status" for op, _ in calls) > kept_after
            return {"closed": False, "sites": ["shop.example"] if done else []}
        if operation == "keep":
            return {"kept": True, "sealed": SEALED}
        raise AssertionError(operation)

    monkeypatch.setattr("rune.connectors.client.call_broker", AsyncMock(side_effect=call))
    return calls


def browser(monkeypatch):
    page = Page()
    session = BrowserSession(owner_id="conv-1")
    handed = []

    async def hand(owner):
        # Each page load while the owner has it is seen, as Playwright's request event would.
        handed.append(owner)
        if owner:
            await page.goto(page.url)

    session.hand_to_owner = hand
    _current.set(session)
    monkeypatch.setattr("rune.capabilities.browser.core._get_browser", AsyncMock(return_value=(None, page)))
    return page, handed


def channel(timeout_ms=5000):
    events, pending = [], {}

    async def emit(event, data):
        events.append((event, data))

    return approval_callback("run1", pending, emit, timeout_ms=timeout_ms), events, pending


async def press(pending, decision):
    async with asyncio.timeout(5):
        while not pending:
            await asyncio.sleep(0.001)
    next(iter(pending.values())).set_result({"decision": decision})


def test_a_site_is_kept_under_its_registrable_name():
    assert site_of("www.shop.example") == "shop.example"
    assert site_of("login.musinsa.com") == "musinsa.com"
    assert site_of("www.shop.co.kr") == "shop.co.kr"
    assert site_of("m.coupang.com.") == "coupang.com"


async def test_the_owner_signs_in_and_keeps_it_with_one_passkey(hosted, monkeypatch):
    calls = broker(monkeypatch, kept_after=2)
    page, handed = browser(monkeypatch)
    approve, events, _ = channel()
    with approval_channel_scope(approve):
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
    assert answer.success, answer.error
    assert "stays signed in to shop.example" in answer.output
    card = next(d for e, d in events if e == "approval_request")
    assert card["login"] == {"site": "shop.example", "url": "https://www.shop.example/login", "sessionId": "conv-1",
                             "keepUrl": "https://approve.example.com/keep/shop.example"}
    assert card["timeoutMs"] >= 15 * 60 * 1000, "signing in can take a while"
    assert handed == [True, False], "the browser goes to the owner and comes back"
    assert page.sizes == [login_module.LOGIN_VIEWPORT, {"width": 1280, "height": 800}]
    operations = [op for op, _ in calls]
    assert operations.index("private") < operations.index("keep"), "the web closes once signed in"
    asked = [p for op, p in calls if op == "login"]
    assert asked[-1]["hosts"] == ["img.shop.example", "shop.example", "www.shop.example"], "never the tracker"
    sealed = next(p for op, p in calls if op == "keep")
    kept = json.loads(sealed["state"])
    assert {c["name"] for c in kept["cookies"]} == {"sid", "cdn"}
    assert [o["origin"] for o in kept["origins"]] == ["https://www.shop.example"]
    record = saved()["shop.example"]
    assert record["sealed"] == SEALED and "s3cret" not in json.dumps(record)
    path = login_module.logins_dir() / "shop.example.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_a_sign_in_not_kept_still_closes_the_web_and_keeps_nothing(hosted, monkeypatch):
    calls = broker(monkeypatch)
    browser(monkeypatch)
    approve, _, pending = channel()
    with approval_channel_scope(approve):
        presser = asyncio.create_task(press(pending, "approve_once"))
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
        await presser
    assert answer.success and "isn't kept" in answer.output
    assert ("private", {}) in calls, "a signed-in page is the owner's data either way"
    assert not any(op == "keep" for op, _ in calls) and saved() == {}


async def test_if_the_broker_cant_record_the_sign_in_the_browser_closes(hosted, monkeypatch):
    calls = broker(monkeypatch, kept_after=0, private_fails=True)
    browser(monkeypatch)
    approve, _, _ = channel()
    released = AsyncMock()
    monkeypatch.setattr(BrowserSession, "release_resources", released)
    with approval_channel_scope(approve):
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
    assert not answer.success and "browser was closed" in answer.error
    assert released.await_count == 1
    assert not any(op == "keep" for op, _ in calls)


async def test_not_now_hands_the_browser_back_and_says_not_to_ask_again(hosted, monkeypatch):
    broker(monkeypatch)
    _, handed = browser(monkeypatch)
    approve, events, pending = channel()
    with approval_channel_scope(approve):
        presser = asyncio.create_task(press(pending, "deny"))
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
        await presser
    assert not answer.success and "Don't ask again" in answer.error
    assert handed == [True, False]
    assert [e for e, _ in events][-1] == "approval_closed"


async def test_a_kept_site_opens_with_its_session_and_shows_no_card(hosted, monkeypatch):
    broker(monkeypatch)
    login_module._write("shop.example", ["shop.example", "www.shop.example"], SEALED)
    _, handed = browser(monkeypatch)
    approve, events, _ = channel()
    with approval_channel_scope(approve):
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/"))
    assert answer.success and "kept sign-in" in answer.output and "again=true" in answer.output
    assert events == [] and handed == []


async def test_signing_in_again_to_a_kept_site_needs_no_passkey(hosted, monkeypatch):
    calls = broker(monkeypatch, kept_at_first=True)
    login_module._write("shop.example", ["shop.example"], SEALED)
    browser(monkeypatch)
    approve, events, pending = channel()
    with approval_channel_scope(approve):
        presser = asyncio.create_task(press(pending, "approve_once"))
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login", again=True))
        await presser
    card = next(d for e, d in events if e == "approval_request")
    assert card["login"]["keepUrl"] == "", "Done, not the approval page"
    assert answer.success and "stays signed in" in answer.output
    assert any(op == "keep" for op, _ in calls)


async def test_without_the_app_there_is_no_one_to_sign_in(hosted, monkeypatch):
    broker(monkeypatch)
    browser(monkeypatch)
    answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
    assert not answer.success and "Rune app" in answer.error
    approve, events, _ = channel()
    with approval_channel_scope(approve), parking_scope(AsyncMock()):
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
    assert not answer.success and events == [], "a background run doesn't wait for a sign-in"


async def test_a_local_install_signs_in_with_take_control(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNE_CONNECTOR_URL", raising=False)
    answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
    assert not answer.success and "take control" in answer.error
    assert not (await browser_login(BrowserLoginParams(url="file:///etc/passwd"))).success


async def test_a_card_left_alone_times_out(hosted, monkeypatch):
    monkeypatch.setattr("rune.api.approvals.LOGIN_TIMEOUT_MS", 50)
    broker(monkeypatch)
    _, handed = browser(monkeypatch)
    approve, _, _ = channel(timeout_ms=50)
    with approval_channel_scope(approve):
        answer = await browser_login(BrowserLoginParams(url="https://www.shop.example/login"))
    assert not answer.success and "in time" in answer.error and handed == [True, False]


class Response:
    def __init__(self, status=200, headers=None, body=b"<html><head><title>shop</title></head></html>"):
        self.status, self.headers, self._body = status, headers or {"content-type": "text/html"}, body

    async def body(self):
        return self._body


class Route:
    def __init__(self, response=None):
        self.done, self.response = [], response or Response()

    async def fallback(self):
        self.done.append("fallback")

    async def fetch(self, **options):
        self.done.append(("fetch", options))
        return self.response

    async def fulfill(self, response=None, **values):
        self.done.append(("fulfill", values.get("body") if values else response))


def request(url, navigation=True):
    return type("Request", (), {"url": url, "is_navigation_request": lambda self: navigation})()


async def test_a_kept_session_opens_only_when_its_site_first_loads(hosted, monkeypatch):
    login_module._write("shop.example", ["shop.example", "www.shop.example"], SEALED)
    opened = []

    async def call(operation, params):
        opened.append(params)
        return {"kept": True, "state": json.dumps({"cookies": STATE["cookies"][:1], "origins": STATE["origins"][:1]})}

    monkeypatch.setattr("rune.connectors.client.call_broker", AsyncMock(side_effect=call))
    context = Context()
    await attach(context)
    [(matcher, handle)] = context.routes
    assert matcher("https://www.shop.example/a") and not matcher("https://news.example/")
    assert opened == [], "nothing opens before the site is visited: opening closes the web"
    first = Route()
    await handle(first, request("https://www.shop.example/"))
    assert opened == [{"site": "shop.example", "sealed": SEALED}]
    assert context.cookies == STATE["cookies"][:1]
    assert first.done[0] == ("fetch", {"max_redirects": 0}), "fetched again with the cookies"
    body = first.done[1][1]
    assert body.startswith(b"<html><head><script>") and b'"token"' in body, "storage before the page's own scripts"
    await asyncio.sleep(0)
    assert "token" in context.scripts[0], "and in every page after it"
    later = Route()
    await handle(later, request("https://www.shop.example/b"))
    await handle(later, request("https://www.shop.example/img.png", navigation=False))
    assert later.done == ["fallback", "fallback"] and len(opened) == 1


async def test_a_site_the_owner_signed_rune_out_of_is_dropped(hosted, monkeypatch):
    login_module._write("shop.example", ["shop.example"], SEALED)
    monkeypatch.setattr("rune.connectors.client.call_broker", AsyncMock(return_value={"kept": False}))
    context = Context()
    await attach(context)
    await context.routes[0][1](Route(), request("https://shop.example/"))
    assert context.cookies == [] and saved() == {}


async def test_keep_refused_by_the_broker_stores_nothing(hosted, monkeypatch):
    monkeypatch.setattr("rune.connectors.client.call_broker",
                        AsyncMock(return_value={"kept": False, "approval_url": "https://approve.example.com/"}))
    assert await keep(Context(), "shop.example", ["shop.example"]) is False
    assert saved() == {}


async def test_a_local_browser_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNE_CONNECTOR_URL", raising=False)
    context = Context()
    await attach(context)
    assert context.routes == []


async def test_a_redirect_within_the_site_is_followed_where_the_storage_can_still_go(hosted, monkeypatch):
    login_module._write("shop.example", ["shop.example", "www.shop.example"], SEALED)
    state = json.dumps({"cookies": [], "origins": STATE["origins"][:1]})
    monkeypatch.setattr("rune.connectors.client.call_broker", AsyncMock(return_value={"kept": True, "state": state}))
    context = Context()
    await attach(context)
    handle = context.routes[0][1]
    hop = Route(Response(302, {"location": "https://www.shop.example/home"}))
    await handle(hop, request("https://shop.example/"))
    assert hop.done[1] == ("fulfill", '<script>location.replace("https://www.shop.example/home")</script>')
    away = Route(Response(302, {"location": "https://sso.other.example/"}))
    await handle(away, request("https://www.shop.example/login"))
    assert isinstance(away.done[1][1], Response), "a redirect out of the site goes to the page as it was"
    arrive = Route()
    await handle(arrive, request("https://www.shop.example/home"))
    assert b'"token"' in arrive.done[1][1]
    after = Route()
    await handle(after, request("https://www.shop.example/next"))
    assert after.done == ["fallback"], "once in, the rest load as usual"


def test_stored_values_cannot_close_the_script_they_are_put_in():
    script = login_module._storage_script([{"origin": "https://shop.example",
                                            "localStorage": [{"name": "x", "value": "</script><img src=x>"}]}])
    assert "</script>" not in script
