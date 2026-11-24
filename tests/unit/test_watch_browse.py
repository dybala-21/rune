"""A page opened in RUNE's own browser: when the browser is started, what is read from it, and what
is never asked, reached or kept. The browser here is a pretend one; a real one is driven in
tests/integration/test_watch_browser.py."""

import asyncio
import socket
import sys
from types import SimpleNamespace

import pytest

from rune.watch import browse
from rune.watch.browse import Browser, Window
from rune.watch.item import Unread
from tests.unit.test_watch_gate import Far, tunnel

COAT = "https://shop.example/p/coat"
OUT = "http://127.0.0.1:9"
MARKUP = "<html><body>a page as its scripts left it</body></html>"


class Shop:
    """What a browser gets at an address: the documents the page becomes, where it ends, its markup."""

    def __init__(self) -> None:
        self.documents = [200]
        self.ends_at: str | None = None
        self.markup = MARKUP
        self.fails: BaseException | None = None
        self.never_quiet = False
        self.waits = 0.0
        # The documents are not told as they come: only what opening the page answered is known.
        self.unheard = False


class Page:
    def __init__(self, shop: Shop) -> None:
        self.shop, self.url = shop, "about:blank"
        self.went: list[str] = []
        self.heard: dict[str, object] = {}

    def on(self, event: str, handler: object) -> None:
        self.heard[event] = handler

    async def goto(self, url: str, **how: object):
        self.went.append(url)
        if url == "about:blank":
            self.url = url
            return None
        if self.shop.waits:
            await asyncio.sleep(self.shop.waits)
        if self.shop.fails is not None:
            raise self.shop.fails
        frame = SimpleNamespace(parent_frame=None)
        for status in () if self.shop.unheard else self.shop.documents:
            request = SimpleNamespace(is_navigation_request=lambda: True, frame=frame)
            self.heard["response"](SimpleNamespace(status=status, request=request))
        # Neither a picture's answer nor a frame's document inside the page is a document the page became.
        picture = SimpleNamespace(is_navigation_request=lambda: False, frame=frame)
        inner = SimpleNamespace(is_navigation_request=lambda: True, frame=SimpleNamespace(parent_frame=frame))
        for other in (picture, inner):
            self.heard["response"](SimpleNamespace(status=404, request=other))
        self.url = self.shop.ends_at or url
        return SimpleNamespace(status=self.shop.documents[0])

    async def wait_for_load_state(self, state: str, timeout: float) -> None:
        if self.shop.never_quiet:
            raise RuntimeError("Timeout 8000ms exceeded")

    async def content(self) -> str:
        return self.shop.markup

    async def evaluate(self, script: str) -> object:
        pytest.fail("the page was asked to run something")

    async def close(self) -> None:
        pytest.fail("the page being read was closed")


class Context:
    def __init__(self) -> None:
        self.heard: dict[str, object] = {}
        self.cleared = 0

    def on(self, event: str, handler: object) -> None:
        self.heard[event] = handler

    async def clear_cookies(self) -> None:
        self.cleared += 1


class Desk:
    """The machine a run reads on: how often its browser was started and shut, and the time it is."""

    def __init__(self) -> None:
        self.shop = Shop()
        self.started = self.shut = 0
        self.context, self.page = Context(), Page(self.shop)
        self.clock = 0.0
        self.pauses: list[float] = []
        self.addresses: dict[str, object] = {}
        self.looked_up: list[str] = []
        # The way out each started browser was given.
        self.outs: list[str] = []

    async def open(self, out: str) -> Window:
        self.started += 1
        self.outs.append(out)

        async def close() -> None:
            self.shut += 1

        return Window(self.context, self.page, close)

    async def resolve(self, host: str) -> str:
        self.looked_up.append(host)
        answer = self.addresses.get(host, "203.0.113.7")
        if isinstance(answer, BaseException):
            raise answer
        return str(answer)

    async def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.clock += seconds

    def browser(self, **more: object) -> Browser:
        return Browser(**{"open": self.open, "resolve": self.resolve, "pause": self.pause,
                          "clock": lambda: self.clock, **more})

    async def refused(self, url: str = COAT, *, registering: bool = False) -> tuple[str, int]:
        async with self.browser() as browser:
            with pytest.raises(Unread) as refusal:
                await browser.page(url, registering=registering)
        return refusal.value.kind, refusal.value.status


@pytest.fixture
def desk(monkeypatch) -> Desk:
    monkeypatch.delenv(browse.SWITCH)
    return Desk()


async def test_the_browser_is_started_by_the_first_page_that_needs_it_and_shut_when_the_run_ends(desk):
    async with desk.browser() as browser:
        assert desk.started == 0, "a run that opens no page starts no browser"
        first = await browser.page(COAT, registering=False)
        await browser.page("https://shop.example/p/mug", registering=False)
        assert (desk.started, desk.shut) == (1, 0), "one window for the run"
    assert desk.shut == 1
    assert (first.url, first.status, first.text) == (COAT, 200, MARKUP)

    async with desk.browser():
        pass
    assert (desk.started, desk.shut) == (1, 1), "nothing was started, so nothing is shut"


async def test_no_browser_is_started_where_it_is_switched_off_or_the_machine_is_hosted(desk, monkeypatch):
    monkeypatch.setenv(browse.SWITCH, "0")
    assert not browse.usable() and await desk.refused() == ("no_browser", 0)
    monkeypatch.delenv(browse.SWITCH)
    assert browse.usable()
    monkeypatch.setattr("rune.cloud.boundary.hosted", lambda: True)
    assert not browse.usable() and await desk.refused() == ("no_browser", 0)
    assert desk.started == 0


async def test_a_machine_with_no_browser_to_start_says_so(desk):
    outs: list[str] = []

    async def none(out: str) -> Window:
        outs.append(out)
        raise Unread("no_browser")

    async with desk.browser(open=none) as browser:
        with pytest.raises(Unread) as refusal:
            await browser.page(COAT, registering=True)
        with pytest.raises(OSError):
            await tunnel(outs[0], "www.musinsa.com:443")   # the way out made for it is shut again at once
    assert refusal.value.kind == "no_browser"


async def test_a_browser_that_cannot_be_set_up_is_shut_and_so_is_its_way_out(desk):
    def gone(event: str, handler: object) -> None:
        raise RuntimeError("the browser went away")

    desk.context.on = gone
    async with desk.browser() as browser:
        with pytest.raises(RuntimeError):
            await browser.page(COAT, registering=True)
        assert (desk.started, desk.shut) == (1, 1)
        with pytest.raises(OSError):
            await tunnel(desk.outs[0], "www.musinsa.com:443")
    assert desk.shut == 1, "and not shut a second time when the run ends"


@pytest.mark.parametrize(("status", "kind"), [(403, "blocked"), (401, "blocked"), (429, "blocked"), (404, "gone"),
                                              (410, "gone"), (503, "server"), (204, "error"), (302, "error")])
async def test_what_a_document_that_is_not_the_page_means(desk, status, kind):
    desk.shop.documents = [status]
    assert await desk.refused() == (kind, status)


async def test_the_page_is_judged_by_the_last_document_it_became(desk):
    # A shop's check let the browser through by itself and loaded the page after it.
    desk.shop.documents = [403, 200]
    async with desk.browser() as browser:
        assert (await browser.page(COAT, registering=False)).status == 200
    # And one that was sent on to a refusal is refused, whatever came first.
    desk.shop.documents = [200, 403]
    assert await desk.refused() == ("blocked", 403)


async def test_a_page_whose_documents_were_not_heard_is_judged_by_what_opening_it_answered(desk):
    async with desk.browser() as browser:
        await browser.page(COAT, registering=False)
        # Nothing is heard of the next page: the one before it does not answer for it.
        desk.shop.documents, desk.shop.unheard = [403], True
        with pytest.raises(Unread) as refusal:
            await browser.page("https://shop.example/p/mug", registering=False)
    assert (refusal.value.kind, refusal.value.status) == ("blocked", 403)


async def test_a_watched_page_must_end_on_its_own_site_and_a_new_one_on_a_public_https_site(desk):
    desk.shop.ends_at = "https://other.example/p/coat"
    assert await desk.refused() == ("moved", 0)
    async with desk.browser() as browser:
        page = await browser.page("https://sho.rt/c", registering=True)   # a short link, when it is set up
    assert page.url == "https://other.example/p/coat"

    desk.addresses["lan.example"] = ValueError("not public")
    desk.shop.ends_at = "https://lan.example/admin"
    assert await desk.refused(registering=True) == ("private", 0)
    for odd in ("http://shop.example/p/coat", "https://shop.example:8443/p/coat", "https://192.0.2.1/p"):
        desk.shop.ends_at = odd
        assert await desk.refused(registering=True) == ("address", 0), odd


async def test_a_host_that_may_not_be_asked_is_refused_before_any_browser_is_started(desk):
    desk.addresses |= {"lan.example": ValueError("not public"), "nowhere.example": socket.gaierror()}
    assert await desk.refused("https://lan.example/p/1", registering=True) == ("private", 0)
    assert await desk.refused("https://nowhere.example/p/1", registering=True) == ("unreachable", 0)
    assert await desk.refused("https://www.musinsa.com/products/1", registering=True) == ("denied", 0)
    assert await desk.refused("http://shop.example/p/coat", registering=True) == ("address", 0)
    assert desk.started == 0


async def test_the_browser_is_given_one_way_out_and_it_reaches_public_hosts_only(desk):
    async with Far() as far:
        # cdn.example is found at this machine here: a tunnel to it shows whose lookup decides.
        desk.addresses |= {"cdn.example": "127.0.0.1", "lan.example": ValueError("not public"),
                           "192.168.0.1": ValueError("not public"), "nowhere.example": socket.gaierror()}
        async with desk.browser() as browser:
            await browser.page(COAT, registering=False)
            [out] = desk.outs
            for _ in range(2):
                status, _, back = await tunnel(out, f"cdn.example:{far.port}")
                assert status == 200
                back.close()
            assert far.reached == 2 and desk.looked_up.count("cdn.example") == 1, "found once, reached there"
            # A site whose terms forbid it is kept out however the browser writes its name.
            for kept_out in ("lan.example:443", "192.168.0.1:80", "nowhere.example:443", "www.musinsa.com:443",
                             "www.musinsa.com.:443", "WWW.MUSINSA.COM:443", "pay.naver.com:443"):
                status, _, back = await tunnel(out, kept_out)
                assert status == 403, kept_out
                back.close()
            await browser.page("https://shop.example/p/mug", registering=False)
            assert desk.outs == [out], "one way out for the run"
        with pytest.raises(OSError):
            await tunnel(out, "www.musinsa.com:443")   # shut with the browser
        assert far.reached == 2


async def test_a_page_that_does_not_come_is_given_up_and_one_that_never_goes_quiet_is_read_as_it_stands(desk):
    desk.shop.fails = type("TimeoutError", (Exception,), {})("Timeout 25000ms exceeded")   # the browser's own
    assert await desk.refused() == ("timeout", 0)
    desk.shop.fails = RuntimeError("net::ERR_CONNECTION_REFUSED at https://shop.example/p/coat")
    assert await desk.refused() == ("unreachable", 0)

    desk.shop.fails, desk.shop.never_quiet = None, True
    async with desk.browser() as browser:
        assert (await browser.page(COAT, registering=False)).text == MARKUP

    desk.shop.waits = 0.2
    async with desk.browser(deadline=0.05) as browser:
        with pytest.raises(Unread) as refusal:
            await browser.page(COAT, registering=False)
    assert refusal.value.kind == "timeout"


async def test_a_page_too_large_is_not_read(desk, monkeypatch):
    monkeypatch.setattr(browse, "MAX_PAGE", 100)
    desk.shop.markup = "x" * 100
    async with desk.browser() as browser:
        assert len((await browser.page(COAT, registering=False)).text) == 100
    desk.shop.markup = "x" * 101
    assert await desk.refused() == ("too_large", 0)


async def test_every_page_is_put_down_before_the_next_and_its_cookies_are_not_carried_on(desk):
    async with desk.browser() as browser:
        await browser.page(COAT, registering=False)
        assert desk.page.went == [COAT, "about:blank"] and desk.context.cleared == 1
        desk.shop.documents = [403]
        with pytest.raises(Unread):
            await browser.page("https://shop.example/p/mug", registering=False)
        assert desk.page.went[-1] == "about:blank" and desk.context.cleared == 2, "also after a page that was refused"


async def test_two_pages_of_one_site_are_opened_a_pause_apart(desk):
    async with desk.browser() as browser:
        await browser.page(COAT, registering=False)
        desk.clock += 0.5
        await browser.page("https://shop.example/p/mug", registering=False)
        await browser.page("https://other.example/p/1", registering=False)
    assert desk.pauses == [1.5], "the rest of two seconds for the same host, none for another"


async def test_a_page_a_script_opened_beside_it_is_closed_unread(desk):
    closed: list[str] = []

    async def close() -> None:
        closed.append("extra")

    async with desk.browser() as browser:
        await browser.page(COAT, registering=False)
        await desk.context.heard["page"](SimpleNamespace(close=close))
        await desk.context.heard["page"](desk.page)   # the page being read is not an extra one
    assert closed == ["extra"]


# -- Starting the real browser, with a pretend Playwright -------------------------------


class Playwright:
    """What starting a browser asks of Playwright, and whether this machine has a screen for a window."""

    def __init__(self, *, screen: bool = True, none: bool = False) -> None:
        self.screen, self.none = screen, none
        self.launched: list[dict] = []
        self.contexts: list[dict] = []
        self.sent: list[tuple[str, dict | None]] = []
        self.closed: list[str] = []
        self.breaks = False

    def module(self) -> SimpleNamespace:
        return SimpleNamespace(async_playwright=lambda: SimpleNamespace(start=self._start))

    async def _start(self) -> SimpleNamespace:
        return SimpleNamespace(chromium=SimpleNamespace(launch=self._launch), stop=lambda: self._closed("driver"))

    async def _launch(self, **how: object) -> SimpleNamespace:
        self.launched.append(how)
        if self.none or (not how["headless"] and not self.screen):
            raise RuntimeError("Missing X server or $DISPLAY")
        return SimpleNamespace(new_browser_cdp_session=self._session, new_context=self._context,
                               close=lambda: self._closed("browser"))

    async def _session(self, *page: object) -> SimpleNamespace:
        async def send(method: str, params: dict | None = None) -> dict:
            self.sent.append((method, params))
            return {"userAgent": "Mozilla/5.0 Chrome/145.0.0.0 Safari/537.36", "windowId": 7}

        return SimpleNamespace(send=send, detach=lambda: self._closed("session"))

    async def _context(self, **how: object) -> SimpleNamespace:
        self.contexts.append(how)
        if self.breaks:
            raise RuntimeError("the browser went away")
        return SimpleNamespace(new_page=lambda: self._closed("page opened"), new_cdp_session=self._session)

    async def _closed(self, what: str) -> None:
        self.closed.append(what)


@pytest.fixture
def playwright(monkeypatch):
    def given(**how: object) -> Playwright:
        pretend = Playwright(**how)
        monkeypatch.setitem(sys.modules, "playwright.async_api", pretend.module())
        return pretend

    return given


async def test_the_browser_is_started_with_a_minimized_window_under_runes_name_and_takes_nothing(playwright):
    pretend = playwright()
    window = await browse.open_window(OUT)
    assert pretend.launched == [{"headless": False, "chromium_sandbox": True,
                                 "args": ["--mute-audio", "--webrtc-ip-handling-policy=disable_non_proxied_udp",
                                          "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"],
                                 "proxy": {"server": OUT, "bypass": "<-loopback>"}}]
    assert pretend.contexts == [{"user_agent": f"Mozilla/5.0 Chrome/145.0.0.0 Safari/537.36 {browse.NAME}",
                                 "accept_downloads": False, "service_workers": "block"}]
    assert ("Browser.setWindowBounds", {"windowId": 7, "bounds": {"windowState": "minimized"}}) in pretend.sent
    assert "RUNE/" in browse.NAME and "#fetcher" in browse.NAME
    await window.close()
    assert pretend.closed[-2:] == ["browser", "driver"]


async def test_a_machine_with_no_screen_runs_the_browser_without_a_window(playwright):
    pretend = playwright(screen=False)
    await browse.open_window(OUT)
    assert [how["headless"] for how in pretend.launched] == [False, True]
    assert all(how["proxy"] == {"server": OUT, "bypass": "<-loopback>"} for how in pretend.launched)
    assert not [method for method, _ in pretend.sent if method == "Browser.setWindowBounds"], "no window to minimize"

    asked = playwright()
    await browse.open_window(OUT, screen=False)
    assert [how["headless"] for how in asked.launched] == [True], "nor is a window tried where none is wanted"


async def test_a_browser_that_cannot_be_started_is_no_browser_and_nothing_is_left_running(playwright, monkeypatch):
    pretend = playwright(none=True)
    with pytest.raises(Unread) as refusal:
        await browse.open_window(OUT)
    assert refusal.value.kind == "no_browser" and pretend.closed == ["driver"]

    broken = playwright()
    broken.breaks = True
    with pytest.raises(RuntimeError):
        await browse.open_window(OUT)
    assert broken.closed == ["browser", "driver"], "what was started is shut before the error goes on"

    monkeypatch.setitem(sys.modules, "playwright.async_api", None)   # the browser extra is not installed
    with pytest.raises(Unread) as refusal:
        await browse.open_window(OUT)
    assert refusal.value.kind == "no_browser"
