"""The watch's browser, driven for real: Chromium opens pretend shops and the watch's own code reads them."""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial

import pytest

from rune.watch import browse
from rune.watch.browse import NAME, Browser, Window, open_window
from rune.watch.item import BROWSER, Unread
from rune.watch.observe import observe
from tests.unit.test_watch_fetch import Web

SHOP = "https://shop.example"
# The price is put into the page by a script, after the page asked the shop for it.
LATE = """<html><head><title>Coat</title></head><body><script>
document.documentElement.dataset.driven = String(navigator.webdriver);
fetch('/api/price').then(answer => answer.json()).then(data => {
  const block = document.createElement('script');
  block.type = 'application/ld+json';
  block.textContent = JSON.stringify({'@type': 'Product', 'name': 'Coat',
                                      'offers': {'price': data.price, 'priceCurrency': 'KRW'}});
  document.head.appendChild(block);
});
</script></body></html>"""
SETS = "<html><body><script>document.cookie = 'seen=1; path=/';</script></body></html>"
# The ways a page has of sending a browser to the machine it runs on or to that machine's network:
# asked for straight, by a redirect, by name, and in packets that are not web requests at all.
REACHES_IN = """<html><body>
<img src="http://127.0.0.1:{tcp}/straight"><img src="https://127.0.0.1:{tcp}/straight">
<img src="/sent/picture"><img src="https://lan.example:{tcp}/reboot">
<script>
fetch('/sent/fetch').catch(() => {{}}); fetch('http://localhost:{tcp}/api').catch(() => {{}});
const call = new RTCPeerConnection({{iceServers: [{{urls: 'stun:127.0.0.1:{udp}'}}]}});
call.createDataChannel('x'); call.createOffer().then(offer => call.setLocalDescription(offer));
</script></body></html>"""
REACHES_OUT = '<html><body><img src="https://open.example:{tcp}/x"></body></html>'


class Inside:
    """This machine, listening: everything that got to it."""

    def __init__(self) -> None:
        self.reached: list[str] = []
        self.tcp = self.udp = 0

    async def __aenter__(self) -> "Inside":
        loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(self._connected, "127.0.0.1", 0)
        self.tcp = self._server.sockets[0].getsockname()[1]
        self._packets, _ = await loop.create_datagram_endpoint(lambda: _Packets(self.reached), local_addr=("127.0.0.1", 0))
        self.udp = self._packets.get_extra_info("sockname")[1]
        return self

    async def __aexit__(self, *why: object) -> None:
        self._server.close()
        self._packets.close()

    async def _connected(self, sent: asyncio.StreamReader, back: asyncio.StreamWriter) -> None:
        self.reached.append("connection")
        back.close()


class _Packets(asyncio.DatagramProtocol):
    def __init__(self, reached: list[str]) -> None:
        self.reached = reached

    def datagram_received(self, data: bytes, sender: object) -> None:
        self.reached.append("packet")


class Shop:
    """A pretend shop behind the browser: every request that reached it, and what it answered."""

    def __init__(self, inside: Inside) -> None:
        self.asked: list[tuple[str, dict[str, str]]] = []
        self.inside = inside

    async def answer(self, route) -> None:
        ports = {"tcp": self.inside.tcp, "udp": self.inside.udp}
        request = route.request
        self.asked.append((request.url, await request.all_headers()))
        path = request.url.removeprefix(SHOP)
        if path == "/p/late":
            await route.fulfill(body=LATE, content_type="text/html", headers={"set-cookie": "visit=1; Path=/"})
        elif path == "/api/price":
            await route.fulfill(body=json.dumps({"price": 100}), content_type="application/json")
        elif path == "/p/sets":
            await route.fulfill(body=SETS, content_type="text/html")
        elif path == "/p/reaches-in":
            await route.fulfill(body=REACHES_IN.format(**ports), content_type="text/html")
        elif path == "/p/reaches-out":
            await route.fulfill(body=REACHES_OUT.format(**ports), content_type="text/html")
        elif path == "/p/sent-in" or path.startswith("/sent/"):
            await route.fulfill(status=302, headers={"location": f"http://127.0.0.1:{self.inside.tcp}/redirected"})
        elif path == "/p/refused":
            await route.fulfill(status=403, body="<html>Are you a person?</html>", content_type="text/html")
        else:
            await route.fulfill(status=404, body="")

    def sent(self, path: str) -> dict[str, str]:
        return next(headers for url, headers in self.asked if url == SHOP + path)


@pytest.fixture
async def inside():
    async with Inside() as listening:
        yield listening


@pytest.fixture
async def shop(monkeypatch, inside):
    pytest.importorskip("playwright.async_api")
    monkeypatch.delenv(browse.SWITCH)
    return Shop(inside)


@pytest.fixture
def looked_up() -> list[str]:
    return []


@pytest.fixture
async def browser(shop, looked_up):
    async def opened(out: str) -> Window:
        try:
            window = await open_window(out, screen=False)
        except Unread:
            pytest.skip("no Chromium to start on this machine")
        # The shop's own pages are answered here. Whatever else the browser asks for goes the way
        # it would for a real shop: out through the watch's gate.
        await window.context.route(f"{SHOP}/**", shop.answer)
        return window

    async def resolve(host: str) -> str:
        looked_up.append(host)
        if host == "shop.example":
            return "203.0.113.7"
        if host == "open.example":
            return "127.0.0.1"   # stands for a public host: the gate connects where the lookup says
        raise ValueError("not public")

    async with Browser(open=opened, resolve=resolve, pause=partial(_no_wait)) as started:
        yield started


async def _no_wait(seconds: float) -> None:
    return None


async def test_a_price_a_script_puts_in_is_read_and_the_browser_says_whose_it_is(shop, browser):
    web = Web()
    web.put("/robots.txt", "User-agent: *\nDisallow: /\n")
    async with web.reader() as reader:
        seen = await observe(reader, f"{SHOP}/p/late", registering=True, now=datetime(2026, 11, 22, tzinfo=UTC),
                             browser=browser)
    assert (seen.reading.price.low, seen.reading.price.currency, seen.via, seen.http_status) == (Decimal(100), "KRW", BROWSER, 200)
    agent = shop.sent("/p/late")["user-agent"]
    assert agent.endswith(NAME) and "Chrome/" in agent, agent
    assert shop.sent("/api/price")["user-agent"] == agent, "the page's own requests carry the name too"
    page = await browser.page(f"{SHOP}/p/late", registering=False)
    assert 'data-driven="true"' in page.text, "the page can tell a program drives the browser"


async def test_nothing_a_page_set_is_sent_with_the_next(shop, browser):
    await browser.page(f"{SHOP}/p/late", registering=False)
    await browser.page(f"{SHOP}/p/sets", registering=False)
    await browser.page(f"{SHOP}/p/late", registering=False)
    # Within a page its shop's cookie works as in any browser; the next page starts without it.
    cookies = [(url.removeprefix(SHOP), headers.get("cookie")) for url, headers in shop.asked]
    assert cookies == [("/p/late", None), ("/api/price", "visit=1"), ("/p/sets", None),
                       ("/p/late", None), ("/api/price", "visit=1")]


async def test_nothing_gets_from_a_page_to_the_machine_the_browser_runs_on(shop, browser, inside, looked_up):
    page = await browser.page(f"{SHOP}/p/reaches-in", registering=False)
    assert page.status == 200, "the page itself is read; what it reached for is not given"
    with pytest.raises(Unread) as refusal:
        await browser.page(f"{SHOP}/p/sent-in", registering=True)   # the address itself sends the browser inside
    # Which of the two depends on how the browser shows the request the gate refused.
    assert refusal.value.kind in {"address", "unreachable"}
    assert inside.reached == []
    assert {"127.0.0.1", "lan.example"} <= set(looked_up), "they were asked for at the gate, and refused there"

    # The same listener hears what the gate does let through.
    await browser.page(f"{SHOP}/p/reaches-out", registering=False)
    assert set(inside.reached) == {"connection"}


async def test_a_shop_that_asks_for_a_person_is_left_unread(shop, browser):
    with pytest.raises(Unread) as refusal:
        await browser.page(f"{SHOP}/p/refused", registering=True)
    assert (refusal.value.kind, refusal.value.status) == ("blocked", 403)
    assert [url for url, _ in shop.asked] == [f"{SHOP}/p/refused"], "asked once, and nothing was tried after it"
