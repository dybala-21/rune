"""Reading a page for the watch: who it says it is, what it obeys, and what it never follows or keeps."""

import asyncio
import gzip
import json
import socket
from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest

from rune.watch import fetch
from rune.watch.denied import DENIED, denied
from rune.watch.fetch import Reader, host_of
from rune.watch.item import Unread
from rune.watch.observe import observe

SHOP = "shop.example"
PRODUCT = ('<html><head><script type="application/ld+json">'
           + json.dumps({"@type": "Product", "name": "Coat", "offers": {"price": 100, "priceCurrency": "KRW"}})
           + "</script></head></html>")


class Body(httpx.AsyncByteStream):
    """An answer's bytes as they come off the wire: in pieces, not yet read."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        for start in range(0, len(self._data), 16_384):
            yield self._data[start:start + 16_384]


class Web:
    """A pretend web: what each host answers at each path, and every request that reached it."""

    def __init__(self) -> None:
        self.answers: dict[tuple[str, str], object] = {}
        self.requests: list[httpx.Request] = []
        self.clock = 0.0
        self.pauses: list[float] = []

    def put(self, path: str, body: bytes | str = b"", status: int = 200, host: str = SHOP, **headers: str) -> None:
        self.answers[(host, path)] = (status, headers, body.encode() if isinstance(body, str) else body)

    def redirect(self, path: str, to: str, host: str = SHOP, status: int = 302) -> None:
        self.put(path, status=status, host=host, location=to)

    def asked(self, host: str = SHOP) -> list[str]:
        return [r.url.raw_path.decode() for r in self.requests if r.headers["host"] == host]

    async def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.get((request.headers["host"], request.url.raw_path.decode()))
        if answer is None:
            return httpx.Response(404, stream=Body(b""))
        if callable(answer):
            return await answer(request)
        status, headers, body = answer
        return httpx.Response(status, headers=headers, stream=Body(body))

    async def _pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.clock += seconds

    def reader(self, **more: object) -> Reader:
        async def resolve(host: str) -> str:
            return "203.0.113.7"

        settings = {"client": httpx.AsyncClient(transport=httpx.MockTransport(self._answer)), "resolve": resolve,
                    "pause": self._pause, "clock": lambda: self.clock, **more}
        return Reader(**settings)


@pytest.fixture
def web() -> Web:
    return Web()


async def refused(web: Web, url: str, *, registering: bool = False, **more: object) -> str:
    async with web.reader(**more) as reader:
        with pytest.raises(Unread) as failed:
            await reader.page(url, registering=registering)
    return failed.value.kind


async def test_a_page_is_asked_for_under_runes_own_name_at_the_address_that_was_checked(web):
    web.put("/p/1?color=red", PRODUCT)
    async with web.reader() as reader:
        page = await reader.page(f"https://{SHOP}/p/1?color=red#reviews", registering=False)
    assert (page.url, page.status, page.text) == (f"https://{SHOP}/p/1?color=red#reviews", 200, PRODUCT)
    robots_txt, asked = web.requests
    assert robots_txt.url.raw_path == b"/robots.txt"
    # The connection goes to the checked address; the site's name travels in the TLS hello and the Host header.
    assert (asked.url.host, asked.headers["host"], asked.extensions["sni_hostname"]) == ("203.0.113.7", SHOP, SHOP)
    assert asked.headers["user-agent"].startswith("Mozilla/5.0 (compatible; RUNE/")
    assert "+https://github.com/dybala-21/rune#fetcher" in asked.headers["user-agent"]
    assert asked.headers["accept-encoding"] == "gzip" and "cookie" not in asked.headers


async def test_robots_txt_is_obeyed_and_read_once_for_a_host(web):
    web.put("/robots.txt", "User-agent: *\nDisallow: /cart\n")
    web.put("/p/1", PRODUCT)
    web.put("/p/2", PRODUCT)
    async with web.reader() as reader:
        await reader.page(f"https://{SHOP}/p/1", registering=False)
        await reader.page(f"https://{SHOP}/p/2", registering=False)
        with pytest.raises(Unread) as failed:
            await reader.page(f"https://{SHOP}/cart", registering=False)
    assert failed.value.kind == "robots"
    assert web.asked() == ["/robots.txt", "/p/1", "/p/2"]  # the refused page was never asked for


async def test_a_site_that_refuses_unnamed_programs_is_not_read_and_one_that_names_rune_is(web):
    web.put("/p/1", PRODUCT)
    web.put("/robots.txt", "User-agent: *\nDisallow: /\n\nUser-agent: Claude-User\nAllow: /p/\n")
    assert await refused(web, f"https://{SHOP}/p/1") == "robots"
    web.put("/robots.txt", "User-agent: *\nDisallow: /\n\nUser-agent: RUNE\nAllow: /p/\n")
    async with web.reader() as reader:
        assert (await reader.page(f"https://{SHOP}/p/1", registering=False)).status == 200


@pytest.mark.parametrize(("status", "kind"), [(403, "blocked"), (401, "blocked"), (429, "blocked"), (500, "robots")])
async def test_a_robots_txt_that_is_refused_or_broken_keeps_the_page_unread(web, status, kind):
    web.put("/robots.txt", status=status)
    web.put("/p/1", PRODUCT)
    assert await refused(web, f"https://{SHOP}/p/1") == kind
    assert web.asked() == ["/robots.txt"]


async def test_robots_txt_is_followed_through_redirects_and_cut_when_very_long(web):
    web.redirect("/robots.txt", f"https://www.{SHOP}/robots.txt")
    web.put("/robots.txt", b"User-agent: *\nDisallow: /cart\n" + b"# filler\n" * 80_000, host=f"www.{SHOP}")
    web.put("/p/1", PRODUCT)
    async with web.reader() as reader:
        assert (await reader.page(f"https://{SHOP}/p/1", registering=False)).status == 200
    assert await refused(web, f"https://{SHOP}/cart") == "robots"
    # One that leads nowhere it can be read is taken as a refusal.
    web.redirect("/robots.txt", "http://insecure.example/robots.txt")
    assert await refused(web, f"https://{SHOP}/p/1") == "robots"


@pytest.mark.parametrize(("status", "kind"), [
    (403, "blocked"), (401, "blocked"), (429, "blocked"), (404, "gone"), (410, "gone"),
    (500, "server"), (503, "server"), (204, "error"), (304, "error"), (418, "error"),
])
async def test_what_an_answer_that_is_not_the_page_means(web, status, kind):
    web.put("/p/1", status=status)
    async with web.reader() as reader:
        with pytest.raises(Unread) as failed:
            await reader.page(f"https://{SHOP}/p/1", registering=False)
    assert (failed.value.kind, failed.value.status) == (kind, status)


async def test_a_watched_page_must_stay_on_its_site(web):
    web.redirect("/p/1", "/p/1-new")
    web.put("/p/1-new", PRODUCT)
    async with web.reader() as reader:
        assert (await reader.page(f"https://{SHOP}/p/1", registering=False)).url == f"https://{SHOP}/p/1-new"
    web.redirect("/p/2", "https://other.example/p/2")
    web.put("/p/2", PRODUCT, host="other.example")
    assert await refused(web, f"https://{SHOP}/p/2") == "moved"
    assert web.asked("other.example") == []
    web.redirect("/loop", "/loop")
    assert await refused(web, f"https://{SHOP}/loop") == "moved"
    assert web.asked().count("/loop") == fetch.DAILY_HOPS + 1


async def test_a_page_being_registered_may_lead_to_another_site_and_every_hop_is_checked(web):
    web.redirect("/s/abc", "https://brand.example/p/9", host="short.example")   # a short link
    web.put("/p/9", PRODUCT, host="brand.example")
    async with web.reader() as reader:
        page = await reader.page("https://short.example/s/abc", registering=True)
    assert page.url == "https://brand.example/p/9"
    for target, kind in (("https://closed.example/p/1", "robots"), ("https://www.musinsa.com/products/1", "denied"),
                         ("http://brand.example/p/9", "address"), ("https://10.0.0.5/admin", "address"),
                         ("https://user:secret@brand.example/p/9", "address")):
        web.put("/robots.txt", "User-agent: *\nDisallow: /\n", host="closed.example")
        web.redirect("/s/next", target, host="short.example")
        assert await refused(web, "https://short.example/s/next", registering=True) == kind, target
    web.redirect("/s/loop", "/s/loop", host="short.example")
    assert await refused(web, "https://short.example/s/loop", registering=True) == "error"
    web.put("/s/bare", status=302, host="short.example")   # a redirect that names no target
    assert await refused(web, "https://short.example/s/bare", registering=True) == "error"


async def test_a_redirect_into_a_private_address_is_refused(web):
    async def resolve(host: str) -> str:
        if host == "inside.example":
            raise ValueError("Connector destination must resolve only to public addresses")
        return "203.0.113.7"

    web.redirect("/s/x", "https://inside.example/admin", host="short.example")
    assert await refused(web, "https://short.example/s/x", registering=True, resolve=resolve) == "private"
    assert web.asked("inside.example") == []


@pytest.mark.parametrize("address", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "100.64.0.1", "::1", "fd00::1"])
async def test_a_name_that_resolves_to_an_address_that_is_not_public_is_never_connected_to(web, monkeypatch, address):
    # One public address among the answers does not make the name public.
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    answers = [(family, socket.SOCK_STREAM, 6, "", (address, 443)),
               (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", AsyncMock(return_value=answers))
    client = httpx.AsyncClient(transport=httpx.MockTransport(web._answer))
    async with Reader(client=client, pause=web._pause, clock=lambda: web.clock) as reader:   # the real check
        with pytest.raises(Unread) as failed:
            await reader.page(f"https://{SHOP}/p/1", registering=True)
    assert failed.value.kind == "private" and web.requests == []


async def test_a_host_that_cannot_be_found_or_reached_is_said_to_be_unreachable(web):
    async def unknown(host: str) -> str:
        raise socket.gaierror(-2, "Name or service not known")

    assert await refused(web, f"https://{SHOP}/p/1", resolve=unknown) == "unreachable"

    async def closes(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused: secret-internal-name", request=request)

    web.answers[(SHOP, "/robots.txt")] = closes
    assert await refused(web, f"https://{SHOP}/p/1") == "unreachable"
    web.put("/robots.txt", status=404)
    web.answers[(SHOP, "/p/1")] = closes
    assert await refused(web, f"https://{SHOP}/p/1") == "unreachable"


async def test_an_answer_that_takes_too_long_altogether_is_given_up(web):
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, stream=Body(PRODUCT.encode()))

    web.answers[(SHOP, "/p/1")] = slow
    assert await refused(web, f"https://{SHOP}/p/1", deadline=0.05) == "timeout"

    async def stalls(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    web.answers[(SHOP, "/p/2")] = stalls
    assert await refused(web, f"https://{SHOP}/p/2") == "timeout"


async def test_a_page_too_large_is_not_read_in_part_and_one_that_inflates_is_stopped(web, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_PAGE", 100_000)
    web.put("/big", b"x" * 100_001)
    assert await refused(web, f"https://{SHOP}/big") == "too_large"
    web.put("/fits", b"x" * 100_000)
    async with web.reader() as reader:
        assert len((await reader.page(f"https://{SHOP}/fits", registering=False)).text) == 100_000
    bomb = gzip.compress(b"\0" * 50_000_000)   # 50 MB in about 50 KB
    assert len(bomb) < 100_000
    web.put("/bomb", bomb, **{"content-encoding": "gzip"})
    grown: list[int] = []
    original = fetch.zlib.decompressobj

    class Watched:
        def __init__(self, *args: object) -> None:
            self._inner = original(*args)

        def decompress(self, data: bytes, limit: int) -> bytes:
            out = self._inner.decompress(data, limit)
            grown.append(len(out))
            return out

        @property
        def unconsumed_tail(self) -> bytes:
            return self._inner.unconsumed_tail

    monkeypatch.setattr(fetch.zlib, "decompressobj", Watched)
    assert await refused(web, f"https://{SHOP}/bomb") == "too_large"
    assert sum(grown) <= 100_001   # never unpacked past the limit to find that out


async def test_a_gzipped_page_is_unpacked_and_other_packings_are_not_accepted(web):
    web.put("/p/1", gzip.compress(PRODUCT.encode()), **{"content-encoding": "gzip"})
    async with web.reader() as reader:
        assert (await reader.page(f"https://{SHOP}/p/1", registering=False)).text == PRODUCT
    web.put("/br", b"\x8b\x02\x80", **{"content-encoding": "br"})
    assert await refused(web, f"https://{SHOP}/br") == "error"
    web.put("/broken", b"not gzip at all", **{"content-encoding": "gzip"})
    assert await refused(web, f"https://{SHOP}/broken") == "error"


async def test_requests_to_one_host_are_spaced_and_a_sites_own_pause_is_honored_up_to_a_cap(web):
    for path in ("/p/1", "/p/2"):
        web.put(path, PRODUCT)
        web.put(path, PRODUCT, host="slow.example")
        web.put(path, PRODUCT, host="slower.example")
    web.put("/robots.txt", "User-agent: *\nCrawl-delay: 5\nDisallow: /cart\n", host="slow.example")
    web.put("/robots.txt", "User-agent: *\nCrawl-delay: 120\nDisallow: /cart\n", host="slower.example")
    async with web.reader() as reader:
        for host in (SHOP, "slow.example", "slower.example"):
            await reader.page(f"https://{host}/p/1", registering=False)
            await reader.page(f"https://{host}/p/2", registering=False)
    # Each host: robots.txt, then two pages. The first request to a host waits for nothing.
    assert web.pauses == [2.0, 2.0, 5.0, 5.0, 10.0, 10.0]


async def test_no_cookie_a_site_sets_is_sent_back(web):
    web.put("/p/1", PRODUCT, **{"set-cookie": "bucket=B; Path=/"})
    web.put("/p/2", PRODUCT)
    async with web.reader() as reader:
        await reader.page(f"https://{SHOP}/p/1", registering=False)
        await reader.page(f"https://{SHOP}/p/2", registering=False)
    assert all("cookie" not in request.headers for request in web.requests)


@pytest.mark.parametrize(("headers", "body"), [
    ({"content-type": "text/html; charset=EUC-KR"}, "<p>가격</p>".encode("euc-kr")),
    ({"content-type": "text/html"}, '<meta charset="euc-kr"><p>가격</p>'.encode("euc-kr")),
    ({"content-type": "text/html"}, '<meta http-equiv="Content-Type" content="text/html; charset=euc-kr"><p>가격</p>'.encode("euc-kr")),
    ({"content-type": "text/html; charset=no-such-encoding"}, "<p>가격</p>".encode()),
    ({}, "<p>가격</p>".encode()),
])
async def test_a_page_is_read_in_the_encoding_it_states(web, headers, body):
    web.put("/p/1", body, **headers)
    async with web.reader() as reader:
        assert "가격" in (await reader.page(f"https://{SHOP}/p/1", registering=False)).text


@pytest.mark.parametrize("url", [
    "http://shop.example/p/1", "ftp://shop.example/p/1", "https://shop.example:8443/p/1",
    "https://user@shop.example/p/1", "https://user:secret@shop.example/p/1", "https://203.0.113.7/p/1",
    "https://[2001:db8::1]/p/1", "https://localhost/p/1", "https:///p/1", "https://shop.example:port/p/1",
    "shop.example/p/1", "", "https://shop.example/" + "a" * 2048,
    # The same site under another spelling: a closing dot, or a "%" a browser turns into one.
    "https://shop.example./p/1", "https://shop.example%2e/p/1", "https://shop%20.example/p/1",
])
def test_an_address_the_watch_does_not_read(url):
    with pytest.raises(Unread) as failed:
        host_of(url)
    assert failed.value.kind == "address"


def test_a_hosts_name_is_taken_in_one_plain_form():
    assert host_of("https://Shop.Example:443/p/1") == "shop.example"
    assert host_of("https://bücher.example/p/1") == "xn--bcher-kva.example"
    korean = host_of("https://예시.한국/상품/1")
    assert korean.isascii() and all(label.startswith("xn--") for label in korean.split("."))


async def test_a_site_whose_terms_forbid_it_is_not_asked_at_all(web):
    assert await refused(web, "https://www.musinsa.com/products/7500264", registering=True) == "denied"
    assert await refused(web, "https://musinsa.com/products/7500264") == "denied"
    for shop in ("brand.naver.com/pam/products/1", "smartstore.naver.com/main/products/1",
                 "m.smartstore.naver.com/shop/products/1", "search.shopping.naver.com/catalog/1",
                 "product.29cm.co.kr/catalog/1", "www.soldout.co.kr/trade/detail/1"):
        assert await refused(web, f"https://{shop}", registering=True) == "denied", shop
    assert web.requests == []


async def test_a_short_link_into_a_site_whose_terms_forbid_it_stops_before_that_site(web):
    web.redirect("/abc", "https://brand.naver.com/pam/products/1", host="naver.me")
    assert await refused(web, "https://naver.me/abc", registering=True) == "denied"
    assert {request.headers["host"] for request in web.requests} == {"naver.me"}


@pytest.mark.parametrize("host", ["musinsa.com", "www.musinsa.com", "WWW.Musinsa.COM", "www.musinsa.com.",
                                  "naver.com", "brand.naver.com", "m.smartstore.naver.com."])
def test_a_listed_site_is_listed_however_its_name_is_spelled(host):
    assert denied(host)


@pytest.mark.parametrize("host", ["notmusinsa.com", "musinsa.com.shop.example", "musinsa.co", "naver.me",
                                  "navercorp.com", "xn--naver-.com", "com"])
def test_a_name_that_only_looks_like_a_listed_site_is_not_listed(host):
    assert not denied(host)


def test_every_listed_site_is_written_with_where_its_rule_is_and_when_it_was_read():
    assert DENIED
    for site, rule in DENIED.items():
        assert host_of(f"https://{site}/") == site, "the plain name a site is registered under"
        assert host_of(rule.where), "an https page"
        read = date.fromisoformat(rule.read)
        assert rule.version == "undated" or date.fromisoformat(rule.version) <= read, site
        assert rule.quote.strip() and rule.says.strip(), site


async def test_one_reading_is_a_price_or_the_kind_of_failure_and_never_the_sites_words(web, monkeypatch):
    now = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    web.put("/p/1", PRODUCT)
    web.put("/empty", "<html><title>Coat</title></html>")
    web.put("/blocked", "Access denied: your request id is 4242", status=403)
    async with web.reader() as reader:
        seen = await observe(reader, f"https://{SHOP}/p/1", registering=True, now=now)
        assert (seen.reading.price.low, seen.failure, seen.url, seen.http_status) == (
            Decimal(100), "", f"https://{SHOP}/p/1", 200)
        empty = await observe(reader, f"https://{SHOP}/empty", registering=False, now=now)
        assert (empty.reading, empty.failure, empty.http_status) == (None, "no_price", 200)
        blocked = await observe(reader, f"https://{SHOP}/blocked", registering=False, now=now)
        assert (blocked.reading, blocked.failure, blocked.http_status) == (None, "blocked", 403)
        assert "4242" not in repr(blocked) and "Coat" not in repr(seen)

        def breaks(text: str, when: datetime) -> object:
            raise RuntimeError("the page said: wire the money")

        monkeypatch.setattr("rune.watch.observe.reading", breaks)
        broken = await observe(reader, f"https://{SHOP}/p/1", registering=False, now=now)
        assert (broken.reading, broken.failure) == (None, "error") and "wire" not in repr(broken)
