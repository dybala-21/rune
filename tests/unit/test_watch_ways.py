"""Which way a page is read: asked for under RUNE's name, or opened in RUNE's own browser."""

import socket
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from rune.watch.fetch import Page
from rune.watch.item import BROWSER, FETCH, Unread
from rune.watch.observe import observe
from tests.unit.test_watch_fetch import PRODUCT, Web

COAT = "https://shop.example/p/coat"
NOW = datetime(2026, 11, 22, tzinfo=UTC)
NOBODY_UNNAMED = "User-agent: *\nDisallow: /\n"


class Opens:
    """A pretend browser: what opening each address gives, and every address it was asked to open."""

    def __init__(self, **answers: str | Unread) -> None:
        self.answers: dict[str, str | Unread] = {COAT: PRODUCT, **answers}
        self.opened: list[str] = []

    async def page(self, url: str, *, registering: bool) -> Page:
        self.opened.append(url)
        answer = self.answers[url]
        if isinstance(answer, Unread):
            raise answer
        return Page(url, 200, answer)


@pytest.fixture
def web() -> Web:
    return Web()


async def read(web: Web, browser: Opens | None, *, via: str = FETCH, registering: bool = True, **more: object):
    async with web.reader(**more) as reader:
        return await observe(reader, COAT, registering=registering, now=NOW, browser=browser, via=via)


async def test_a_page_a_site_gives_when_asked_is_not_opened_in_a_browser(web):
    web.put("/p/coat", PRODUCT)
    browser = Opens()
    seen = await read(web, browser)
    assert (seen.reading.price.low, seen.via, browser.opened) == (Decimal(100), FETCH, [])


@pytest.mark.parametrize("refuses", ["robots", "blocked", "no_price"])
async def test_a_page_refused_or_empty_when_asked_is_opened_in_the_browser(web, refuses):
    if refuses == "robots":
        web.put("/robots.txt", NOBODY_UNNAMED)
    web.put("/p/coat", "<html>no price here</html>", status=403 if refuses == "blocked" else 200)
    assert (await read(web, None)).failure == refuses, "with no browser, what asking gave is the answer"

    browser = Opens()
    seen = await read(web, browser)
    assert (seen.reading.price.low, seen.via, seen.url, browser.opened) == (Decimal(100), BROWSER, COAT, [COAT])


@pytest.mark.parametrize(("status", "kind"), [(404, "gone"), (500, "server"), (400, "error")])
async def test_a_page_that_is_not_there_is_not_looked_for_in_a_browser(web, status, kind):
    web.put("/p/coat", status=status)
    browser = Opens()
    assert ((await read(web, browser)).failure, browser.opened) == (kind, [])


async def test_a_site_that_cannot_be_reached_or_forbids_it_in_its_terms_is_not_opened_either(web):
    browser = Opens()

    async def unknown(host: str) -> str:
        raise socket.gaierror

    assert ((await read(web, browser, resolve=unknown)).failure, browser.opened) == ("unreachable", [])
    async with web.reader() as reader:
        seen = await observe(reader, "https://www.musinsa.com/products/1", registering=True, now=NOW, browser=browser)
    assert (seen.failure, browser.opened) == ("denied", [])


async def test_a_site_that_names_rune_and_refuses_it_is_not_opened_in_the_browser(web):
    web.put("/robots.txt", "User-agent: *\nAllow: /\n\nUser-agent: RUNE\nDisallow: /p/\n")
    browser = Opens()
    seen = await read(web, browser)
    assert (seen.failure, seen.via, browser.opened) == ("robots", FETCH, [])
    assert (await read(web, browser, via=BROWSER)).failure == "robots" and browser.opened == [], "nor one read that way before"

    # A group for RUNE that leaves the page open says nothing against it: the site's refusal of
    # the request itself is then like any other.
    web.put("/robots.txt", "User-agent: RUNE\nDisallow: /cart\n")
    web.put("/p/coat", status=403)
    assert (await read(web, browser)).via == BROWSER and browser.opened == [COAT]


@pytest.mark.parametrize("status", [403, 500])
async def test_a_robots_txt_that_cannot_be_had_names_nobody(web, status):
    web.put("/robots.txt", status=status)
    browser = Opens()
    assert (await read(web, browser)).via == BROWSER and browser.opened == [COAT]


async def test_a_page_read_in_the_browser_before_is_opened_at_once_and_not_asked_for(web):
    web.put("/p/coat", status=403)   # what asking would get, were it asked
    browser = Opens()
    seen = await read(web, browser, via=BROWSER, registering=False)
    assert (seen.reading.price.low, seen.via, browser.opened) == (Decimal(100), BROWSER, [COAT])
    assert web.asked() == ["/robots.txt"], "only whether the site names RUNE was looked up"

    async def unknown(host: str) -> str:
        raise socket.gaierror

    # robots.txt on a host that can't be found: opening the page is what says so.
    browser = Opens(**{COAT: Unread("unreachable")})
    assert (await read(web, browser, via=BROWSER, resolve=unknown)).failure == "unreachable"


async def test_what_the_browser_gave_is_the_answer_when_both_ways_fail(web):
    web.put("/robots.txt", NOBODY_UNNAMED)
    seen = await read(web, Opens(**{COAT: Unread("blocked", 403)}))
    assert (seen.failure, seen.http_status, seen.via, seen.reading) == ("blocked", 403, BROWSER, None)
    seen = await read(web, Opens(**{COAT: "<html>a check that asks for a person</html>"}))
    assert (seen.failure, seen.via) == ("no_price", BROWSER)


async def test_on_a_machine_with_no_browser_what_asking_gave_stands(web):
    web.put("/robots.txt", NOBODY_UNNAMED)
    nothing = Opens(**{COAT: Unread("no_browser")})
    seen = await read(web, nothing)
    assert (seen.failure, seen.via, nothing.opened) == ("robots", FETCH, [COAT])

    # A page read in a browser before, on a machine that has none now, is asked for after all.
    web.put("/robots.txt", "")
    web.put("/p/coat", PRODUCT)
    seen = await read(web, nothing, via=BROWSER)
    assert (seen.reading.price.low, seen.via) == (Decimal(100), FETCH)
    assert (await read(web, None, via=BROWSER)).via == FETCH, "and so is one read with no browser handed over"
