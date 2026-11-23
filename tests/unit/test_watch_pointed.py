"""A page that shows its price without writing it as structured data, read through the watch's two ways:
what is kept once its owner has said the price, and how it is read again."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from rune.watch.extract import Reading
from rune.watch.item import BROWSER, FETCH, Price
from rune.watch.observe import Pointed, observe
from tests.unit.test_watch_fetch import PRODUCT, Web
from tests.unit.test_watch_places import SHOP
from tests.unit.test_watch_ways import COAT, Opens

D = Decimal
NOW = datetime(2026, 11, 23, tzinfo=UTC)
ALL = ("key:DcPrice", "key:SellPrice", "key:discountedPrice", "meta:eg:saleprice")
EMPTY = "<html><body>the price comes once the page's scripts have run</body></html>"
DISCOUNTED = SHOP.replace('"DcPrice":"45500"', '"DcPrice":"40000"').replace('"discountedPrice": 45500.0', '"discountedPrice": 40000')


@pytest.fixture
def web() -> Web:
    return Web()


def shop(page: str = SHOP) -> Opens:
    return Opens(**{COAT: page})


async def see(web: Web, browser: Opens | None, how: Pointed | None = None, *, registering: bool = True, via: str = FETCH):
    async with web.reader() as reader:
        return await observe(reader, COAT, registering=registering, now=NOW, browser=browser, via=via, pointed_at=how)


async def test_a_page_with_no_structured_price_has_none_until_its_owner_says_one(web):
    web.put("/p/coat", SHOP)
    seen = await see(web, shop())
    assert (seen.failure, seen.via, seen.reading, seen.places) == ("no_price", BROWSER, None, ()), "nothing is picked for them"
    assert (await see(web, None)).failure == "no_price"


async def test_the_amount_the_owner_said_is_kept_with_the_places_that_hold_it(web):
    web.put("/p/coat", SHOP)
    browser = shop()
    seen = await see(web, browser, Pointed(currency="KRW", said=D(45500)))
    assert seen.reading == Reading(Price(D(45500), D(45500), "KRW"), 1, "pointed")
    assert (seen.places, seen.via, browser.opened) == (ALL, FETCH, []), "the page it was given holds it: no browser"

    # A shop that puts its price in by script: the page a plain request gets does not hold the amount.
    web.put("/p/coat", EMPTY)
    seen = await see(web, browser, Pointed(currency="KRW", said=D(45500)))
    assert (seen.reading.price.low, seen.places, seen.via, browser.opened) == (D(45500), ALL, BROWSER, [COAT])

    seen = await see(web, shop(), Pointed(currency="KRW", said=D(44000)))
    assert (seen.failure, seen.via, seen.places) == ("not_shown", BROWSER, ()), "held by neither page"
    assert (await see(web, None, Pointed(currency="KRW", said=D(44000)))).failure == "not_shown"


async def test_a_page_that_writes_its_price_as_data_is_read_from_that_whatever_was_said(web):
    web.put("/p/coat", PRODUCT)
    seen = await see(web, shop(), Pointed(currency="KRW", said=D(45500)))
    assert (seen.reading.price.low, seen.reading.source, seen.places) == (D(100), "jsonld", ())


async def test_a_page_with_several_products_is_not_one_to_point_at(web):
    block = '<script type="application/ld+json">{"@type":"Product","offers":{"price":PRICE,"priceCurrency":"KRW"}}</script>'
    two = block.replace("PRICE", "100") + block.replace("PRICE", "200")
    web.put("/p/coat", two + SHOP)
    seen = await see(web, shop(two + SHOP), Pointed(currency="KRW", said=D(45500)))
    assert (seen.failure, seen.places) == ("many_products", ())


async def test_the_kept_places_are_read_each_day_and_nothing_else_is(web):
    kept = Pointed(("key:DcPrice", "key:SellPrice"), "KRW")
    web.put("/p/coat", PRODUCT + SHOP.replace("45500", "39900"))   # structured data it never had says 100
    browser = shop()
    seen = await see(web, browser, kept, registering=False)
    assert seen.reading == Reading(Price(D(39900), D(39900), "KRW"), 1, "pointed")
    assert (seen.places, seen.via, browser.opened) == (kept.places, FETCH, [])

    web.put("/p/coat", DISCOUNTED)
    seen = await see(web, browser, kept, registering=False)
    assert (seen.failure, seen.reading, browser.opened) == ("split", None, []), "no browser tells which is the price"


async def test_places_a_plain_request_no_longer_gets_are_looked_for_in_the_browser_that_day(web):
    kept = Pointed(("key:DcPrice", "key:SellPrice"), "KRW")
    web.put("/p/coat", EMPTY)
    browser = shop()
    seen = await see(web, browser, kept, registering=False)
    assert (seen.reading.price.low, seen.via, browser.opened) == (D(45500), BROWSER, [COAT])
    gone = await see(web, shop(EMPTY), kept, registering=False)
    assert (gone.failure, gone.via) == ("no_price", BROWSER)
    opened = await see(web, shop(DISCOUNTED), kept, registering=False, via=BROWSER)
    assert (opened.failure, opened.via) == ("split", BROWSER)


async def test_pointing_again_keeps_the_places_that_held_the_price_both_times(web):
    web.put("/p/coat", DISCOUNTED)
    before = ("key:DcPrice", "key:SellPrice", "meta:eg:saleprice")
    seen = await see(web, None, Pointed(before, "KRW", D(40000)))
    assert (seen.reading.price.low, seen.places) == (D(40000), ("key:DcPrice",))
    seen = await see(web, None, Pointed(before, "KRW", D(3185)))
    assert seen.places == ("key:couponAmount",), "none of them holds it: the places that do"
    assert (await see(web, None, Pointed(before, "KRW", D(41000)))).failure == "not_shown"
