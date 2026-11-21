"""The price a page writes, read from its structured data: real shops' shapes, and what is refused."""

import json
from dataclasses import fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from rune.watch.extract import Reading, reading
from rune.watch.item import Price, Unread

# Real product pages read on 2026-11-20, cut down to their structured data, with every word the
# page wrote (names, identifiers, option labels) replaced by a marker.
PAGES = Path(__file__).parent.parent / "fixtures" / "watch"
NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)


def page(name: str) -> str:
    return (PAGES / f"{name}.html").read_text(encoding="utf-8")


def html(*blocks: object, tags: dict[str, str] | None = None) -> str:
    """A page holding these JSON-LD blocks (a string is put in as written) and these meta tags."""
    scripts = "".join('<script type="application/ld+json">' + (b if isinstance(b, str) else json.dumps(b)) + "</script>"
                      for b in blocks)
    metas = "".join(f'<meta property="{key}" content="{value}">' for key, value in (tags or {}).items())
    return f"<html><head><title>Coat</title>{metas}{scripts}</head><body>Coat 99,000</body></html>"


def product(offers: object, **more: object) -> dict[str, object]:
    return {"@context": "https://schema.org", "@type": "Product", "name": "Coat", "offers": offers, **more}


def offer(price: object = 100, currency: object = "KRW", **more: object) -> dict[str, object]:
    return {"@type": "Offer", "price": price, "priceCurrency": currency, **more}


def unread(text: str) -> str:
    with pytest.raises(Unread) as refused:
        reading(text, NOW)
    return refused.value.kind


@pytest.mark.parametrize(("name", "low", "high", "currency", "offers", "source", "listed"), [
    ("apple-one-offer", 369000, 369000, "KRW", 1, "jsonld", ()),
    ("apple-five-offers", 849000, 849000, "KRW", 5, "jsonld", ()),      # one per colour, one price
    ("danawa", 547460, 547460, "KRW", 1, "jsonld", ()),                 # AggregateOffer with lowPrice only
    ("yes24", 18900, 18900, "KRW", 1, "jsonld", ()),                    # "@type": ["Book", "Product"]
    ("ikea", 4, 4, "EUR", 1, "jsonld", ()),                             # the price is the string "4"
    ("nike", 149000, 149000, "KRW", 17, "jsonld", ()),                  # ProductGroup; 9 variants have no offer
    ("costco", 439990, 439990, "KRW", 1, "jsonld", ()),                 # "@type": "product", price "439990.0"
    ("29cm", 110080, 110080, "KRW", 1, "jsonld", (128000,)),            # the list price is tagged beside it
    ("cafe24-sale", 62700, 62700, "KRW", 3, "og_sale", (66000,)),       # JSON-LD says 66000; the sale tag 62700
])
def test_real_product_pages_give_their_price(name, low, high, currency, offers, source, listed):
    got = reading(page(name), NOW)
    assert got == Reading(Price(Decimal(low), Decimal(high), currency), offers, source,
                          tuple(Decimal(amount) for amount in listed))


@pytest.mark.parametrize("name", ["kurly", "uniqlo", "amazon-interstitial"])
def test_a_page_with_no_structured_price_is_not_read(name):
    # A shell filled in by scripts, state kept in the app's own format, a "continue shopping" page: all answer 200.
    assert unread(page(name)) == "no_price"


def test_nothing_the_page_wrote_in_words_is_kept():
    for name in ("apple-one-offer", "nike", "cafe24-sale", "29cm", "costco"):
        got = reading(page(name), NOW)
        assert "PAGE-" not in repr(got)
    # Whatever is read can only be a number, a currency code or one of this module's own words.
    assert {field.name for field in fields(Reading)} == {"price", "offers", "source", "listed"}
    assert {field.name for field in fields(Price)} == {"low", "high", "currency"}


def test_only_the_pages_own_product_is_read_not_products_listed_inside_something_else():
    related = product(offer(100), isRelatedTo=[product(offer(5))], review={"itemReviewed": product(offer(7))})
    listing = {"@type": "ItemList", "itemListElement": [{"@type": "ListItem", "item": product(offer(9))}]}
    assert reading(html(related, listing), NOW).price == Price(Decimal(100), Decimal(100), "KRW")
    assert unread(html(listing)) == "no_price"  # a category page is not a product page


def test_two_products_at_the_top_with_different_prices_are_refused_and_the_same_one_twice_is_one():
    assert unread(html(product(offer(100)), product(offer(200)))) == "many_products"
    same = reading(html(product(offer(100)), product([offer("100.00"), offer(100)])), NOW)
    assert (same.price.low, same.offers) == (Decimal(100), 2)


@pytest.mark.parametrize("wrapped", [
    {"@context": "https://schema.org", "@graph": [{"@type": "WebSite"}, product(offer(100))]},
    [{"@type": "BreadcrumbList"}, product(offer(100))],
    {"@type": "ItemPage", "mainEntity": product(offer(100))},
    {"@graph": [{"@type": "WebPage", "mainEntity": product(offer(100))}]},
    product(offer(100), **{"@type": "https://schema.org/Product"}),
    product(offer(100), **{"@type": "schema:Product"}),
    product([offer(100)]),
    product({"price": 100, "priceCurrency": "KRW"}),  # an offer that doesn't say it is one
])
def test_where_a_product_node_may_sit_and_how_its_type_may_be_written(wrapped):
    assert reading(html(wrapped), NOW).price == Price(Decimal(100), Decimal(100), "KRW")


def test_several_offers_are_watched_by_their_lowest_and_highest():
    got = reading(html(product([offer(300), offer(100), offer(200)])), NOW)
    assert (got.price, got.offers) == (Price(Decimal(100), Decimal(300), "KRW"), 3)
    ranged = product({"@type": "AggregateOffer", "lowPrice": "90", "highPrice": 150, "priceCurrency": "KRW"})
    assert reading(html(ranged), NOW).price == Price(Decimal(90), Decimal(150), "KRW")
    group = {"@type": "ProductGroup", "hasVariant": [product(offer(120)), {"@type": "Product"}, product(offer(80))]}
    assert reading(html(group), NOW).price == Price(Decimal(80), Decimal(120), "KRW")


def test_a_price_given_only_as_a_specification_is_read_when_exactly_one_is_the_asking_price():
    def spec(price: object, **more: object) -> dict[str, object]:
        return {"@type": "UnitPriceSpecification", "price": price, "priceCurrency": "KRW", **more}

    # As WooCommerce writes a product on sale: the sale price, then the regular price marked as the list price.
    woo = product({"@type": "Offer", "priceSpecification": [
        spec("79.00"), spec("99.00", priceType="https://schema.org/ListPrice")]})
    got = reading(html(woo), NOW)
    assert (got.price.low, got.listed) == (Decimal(79), (Decimal(99),))
    members = product({"@type": "Offer", "priceSpecification": [
        spec(100), spec(80, validForMemberTier={"@type": "MemberProgramTier"}),
        spec(3, referenceQuantity={"@type": "QuantitativeValue", "value": 100})]})
    assert reading(html(members), NOW).price.low == Decimal(100)  # not the members' price, not the price per 100 g
    assert unread(html(product({"@type": "Offer", "priceSpecification": [spec(100), spec(90)]}))) == "no_price"
    # The currency may be written on the specification instead of the offer.
    beside = product({"@type": "Offer", "price": 100, "priceSpecification": spec(120, priceType="ListPrice")})
    assert reading(html(beside), NOW) == Reading(Price(Decimal(100), Decimal(100), "KRW"), 1, "jsonld", (Decimal(120),))


@pytest.mark.parametrize("price", [
    "LookBook",        # a real shop's placeholder
    "39,99",           # 39.99 or 3999: reading it either way is a guess
    "1,100", "₩1100", "1100원", "1100 KRW", "", " ", "-5", "1e3", "１２３",
    0, "0", -1, True, 12.34567, "12.34567", 10 ** 12, [100], {"value": 100},
])
def test_a_price_not_written_as_a_plain_amount_fails_the_read_instead_of_being_repaired(price):
    assert unread(html(product(offer(price)))) == "no_price"
    # One unreadable price among good ones fails it too: leaving it out would move the lowest and highest.
    assert unread(html(product([offer(100), offer(price)]))) == "no_price"


@pytest.mark.parametrize(("price", "amount"), [
    (100, "100"), ("100", "100"), (" 100 ", "100"), ("56.00", "56"), (56.5, "56.5"), ("0.0001", "0.0001"), (1e3, "1000"),
])
def test_an_amount_is_the_same_however_it_is_written(price, amount):
    assert reading(html(product(offer(price))), NOW).price.low == Decimal(amount)


@pytest.mark.parametrize("currency", ["krw", "₩", "WON", "KR", "ABC", "", None, 410, "KRW ", ["KRW"]])
def test_a_currency_that_is_not_a_real_code_fails_the_read(currency):
    assert unread(html(product(offer(100, currency)))) == "no_price"


def test_prices_in_two_currencies_are_refused():
    assert unread(html(product([offer(100, "KRW"), offer(1, "USD")]))) == "no_price"


def test_an_offer_without_any_price_is_passed_over_and_a_product_with_none_has_no_price():
    assert reading(html(product([{"@type": "Offer", "availability": "InStock"}, offer(100)])), NOW).offers == 1
    assert reading(html(product([{"@type": "Offer", "price": None}, offer(100)])), NOW).offers == 1
    assert unread(html(product({"@type": "Offer", "availability": "InStock"}))) == "no_price"
    assert unread(html({"@type": "Product", "name": "Coat"})) == "no_price"


def test_a_broken_or_strange_block_does_not_hide_the_product_block_beside_it():
    good = product(offer(100))
    assert reading(html("{ not json", good), NOW).price.low == Decimal(100)
    assert reading(html('{"@type": "Thing", "weight": NaN}', good), NOW).price.low == Decimal(100)
    assert reading(html("[" * 5000 + "]" * 5000, good), NOW).price.low == Decimal(100)
    deep: object = product(offer(1))
    for _ in range(60):
        deep = {"@graph": [deep]}
    assert reading(html(deep, good), NOW).price.low == Decimal(100)
    huge = json.dumps(product(offer(1), description="x" * 600_000))
    assert reading(html(huge, good), NOW).price.low == Decimal(100)
    # And when the product block itself is the broken one, there is no price.
    assert unread(html('{"@type": "Product", "offers": {"price": Infinity, "priceCurrency": "KRW"}}')) == "no_price"


SALE = {"product:price:amount": "66000", "product:price:currency": "KRW", "product:sale_price:amount": "62700"}


def test_a_sale_price_tag_below_the_structured_price_is_the_price_and_the_other_is_shown_as_listed():
    got = reading(html(product(offer(66000)), tags=SALE), NOW)
    assert (got.price.low, got.source, got.listed) == (Decimal(62700), "og_sale", (Decimal(66000),))
    # A sale tag that is not lower changes nothing.
    assert reading(html(product(offer(60000)), tags=SALE), NOW).source == "jsonld"
    # Without structured data the tags are read by themselves.
    assert reading(html(tags=SALE), NOW).price.low == Decimal(62700)
    plain = {"product:price:amount": "66000", "product:price:currency": "KRW"}
    assert reading(html(tags=plain), NOW) == Reading(Price(Decimal(66000), Decimal(66000), "KRW"), 1, "og")


@pytest.mark.parametrize(("dates", "used"), [
    ({}, True),
    ({"product:sale_price_dates:start": "2026-11-01T00:00:00+09:00", "product:sale_price_dates:end": "2026-11-30T23:59:59+09:00"}, True),
    ({"product:sale_price_dates:end": "2026-11-19T23:59:59+09:00"}, False),     # over
    ({"product:sale_price_dates:start": "2026-12-01T00:00:00+09:00"}, False),   # not begun
    ({"product:sale_price_dates:end": "next friday"}, False),                   # can't be read
    ({"product:sale_price_dates:end": "2026-11-30 23:59"}, False),              # no time zone: can't be placed
])
def test_a_sale_tag_is_used_only_while_its_dates_can_be_read_and_hold(dates, used):
    got = reading(html(product(offer(66000)), tags={**SALE, **dates}), NOW)
    assert got.price.low == (Decimal(62700) if used else Decimal(66000))


@pytest.mark.parametrize("tags", [
    {**SALE, "product:sale_price:amount": "0"}, {**SALE, "product:sale_price:amount": "62,700"},
    {**SALE, "product:sale_price:currency": "USD"}, {"product:sale_price:amount": "62700"},
])
def test_a_sale_tag_that_cannot_be_read_or_compared_is_left_out(tags):
    assert reading(html(product(offer(66000)), tags=tags), NOW).price.low == Decimal(66000)


def test_an_unreadable_price_tag_alone_gives_no_price():
    assert unread(html(tags={"product:price:amount": "0", "product:price:currency": "KRW"})) == "no_price"
    assert unread(html(tags={"product:price:amount": "66000"})) == "no_price"
    assert unread("") == "no_price" and unread("<html><body>99,000</body></html>") == "no_price"


def test_markup_as_pages_really_write_it():
    text = ('<SCRIPT TYPE="Application/LD+JSON; charset=utf-8">' + json.dumps(product(offer(100))) + "</SCRIPT>"
            '<meta name="product:price:amount" content="1"><script>var x = {"price": 5}</script>')
    assert reading(text, NOW).price.low == Decimal(100)
    escaped = '<script type="application/ld+json">{"@type":"Product","name":"A &amp; B","offers":{"price":100,"priceCurrency":"KRW"}}</script>'
    assert reading(escaped, NOW).price.low == Decimal(100)
