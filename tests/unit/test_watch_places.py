"""A price a page only shows: the places it is written in, which of them are kept, and how they are read again."""

from decimal import Decimal

import pytest

from rune.watch import places
from rune.watch.item import Unread

D = Decimal
# A product page as the shops that write no structured price make it: the price is in the shop's own
# script state under several keys, beside other products' prices and amounts that are no price.
SHOP = """<html><head>
<meta property="eg:salePrice" content="45500"><meta property="eg:originalPrice" content="65000">
<meta property="og:title" content="Coat 2026"><meta name="viewport" content="width=device-width">
<script src="https://cdn.example/app.js">var SellPrice = 1;</script>
<script>var goods = {"SellPrice":45500.0000,"DcPrice":"45500", 'couponAmount': 3185, "name":"Coat"};
window.state = {SellPrice: 45500, related: [{"price": 12000}, {"price": 9900}], stock:7};</script>
<script type="application/json">{"discountedPrice": 45500.0, "ratio": 0.3, "code": "45500x"}</script>
</head><body><span class="price" data-price="45500">45,500</span><del>65,000</del></body></html>"""


def test_the_places_a_page_holds_numbers_in_are_its_meta_tags_and_its_inline_scripts_keys():
    held = places.written(SHOP)
    assert held["meta:eg:saleprice"] == {D(45500)} and held["meta:eg:originalprice"] == {D(65000)}
    assert held["key:SellPrice"] == {D(45500)}, "written three ways, one amount"
    assert held["key:DcPrice"] == {D(45500)} and held["key:discountedPrice"] == {D(45500)}
    assert held["key:price"] == {D(12000), D(9900)} and held["key:couponAmount"] == {D(3185)}
    assert held["key:ratio"] == {D("0.3")} and held["key:stock"] == {D(7)}
    # Words, a number with letters after it, what the page shows a person, an attribute, and a script
    # that is only pointed at are no place.
    assert not {"meta:og:title", "meta:viewport", "key:name", "key:code", "key:data-price"} & set(held)
    assert D(1) not in held["key:SellPrice"]


def test_the_places_kept_are_those_that_hold_the_owners_amount_and_no_other():
    assert places.holding(SHOP, D(45500)) == ("key:DcPrice", "key:SellPrice", "key:discountedPrice", "meta:eg:saleprice")
    assert places.holding(SHOP, D("45500.00")) == places.holding(SHOP, D(45500)), "the amount, however it is written"
    assert places.holding(SHOP, D(12000)) == (), "a key written with two amounts holds neither alone"
    assert places.holding(SHOP, D(45501)) == ()
    many = "<script>" + "".join(f"var k{n:02} = 500;" for n in range(30)) + "</script>"
    assert places.holding(many, D(500)) == tuple(f"key:k{n:02}" for n in range(places.MAX_PLACES))


def test_a_place_that_held_the_price_before_is_kept_over_one_that_holds_it_only_now():
    discounted = SHOP.replace('"DcPrice":"45500"', '"DcPrice":"40000"').replace('"discountedPrice": 45500.0', '"discountedPrice": 40000')
    before = ("key:DcPrice", "key:SellPrice", "key:discountedPrice")
    assert places.holding(discounted, D(40000), among=before) == ("key:DcPrice", "key:discountedPrice")
    assert places.holding(discounted, D(65000), among=before) == ("meta:eg:originalprice",), "none of them: all that hold it"


def test_the_kept_places_are_read_again_and_must_agree():
    kept = ("key:DcPrice", "key:SellPrice", "meta:eg:saleprice")
    assert places.read(SHOP, kept) == D(45500)
    lower = SHOP.replace("45500", "39900")
    assert places.read(lower, kept) == D(39900), "the price moved in every place"
    assert places.read(SHOP, ("key:DcPrice", "key:gone")) == D(45500), "a place that is gone says nothing"

    def fails(page: str, of: tuple[str, ...] = kept) -> str:
        with pytest.raises(Unread) as refusal:
            places.read(page, of)
        return refusal.value.kind

    assert fails(SHOP, ("key:gone", "meta:gone")) == "no_price" and fails("<html></html>") == "no_price"
    assert fails(SHOP.replace('"DcPrice":"45500"', '"DcPrice":"40000"')) == "split", "a discount in one place only"
    assert fails(SHOP, ("key:price",)) == "split", "one place, two amounts"
    for odd in ("0", "0.0000", "-45500", "45500.12345", "1" + "0" * 12):
        assert fails(SHOP.replace('content="45500"', f'content="{odd}"'), ("meta:eg:saleprice",)) == "split", odd


def test_a_number_is_read_only_as_a_program_writes_one():
    page = """<script>var a = {"plain": 19900, "quoted": "19900.00", 'single': '19900', dotted: 19900.0,
      "grouped": "19,900", "decimal_comma": "39,99", "spaced": "19 900", "with_unit": "19900원", "sign": "+19900",
      "exponent": 1.99e4, "bare_grouped": 19,900, "mixed": "19900', "padded": " 19900 "};
    var b = "{\\"escaped\\":19900,\\"escaped_text\\":\\"19900.0\\"}"; c.price=19900; d = 19900;</script>"""
    held = places.written(page)
    read = {place.removeprefix("key:") for place, numbers in held.items() if numbers == {D(19900)}}
    assert read == {"plain", "quoted", "single", "dotted", "escaped", "escaped_text", "d"}
    # What is written for a person is passed over whole: "19,900" is neither 19 nor 19900 here.
    assert not {"key:grouped", "key:decimal_comma", "key:spaced", "key:with_unit", "key:sign", "key:exponent",
                "key:bare_grouped", "key:mixed", "key:padded", "key:price"} & set(held)
    both = '<script>var goods = {"DcPrice": 45500}; var shown = {"DcPrice": "45,500"};</script>'
    assert places.holding(both, D(45500)) == ("key:DcPrice",), "a key also written for a person still holds one amount"
    assert places.written("<html><body>39,99 <b>1.106,18</b> 45500</body></html>") == {}, "nor is what the page shows"
    euro = '<script>var offer = {"price": 1106.18, "label": "1.106,18 €"};</script>'
    assert places.holding(euro, D("1106.18")) == ("key:price",) and places.holding(euro, D("1.11")) == ()


@pytest.mark.parametrize(("page", "said", "kept"), [
    # Olive Young, Auction and Gmarket as measured on 2026-11-23: where each carries its price.
    ('<meta property="eg:salePrice" content="25900"><meta property="eg:originalPrice" content="38000">',
     "25900", ("meta:eg:saleprice",)),
    ('<script>var item = {"SellingPrice":52800.0000,"OriginSellingPrice":160000.0000,"DCPRICE":52800,'
     '"discountedPrice":52800.0000,"optimalPrice":40650.0000,"Price":0.0000,"Price":3000}</script>',
     "52800", ("key:DCPRICE", "key:SellingPrice", "key:discountedPrice")),
    ('<script>goods = {SellPrice: 28800, DcPrice: "28,800", couponAppliedPrice: 27800, optimalPrice: 27800.0};</script>',
     "27800", ("key:couponAppliedPrice", "key:optimalPrice")),
])
def test_the_shops_measured_carry_their_price_where_it_can_be_kept(page, said, kept):
    assert places.holding(page, D(said)) == kept and places.read(page, kept) == D(said)
