"""The price a product page writes about itself, read from its structured data by code.

Only numbers and a currency code are taken from the page. Its name, its description and every
other word stay behind: what is read here ends up in a message to the owner and in the thread a
model reads later, and a page's own text must reach neither.

A value that isn't written the way the standard says is not repaired. "39,99" may be 39.99 or
3999; reading it either way is a guess, and a guessed price told to the owner is a made-up one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

from rune.utils.logger import get_logger
from rune.watch.item import Price, Unread

log = get_logger(__name__)

MAX_BLOCK = 512 * 1024
MAX_DEPTH = 40
MAX_AMOUNT = Decimal(10) ** 12
_AMOUNT = re.compile(r"[0-9]+(?:\.[0-9]{1,4})?")
_PRODUCT = frozenset({"product", "productgroup"})
_LIST_PRICE = frozenset({"listprice", "strikethroughprice", "msrp", "srp", "regularprice"})
_OG = frozenset({"product:price:amount", "product:price:currency", "product:sale_price:amount",
                 "product:sale_price:currency", "product:sale_price_dates:start", "product:sale_price_dates:end"})
# ISO 4217 codes in use. A currency is shown to the owner, so it is one of these or the read fails.
CURRENCIES = frozenset({
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", "BAM", "BBD",
    "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", "BTN", "BWP", "BYN",
    "BZD", "CAD", "CDF", "CHF", "CLP", "CNY", "COP", "CRC", "CUP", "CVE", "CZK", "DJF",
    "DKK", "DOP", "DZD", "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS",
    "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR",
    "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF", "KPW", "KRW",
    "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA",
    "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MYR", "MZN", "NAD",
    "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN",
    "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD",
    "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB", "TJS", "TMT",
    "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU", "UZS", "VED",
    "VES", "VND", "VUV", "WST", "XAF", "XCD", "XCG", "XOF", "XPF", "YER", "ZAR", "ZMW",
    "ZWG",
})


@dataclass(frozen=True, slots=True)
class Reading:
    """What one page gave: the watched price, and what is shown beside it once, at registration."""

    price: Price
    offers: int
    source: str  # "jsonld" | "og_sale" | "og"
    # Prices the page marks as list prices, or wrote as its price above a sale price it also wrote.
    listed: tuple[Decimal, ...] = ()


@dataclass(frozen=True, slots=True)
class _Found:
    low: Decimal
    high: Decimal
    currency: str
    offers: int
    listed: tuple[Decimal, ...]


class _Page(HTMLParser):
    """The page's JSON-LD blocks and its Open Graph price tags; nothing else is kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.tags: dict[str, str] = {}
        self._block: list[str] | None = None
        self._size = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        given = {name.lower(): value or "" for name, value in attrs}
        if tag == "script" and given.get("type", "").strip().lower().startswith("application/ld+json"):
            self._block, self._size = [], 0
        elif tag == "meta":
            key = (given.get("property") or given.get("name") or "").strip().lower()
            if key in _OG and key not in self.tags:
                self.tags[key] = given.get("content", "").strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._block is not None:
            if self._size <= MAX_BLOCK:
                self.blocks.append("".join(self._block))
            else:
                log.debug("watch_block_skipped", why="too_large", size=self._size)
            self._block = None

    def handle_data(self, data: str) -> None:
        if self._block is not None:
            self._size += len(data)
            if self._size <= MAX_BLOCK:
                self._block.append(data)


def reading(html: str, now: datetime) -> Reading:
    """The price the page writes. Raises ``Unread`` ("no_price", "many_products") when it gives none
    that can be read without guessing."""
    page = _Page()
    page.feed(html)
    page.close()
    found = _from_json_ld(page.blocks)
    sale = _sale(page.tags, now)
    tagged = _tagged(page.tags, "product:price:amount", page.tags.get("product:price:currency", ""))
    if found is not None:
        if sale is not None and sale[1] == found.currency and sale[0] < found.low:
            # The page calls the lower figure its sale price: that is what a visitor pays, and the
            # figure in the structured data is the price before it.
            listed = tuple(sorted({found.low, found.high, *found.listed}))
            return Reading(Price(sale[0], sale[0], found.currency), found.offers, "og_sale", listed)
        return Reading(Price(found.low, found.high, found.currency), found.offers, "jsonld",
                       tuple(sorted(set(found.listed))))
    if tagged is None:
        raise Unread("no_price")
    if sale is not None and sale[1] == tagged[1] and sale[0] < tagged[0]:
        return Reading(Price(sale[0], sale[0], tagged[1]), 1, "og_sale", (tagged[0],))
    return Reading(Price(tagged[0], tagged[0], tagged[1]), 1, "og")


def _from_json_ld(blocks: list[str]) -> _Found | None:
    priced = [found for block in blocks for node in _products(_load(block)) if (found := _prices(node)) is not None]
    if not priced:
        return None
    if len({(found.low, found.high, found.currency) for found in priced}) > 1:
        raise Unread("many_products")  # which of them the owner means can't be told from here
    return max(priced, key=lambda found: found.offers)


def _load(block: str) -> Any:
    """One block as data, or nothing when it is broken, too deep, or holds a number that isn't one."""
    def not_a_number(name: str) -> Any:
        raise ValueError(name)

    try:
        data = json.loads(block, parse_float=Decimal, parse_constant=not_a_number)
    except (ValueError, RecursionError) as exc:
        # A theme's broken breadcrumb block must not hide the product block beside it.
        log.debug("watch_block_skipped", why=type(exc).__name__)
        return None
    if _deeper_than(data, MAX_DEPTH):
        log.debug("watch_block_skipped", why="too_deep")
        return None
    return data


def _deeper_than(data: Any, limit: int) -> bool:
    level: list[Any] = [data]
    for _ in range(limit):
        level = [child for node in level if isinstance(node, dict | list)
                 for child in (node.values() if isinstance(node, dict) else node)]
        if not level:
            return False
    return True


def _products(data: Any) -> list[dict[str, Any]]:
    """The product the page is about: product nodes at the top of a block, not ones listed inside
    something else (related products, a category's items)."""
    members: list[dict[str, Any]] = []
    for root in data if isinstance(data, list) else [data]:
        if isinstance(root, dict):
            members.append(root)
            graph = root.get("@graph")
            if isinstance(graph, list):
                members.extend(node for node in graph if isinstance(node, dict))
    for member in list(members):
        main = member.get("mainEntity")
        members.extend(node for node in (main if isinstance(main, list) else [main]) if isinstance(node, dict))
    return [member for member in members if _types(member) & _PRODUCT]


def _types(node: dict[str, Any]) -> set[str]:
    """The node's type names, lower-cased and without their vocabulary prefix (shops write "product")."""
    given = node.get("@type")
    return {re.split(r"[/#:]", name)[-1].lower() for name in (given if isinstance(given, list) else [given])
            if isinstance(name, str)}


def _prices(node: dict[str, Any]) -> _Found | None:
    """What a product node's offers are priced at; nothing when none of them carries a price."""
    offers = _offers(node)
    if "productgroup" in _types(node):
        variants = node.get("hasVariant")
        for variant in variants if isinstance(variants, list) else []:
            offers.extend(_offers(variant) if isinstance(variant, dict) else [])
    amounts: list[Decimal] = []
    currencies: set[str] = set()
    listed: list[Decimal] = []
    count = 0
    for offer in offers:
        priced, currency, before = _offer(offer)
        if priced:
            amounts.extend(priced)
            currencies.add(currency)
            listed.extend(before)
            count += 1
    if not amounts:
        return None
    if len(currencies) != 1:
        raise Unread("no_price")  # prices in two currencies can't be set side by side
    return _Found(min(amounts), max(amounts), currencies.pop(), count, tuple(listed))


def _offers(node: dict[str, Any]) -> list[dict[str, Any]]:
    given = node.get("offers")
    return [offer for offer in (given if isinstance(given, list) else [given]) if isinstance(offer, dict)]


def _offer(offer: dict[str, Any]) -> tuple[list[Decimal], str, list[Decimal]]:
    """One offer's current price or prices, its currency, and the list prices written beside it.

    An offer with no price field is passed over; one whose price field can't be read fails the read.
    """
    given = offer.get("priceSpecification")
    specs = [spec for spec in (given if isinstance(given, list) else [given]) if isinstance(spec, dict)]
    before = [amount for spec in specs if _price_type(spec) in _LIST_PRICE
              if (amount := _readable(spec.get("price"))) is not None]
    current = [spec for spec in specs if spec.get("price") is not None and _price_type(spec) in ("", "saleprice")
               # A member's price and a price per unit are not what a visitor is asked to pay.
               and "validForMemberTier" not in spec and "referenceQuantity" not in spec]
    if "aggregateoffer" in _types(offer) and offer.get("lowPrice") is not None:
        amounts = [_amount(offer["lowPrice"])]
        if offer.get("highPrice") is not None:
            amounts.append(_amount(offer["highPrice"]))
    elif offer.get("price") is not None:
        amounts = [_amount(offer["price"])]
    else:
        amounts = sorted({_amount(spec["price"]) for spec in current})
        if len(amounts) > 1:
            raise Unread("no_price")  # several prices and nothing says which one is asked
    if not amounts:
        return [], "", before
    currency = offer.get("priceCurrency") or next((spec["priceCurrency"] for spec in specs
                                                   if spec.get("priceCurrency")), None)
    if not isinstance(currency, str) or currency not in CURRENCIES:
        raise Unread("no_price")
    return amounts, currency, before


def _price_type(spec: dict[str, Any]) -> str:
    given = spec.get("priceType")
    return re.split(r"[/#:]", given)[-1].lower() if isinstance(given, str) else ""


def _amount(value: Any) -> Decimal:
    amount = _readable(value)
    if amount is None:
        raise Unread("no_price")
    return amount


def _readable(value: Any) -> Decimal | None:
    """The value as an amount of money when it is written as one: a number, or digits with at most
    four decimals after a point. Nothing else is read, repaired or rounded."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | Decimal):
        amount = Decimal(value)
    elif isinstance(value, str) and _AMOUNT.fullmatch(value.strip()):
        amount = Decimal(value.strip())
    else:
        return None
    exponent = amount.as_tuple().exponent
    if not amount.is_finite() or not isinstance(exponent, int) or exponent < -4 or not 0 < amount < MAX_AMOUNT:
        return None
    return amount


def _tagged(tags: dict[str, str], key: str, currency: str) -> tuple[Decimal, str] | None:
    amount = _readable(tags.get(key))
    return (amount, currency) if amount is not None and currency in CURRENCIES else None


def _sale(tags: dict[str, str], now: datetime) -> tuple[Decimal, str] | None:
    """The page's sale-price tag, when it is in force. A sale whose dates can't be read is not used:
    telling a price lower than the real one is worse than missing a sale."""
    currency = tags.get("product:sale_price:currency") or tags.get("product:price:currency", "")
    sale = _tagged(tags, "product:sale_price:amount", currency)
    if sale is None:
        return None
    for key, starts in (("product:sale_price_dates:start", True), ("product:sale_price_dates:end", False)):
        if key not in tags:
            continue
        try:
            when = datetime.fromisoformat(tags[key])
        except ValueError:
            return None
        if when.tzinfo is None or (now < when if starts else now > when):
            return None
    return sale
