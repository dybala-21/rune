"""Public NOL entry points; actual rates still require observed detail-page evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from rune.lodging.evidence import StayQuery


def entrypoints(query: StayQuery) -> dict[str, str]:
    # The search route was exercised in the live short-request benchmark.
    params = {"q": query.city, "verticalCategory": "PRODUCT_CATEGORY_GLOBAL_ACCOMMODATION",
              "checkInDate": str(query.check_in), "checkOutDate": str(query.check_out),
              "adultCount": query.adults, "roomCount": query.rooms}
    return {"home": "https://nol.yanolja.com/",
            "overseas_search": "https://nol.yanolja.com/discovery/list/search?" + urlencode(params)}


def source_guidance(url: str) -> dict[str, str | bool]:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    rate_source = not path.startswith("/stay/overseas/") or path.endswith("/list")
    guidance: dict[str, str | bool] = {"rate_source": rate_source,
            "record_guidance": (
                "This view may supply room/rate quotes; it does not prove all conditions are present. "
                "Record room as the room name alone; cite applied occupancy from the date/guest header, not maximum capacity. "
                "Choose a current DOM rate and use inspect_rate for its complete conditions. Struck-through amounts are not current prices. Member/coupon eligibility is unverified and must remain explicit."
                if rate_source else
                "Property overview: check the address is in the requested city before opening rooms; "
                "search results may include nearby cities. Skip properties outside the requested city. "
                "Save name, address and tax policies here. Its teaser cannot supply "
                "price/cancellation/discount evidence. Navigate to room_list_url for this property's rates; "
                "the general property-information modal does not open its room list."
            )}
    # This public sibling route was verified on live NOL pages. Preserve the observed stay scope.
    if parts.hostname == "nol.yanolja.com" and path.startswith("/stay/overseas/") and not rate_source:
        guidance["room_list_url"] = urlunsplit(parts._replace(path=path + "/list", fragment=""))
    return guidance
