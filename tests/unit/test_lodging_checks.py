import json
from copy import deepcopy

import pytest

from tests.e2e.lodging_checks import StayRequest, nol_url, numbers, verify_comparison


def test_numeric_evidence_accepts_punctuation_but_not_parts_of_decimals():
    assert numbers("객실 1, 성인 2, 아동 0") == {0, 1, 2}
    assert numbers("1 room, 2 adults.") == {1, 2}
    assert numbers("120,000 KRW, 13.5 m², 1,234.50 USD") == {120000}


@pytest.fixture
def comparison_case():
    stay = StayRequest(city="Seoul", check_in="2026-11-15", check_out="2026-11-16",
                       adults=2, max_total_krw=200000, count=1)
    offer = {"name": "Example hotel", "url": "https://nol.yanolja.com/stay/domestic/123",
             "room": "Double room", "total_krw": 120000,
             "price_quote": "Double room: KRW 120,000 for the full stay",
             "stay_quote": "2026-11-15 to 2026-11-16", "occupancy_quote": "2 adults, 1 room",
             "location_quote": "Seoul, Example street",
             "cancellation_quote": "Non-refundable",
             "price_conditions_quote": "Taxes included. No coupon required.",
             "taxes_quote": "Taxes included.", "price_basis": "displayed_rate_subject_to_quoted_conditions",
             "eligibility": "not_verified", "purchase_verified": False, "dom_rate": {"room_id": "room-a", "index": 0}}
    text = "\n".join([offer["name"], *[v for k, v in offer.items() if k.endswith("_quote")]])
    return stay, {"offers": [offer], "limitations": []}, [{"url": offer["url"], "text": text}]


def check(case):
    stay, result, pages = case
    return verify_comparison(json.dumps(result), stay, pages, check_dom=False)


def test_all_quotes_must_be_observed_on_the_actual_detail_page(comparison_case):
    assert check(comparison_case)["passed"]
    comparison_case[2].clear()
    assert not check(comparison_case)["passed"]


@pytest.mark.parametrize("field,value", [
    ("total_krw", 110000), ("room", "Suite"), ("location_quote", "Busan"),
    ("stay_quote", "2025-11-15 to 2025-11-16"), ("occupancy_quote", "20 adults"),
    ("cancellation_quote", "Free cancellation anytime"),
    ("price_conditions_quote", "No taxes or fees"), ("total_krw", True),
    ("url", "https://nol.yanolja.com.evil.example/stay/domestic/123"),
    ("url", "https://nol.yanolja.com:invalid/stay/domestic/123"),
])
def test_fabricated_or_mismatched_claims_fail(comparison_case, field, value):
    comparison_case[1]["offers"][0][field] = value
    assert not check(comparison_case)["passed"]


def test_wrong_stay_fails_even_when_the_quote_is_real(comparison_case):
    comparison_case[1]["offers"][0]["stay_quote"] = "2025-11-15 to 2025-11-16"
    comparison_case[2][0]["text"] = comparison_case[2][0]["text"].replace("2026-", "2025-")
    assert not check(comparison_case)["passed"]


def test_cannot_combine_different_page_states(comparison_case):
    page = comparison_case[2][0]
    second = deepcopy(page)
    page["text"] = page["text"].replace("Non-refundable", "")
    second["text"] = second["text"].replace("Double room: KRW 120,000 for the full stay", "")
    comparison_case[2].append(second)
    assert not check(comparison_case)["passed"]


def test_duplicate_property_with_different_query_is_not_another_hotel(comparison_case):
    stay, result, pages = comparison_case
    stay.count = 2
    duplicate = deepcopy(result["offers"][0])
    duplicate["url"] += "?other=1"
    result["offers"].append(duplicate)
    pages.append({**pages[0], "url": duplicate["url"]})
    assert not check(comparison_case)["passed"]


def test_price_limit_and_unresolved_conditions_are_failures(comparison_case):
    comparison_case[0].max_total_krw = 100000
    assert not check(comparison_case)["passed"]
    comparison_case[0].max_total_krw = 200000
    comparison_case[1]["limitations"] = ["Cannot establish taxes"]
    assert not check(comparison_case)["passed"]


def test_access_error_is_not_an_empty_success(comparison_case):
    stay, _, _ = comparison_case
    result = verify_comparison('{"offers":[],"limitations":["Service unavailable"]}', stay, [])
    assert not result["passed"]
    assert result["offers_checked"] == 0
    assert not verify_comparison("Done!", stay, [])["passed"]


def test_failed_http_page_cannot_support_an_offer(comparison_case):
    comparison_case[2][0]["http_status"] = 403
    assert not check(comparison_case)["passed"]


def test_overseas_requires_explicit_market_and_observed_detail(comparison_case):
    stay, result, pages = comparison_case
    result["offers"][0]["url"] = "https://nol.yanolja.com/stay/overseas/TDP-125731"
    pages[0]["url"] = result["offers"][0]["url"]
    assert not check(comparison_case)["passed"]
    stay.market = "overseas"
    assert check(comparison_case)["passed"]
    pages.clear()
    assert not check(comparison_case)["passed"]


def test_overseas_search_is_not_a_property_detail(comparison_case):
    stay, result, pages = comparison_case
    stay.market = "overseas"
    result["offers"][0]["url"] = "https://nol.yanolja.com/stay/overseas/search"
    pages[0]["url"] = result["offers"][0]["url"]
    assert not check(comparison_case)["passed"]


def test_overseas_room_list_is_the_same_property(comparison_case):
    stay, result, pages = comparison_case
    stay.market, stay.count = "overseas", 2
    url = "https://nol.yanolja.com/stay/overseas/TDP-125731"
    result["offers"][0]["url"] = pages[0]["url"] = url
    result["offers"].append({**result["offers"][0], "url": url + "/list"})
    pages.append({**pages[0], "url": url + "/list"})
    assert "Offer 2: duplicate property" in check(comparison_case)["failures"]
    result["offers"].pop(0)
    stay.count = 1
    assert check(comparison_case)["passed"]


def test_explicit_destination_label_allows_localized_address(comparison_case):
    comparison_case[0].city = "서울"
    assert not check(comparison_case)["passed"]
    comparison_case[0].city_labels = ["Seoul"]
    assert check(comparison_case)["passed"]


@pytest.mark.parametrize("url", ["https://[broken", "http://nol.yanolja.com/", "https://user@nol.yanolja.com/"])
def test_untrusted_urls_are_rejected(url):
    assert not nol_url(url)
