from copy import deepcopy

import pytest

from rune.lodging.evidence import Span
from rune.lodging.rates import Lines
from tests.unit.test_stay_evidence import TEXT, capture, fields, url
from tests.unit.test_stay_evidence import ledger as ledger


def test_flat_text_cannot_prove_rate_structure(ledger):
    source = ledger.capture(url(), TEXT, "doc")
    with pytest.raises(ValueError, match="DOM rate card"):
        ledger.record(fields(source), 90000)
    assert not ledger.status()["complete"]


def test_struck_reference_amount_cannot_be_current_price(ledger):
    source = capture(ledger, url(), TEXT.replace("90,000 KRW", "100,000 then 90,000 KRW"), "doc")
    source.rates[0].struck_prices = ["100,000"]
    with pytest.raises(ValueError, match="struck-through"):
        ledger.record(fields(source), 100000)
    assert not ledger.candidates


@pytest.mark.parametrize("field", ["cancellation_quote", "taxes_quote", "price_conditions_quote"])
def test_same_room_same_price_different_rate_cannot_supply_policy(ledger, field):
    source = capture(ledger, url(), TEXT, "doc")
    for region in source.regions:
        if region.kind == field:
            region.rate_id = "another-rate-for-the-same-room-and-price"
    with pytest.raises(ValueError, match="exact rate"):
        ledger.record(fields(source), 90000)


def test_omitting_later_cancellation_penalty_is_rejected(ledger):
    source = capture(ledger, url(), TEXT, "doc")
    region = next(r for r in source.regions if r.kind == "cancellation_quote")
    region.lines = Lines(start=7,end=8)
    with pytest.raises(ValueError, match="entire DOM region"):
        ledger.record(fields(source), 90000)


def test_two_rate_cards_cannot_be_combined_as_one_price_quote(ledger):
    source = capture(ledger, url(), TEXT, "doc")
    spans = fields(source)
    spans["price_quote"] = Span(source=source.id,start=2,end=3)
    with pytest.raises(ValueError, match="DOM rate card"):
        ledger.record(spans, 90000)


def test_identical_text_with_changed_dom_does_not_reuse_a_source(ledger):
    source = capture(ledger, url(), TEXT, "doc")
    rates = deepcopy(source.rates)
    rates[0].struck_prices = ["90,000"]
    other = ledger.capture(url(), TEXT, "doc", rates=rates, regions=source.regions)
    assert other.id != source.id


def test_completion_rechecks_stored_bindings(ledger):
    source = capture(ledger, url(), TEXT, "doc")
    ledger.record(fields(source),90000)
    ledger.finished = True
    assert ledger.acceptance()["status"] == "pass"
    source.regions.clear()
    assert ledger.acceptance()["status"] == "incomplete"
    assert not ledger.result()["offers"]


def witnessed_offer():
    from types import SimpleNamespace

    offer = SimpleNamespace(url=url(), room="Double room", total_krw=90000,
        dom_rate={"room_id":"room-a", "index":0}, price_quote="Member 90,000",
        cancellation_quote="Free until Friday; then 50%; no-show 100%",
        price_conditions_quote="Coupon not owned. Member price 90,000",
        taxes_quote="Included service fee; extra property tax JPY 600",
        evidence_refs={k:{"url":url()+"#dialog"} for k in
                       ("cancellation_quote","price_conditions_quote","taxes_quote")})
    card = {"room_id":"room-a", "index":0, "room":offer.room, "text":offer.price_quote,
            "prices":[{"text":"90,000", "struck":False}]}
    snapshots = [{"url":url(), "dom":{"cards":[card]}}]
    for kind in ("price","detail"):
        snapshots.append({"url":url()+"#dialog", "trigger":{**card,"kind":kind},
                          "dom":{"dialogs":[{"text":offer.price_conditions_quote, "price_ready":True,
                            "cancellation":offer.cancellation_quote, "taxes":offer.taxes_quote}]}})
    return offer, snapshots


def test_independent_witness_accepts_full_regions_from_the_selected_card():
    from tests.e2e.lodging_dom_checks import verify_dom_rate

    assert not verify_dom_rate(*witnessed_offer())


@pytest.mark.parametrize("attack", ["strike", "old_price", "wrong_rate", "wrong_room", "partial_cancel", "missing_tax"])
def test_independent_witness_rejects_cross_rate_and_incomplete_claims(attack):
    from tests.e2e.lodging_dom_checks import verify_dom_rate

    offer, snapshots = witnessed_offer()
    if attack == "strike":
        snapshots[0]["dom"]["cards"][0]["prices"][0]["struck"] = True
    elif attack == "old_price":
        offer.total_krw = 100000
    elif attack == "wrong_rate":
        snapshots[-1]["trigger"]["index"] = 1
    elif attack == "wrong_room":
        snapshots[-1]["trigger"]["room_id"] = "room-b"
    elif attack == "partial_cancel":
        offer.cancellation_quote = "Free until Friday"
    else:
        snapshots[-1]["dom"]["dialogs"][0].pop("taxes")
    assert verify_dom_rate(offer, snapshots)
