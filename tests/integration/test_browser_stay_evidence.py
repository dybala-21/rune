"""Real DOM capture and persisted recovery; no live provider or NOL traffic."""

import json
from pathlib import Path

import pytest

from rune.capabilities.browser.session import current_session
from rune.capabilities.browser.stay_evidence import BrowserStayEvidenceParams, browser_stay_evidence
from rune.lodging.evidence import Span, StayQuery
from tests.integration.stay_fixture import HTML
from tests.integration.test_browser_foundation import screen as screen
from tests.unit.test_stay_evidence import TEXT, url


async def test_capture_persist_recover_and_fill_only_missing_field(screen):
    await screen.route("https://nol.yanolja.com/**", lambda route: route.fulfill(body=HTML, content_type="text/html"))
    await screen.goto(url())
    query = StayQuery(city="Sapporo", check_in="2026-11-15", check_out="2026-11-16",
                      adults=2, count=1, max_total_krw=200000)
    assert (await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=query))).success
    captured = await browser_stay_evidence(BrowserStayEvidenceParams(action="capture"))
    assert captured.success, captured.error
    ledger = current_session().stay_evidence
    source = next(iter(ledger.sources.values()))
    assert len(source.rates) == 2 and source.rates[0].struck_prices == ["100,000"]
    selected = source.rates[0]
    inspected = await browser_stay_evidence(BrowserStayEvidenceParams(action="inspect_rate", source=source.id, rate=selected.id))
    assert inspected.success, inspected.error
    spans, total = ledger.rate_fields(source.id, selected.id)
    taxes = spans.pop("taxes_quote")
    spans.update({k: Span(source=source.id,start=n,end=n) for k,n in
                  (("stay_quote",2),("occupancy_quote",3),("location_quote",4))})
    recorded = await browser_stay_evidence(BrowserStayEvidenceParams(action="record", fields=spans, total_krw=total))
    assert recorded.success, recorded.error
    candidate = json.loads(recorded.output)["recorded"]
    assert json.loads(recorded.output)["missing"] == ["taxes_quote"]
    assert Path(recorded.metadata["evidence_path"]).exists()
    # Already inspected regions do not cause repeated network/UI work.
    repeated = await browser_stay_evidence(BrowserStayEvidenceParams(action="inspect_rate", source=source.id, rate=selected.id))
    assert repeated.success and await screen.evaluate("window.reads") == 2
    assert await screen.evaluate("window.bookings") == 0
    current_session().stay_evidence = None
    await screen.goto("about:blank")
    saved = await browser_stay_evidence(BrowserStayEvidenceParams(action="read", source=taxes.source, start=taxes.start, limit=1))
    assert saved.success and "JPY 600" in json.loads(saved.output)["lines"]
    rejected = await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=query.model_copy(update={"count": 2})))
    assert not rejected.success and "fixed" in rejected.error
    filled = await browser_stay_evidence(BrowserStayEvidenceParams(action="record", candidate=candidate, fields={"taxes_quote": taxes}))
    assert filled.success and filled.metadata["retrieval_acceptance"]["status"] == "incomplete"
    finished = await browser_stay_evidence(BrowserStayEvidenceParams(action="finish"))
    assert finished.success and finished.metadata["retrieval_acceptance"]["status"] == "pass"
    offer = json.loads(finished.output)["offers"][0]
    assert offer["taxes_quote"] == "Pay local accommodation tax at the property: JPY 600"
    assert offer["eligibility"] == "not_verified" and not offer["purchase_verified"]
    assert "After that: non-refundable" in offer["cancellation_quote"]
    assert finished.metadata["source_count"] == 3


async def test_changed_rate_cannot_reuse_an_old_card_ordinal(screen):
    await screen.route("https://nol.yanolja.com/**", lambda route: route.fulfill(body=HTML, content_type="text/html"))
    await screen.goto(url())
    await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=StayQuery(
        city="Sapporo", check_in="2026-11-15", check_out="2026-11-16", adults=2, max_total_krw=200000)))
    await browser_stay_evidence(BrowserStayEvidenceParams(action="capture"))
    source = next(iter(current_session().stay_evidence.sources.values()))
    await screen.locator('span.typography-subtitle-20-bold').first.evaluate("e=>e.textContent='80,000'")
    result = await browser_stay_evidence(BrowserStayEvidenceParams(action="inspect_rate", source=source.id, rate=source.rates[0].id))
    assert not result.success and "changed" in result.error
    assert await screen.evaluate("window.reads") == 0


async def test_capture_waits_for_room_data_after_dom_quiet(screen):
    await screen.route("https://nol.yanolja.com/**", lambda route: route.fulfill(
        body='<div class="animate-pulse" style="height:20px">Placeholder</div>', content_type="text/html"))
    await screen.goto(url())
    await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=StayQuery(
        city="Sapporo", check_in="2026-11-15", check_out="2026-11-16", adults=2, count=1, max_total_krw=200000)))
    await screen.evaluate("text => setTimeout(() => {document.body.innerText = text}, 800)", TEXT)
    captured = await browser_stay_evidence(BrowserStayEvidenceParams(action="capture"))
    assert captured.success, captured.error
    assert next(iter(current_session().stay_evidence.sources.values())).text == TEXT


async def test_loading_timeout_does_not_store_placeholder_as_evidence(screen, monkeypatch):
    from rune.capabilities.browser.stay_readiness import wait_for_stay_content

    await screen.route("https://nol.yanolja.com/**", lambda route: route.fulfill(
        body='<div class="animate-pulse" style="height:20px">Placeholder</div>', content_type="text/html"))
    await screen.goto(url())
    await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=StayQuery(
        city="Sapporo", check_in="2026-11-15", check_out="2026-11-16", adults=2, count=1, max_total_krw=200000)))

    async def bounded(page):
        await wait_for_stay_content(page, timeout_ms=100)

    monkeypatch.setattr("rune.capabilities.browser.stay_evidence.wait_for_stay_content", bounded)
    captured = await browser_stay_evidence(BrowserStayEvidenceParams(action="capture"))
    assert not captured.success and "loading" in captured.error
    assert not current_session().stay_evidence.sources


@pytest.mark.parametrize("body", ['<p>No available rooms</p>', '<div hidden class="animate-pulse">Hidden</div>'])
async def test_no_rates_or_hidden_placeholder_does_not_wait_forever(screen, body):
    from rune.capabilities.browser.stay_readiness import wait_for_stay_content

    await screen.route("https://nol.yanolja.com/**", lambda route: route.fulfill(body=body, content_type="text/html"))
    await screen.goto(url())
    await wait_for_stay_content(screen, timeout_ms=100)
