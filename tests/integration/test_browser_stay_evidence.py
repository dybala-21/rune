"""Real DOM capture and persisted recovery; no live provider or NOL traffic."""

import html
import json
from pathlib import Path

from rune.capabilities.browser.session import current_session
from rune.capabilities.browser.stay_evidence import BrowserStayEvidenceParams, browser_stay_evidence
from rune.lodging.evidence import StayQuery
from tests.integration.test_browser_foundation import screen as screen
from tests.unit.test_stay_evidence import TEXT, fields, url


async def test_capture_persist_recover_and_fill_only_missing_field(screen):
    await screen.route("https://nol.yanolja.com/**", lambda route: route.fulfill(
        body="<pre>" + html.escape(TEXT) + "</pre>", content_type="text/html"))
    await screen.goto(url())
    query = StayQuery(city="Sapporo", check_in="2026-11-15", check_out="2026-11-16",
                      adults=2, count=1, max_total_krw=200000)
    begin = await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=query))
    assert begin.success
    captured = await browser_stay_evidence(BrowserStayEvidenceParams(action="capture"))
    assert captured.success, captured.error
    ledger = current_session().stay_evidence
    source = next(iter(ledger.sources.values()))
    spans = fields(source)
    taxes = spans.pop("taxes_quote")
    recorded = await browser_stay_evidence(BrowserStayEvidenceParams(action="record", fields=spans, total_krw=90000))
    assert recorded.success, recorded.error
    candidate = json.loads(recorded.output)["recorded"]
    assert json.loads(recorded.output)["missing"] == ["taxes_quote"]
    assert Path(recorded.metadata["evidence_path"]).exists()
    current_session().stay_evidence = None
    await screen.goto("about:blank")  # Recovery must use saved evidence, not this page.
    saved = await browser_stay_evidence(BrowserStayEvidenceParams(action="read", source=source.id, start=9, limit=1))
    assert saved.success and "JPY 600" in json.loads(saved.output)["lines"]
    rejected = await browser_stay_evidence(BrowserStayEvidenceParams(action="begin", query=query.model_copy(update={"count": 2})))
    assert not rejected.success and "fixed" in rejected.error
    filled = await browser_stay_evidence(BrowserStayEvidenceParams(action="record", candidate=candidate, fields={"taxes_quote": taxes}))
    assert filled.success and filled.metadata["retrieval_acceptance"]["status"] == "incomplete"
    finished = await browser_stay_evidence(BrowserStayEvidenceParams(action="finish"))
    assert finished.success and finished.metadata["retrieval_acceptance"]["status"] == "pass"
    assert json.loads(finished.output)["offers"][0]["taxes_quote"] == TEXT.splitlines()[8]
    assert finished.metadata["source_count"] == 1
