"""Collect browser evidence with persistent sources and exact line citations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from rune.capabilities.browser.core import _get_browser
from rune.capabilities.browser.session import BrowserSession, browser_operation, current_session
from rune.capabilities.browser.stay_dom import capture_rates, inspect_rate
from rune.capabilities.browser.stay_readiness import wait_for_stay_content
from rune.lodging.evidence import FIELDS, Span, StayEvidence, StayQuery, digest
from rune.lodging.nol import entrypoints, source_guidance
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.capabilities.registry import CapabilityRegistry

log = get_logger(__name__)


class BrowserStayEvidenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["begin", "capture", "inspect_rate", "read", "find", "record", "status", "finish"]
    query: StayQuery | None = None
    source: str = Field(default="", description="Issued source ID for read/find; leave empty for capture (the tool issues it).")
    rate: str = Field(default="", description="Issued DOM rate ID from capture. inspect_rate opens its read-only price and policy panels; record auto-fills these exact fields when source/rate are provided.")
    start: int = Field(default=1, ge=1, description="1-based saved line. Leave unset for capture's header/detail preview.")
    limit: int = Field(default=120, ge=1, le=300, description="Maximum lines for read, or capture with explicit start.")
    text: str = Field(default="", description="For find: a literal observed label/phrase to locate in saved source text; no network access.")
    candidate: str = Field(default="", description="Leave empty to create a candidate. To fill missing fields, copy the ID returned by record/status; never invent an ID.")
    total_krw: int | None = Field(default=None, gt=0)
    fields: dict[str, Span] = Field(default_factory=dict, description=(
        "Exact captured line ranges for name, room, price_quote, stay_quote, occupancy_quote, "
        "location_quote, cancellation_quote, price_conditions_quote (discount/rate conditions), taxes_quote. "
        "The tool copies the original text; never synthesize or abbreviate quotes. "
        "price_quote must cover exactly one issued DOM rate card. Copy suggested_fields from inspect_rate. "
        "For a conditional price, open the selected room/rate and price details: a generic get-coupon button is "
        "not the discount conditions. Include the displayed coupon/member eligibility and rate-specific fees. "
        "Use the current displayed price, never a struck-through amount. Eligibility remains unverified; do not log in or book."
    ))


def _identity() -> tuple[BrowserSession, str, Path]:
    from rune.agent.run_control import current_control
    from rune.utils.paths import rune_data

    session = current_session()
    control = current_control()
    run_id = str(control.run_id) if control else session.id
    path = rune_data() / "browser-evidence" / digest(session.id) / (digest(run_id) + ".json")
    return session, run_id, path


@browser_operation
async def browser_stay_evidence(params: BrowserStayEvidenceParams) -> CapabilityResult:
    try:
        session, run_id, path = _identity()
        ledger = session.stay_evidence
        if ledger is None or ledger.run_id != run_id:
            ledger = StayEvidence.restore(path, session.id, run_id) if path.exists() else None
        if params.action == "begin":
            if params.query is None:
                raise ValueError("begin requires the user's full city, dates, occupancy, count and budget")
            if ledger is not None and ledger.query != params.query:
                raise ValueError("The comparison contract is fixed; it cannot be weakened after collection starts")
            ledger = ledger or StayEvidence(owner=session.id, run_id=run_id, query=params.query)
            session.stay_evidence = ledger
        if ledger is None:
            raise ValueError("Use begin with the requested stay conditions before collecting evidence")
        session.stay_evidence = ledger
        payload = {}
        if params.action == "begin":
            payload = {"entrypoints": entrypoints(ledger.query),
                       "navigation": "For overseas stays, use overseas_search, then follow observed property links. "
                                     "Use home for domestic stays. Verify dates/guests on each detail page. "
                                     "Check the property's address is in the requested city before collecting room rates; "
                                     "search results may include nearby cities. "
                                     "Do not invent property IDs or treat search prices as confirmed rates."}
        if params.action == "capture":
            _, page = await _get_browser()
            response = session.navigation_response
            if response is not None and session.navigation_page is page and response.status >= 400:
                raise ValueError(f"HTTP {response.status}: error pages cannot be evidence")
            url = page.url
            await wait_for_stay_content(page)
            text, rates = await capture_rates(page)
            if page.url != url:
                raise ValueError("Page changed during capture; observe the new page before collecting")
            source = ledger.capture(url, text, session.elements.document_id if session.elements else "", rates=rates)
            payload = (ledger.read(source.id, params.start, params.limit)
                       if "start" in params.model_fields_set else ledger.preview(source.id))
            payload["captured"] = True
            payload.update(source_guidance(source.url))
            if rates:
                payload["rates"] = [{"id": r.id, "room": r.room, "current_price": r.current_price,
                                     "price_label": r.price_label, "struck_prices": r.struck_prices}
                                    for r in rates[:6]]
                payload["rate_count"] = len(rates)
                payload["rate_guidance"] = "Choose an issued rate under budget, then inspect_rate(source, rate). It opens only price/policy details, saves complete regions and returns suggested_fields."
        elif params.action == "inspect_rate":
            _, page = await _get_browser()
            await inspect_rate(page, ledger, params.source, params.rate,
                               session.elements.document_id if session.elements else "")
            session.needs_observation = True
            suggested, total = ledger.rate_fields(params.source, params.rate)
            payload = {"source": params.source, "rate": params.rate, "total_krw": total,
                       "suggested_fields": {k: v.model_dump() for k, v in suggested.items()},
                       "quotes": {k: ledger.sources[v.source].quote(v) for k, v in suggested.items()},
                       "guidance": "record with source/rate auto-fills these exact fields. Supply only missing stay/occupancy/location quotes from saved sources. This is a displayed conditional rate; personal coupon eligibility and checkout are unverified."}
        elif params.action == "read":
            payload = ledger.read(params.source, params.start, params.limit)
        elif params.action == "find":
            payload = ledger.find(params.source, params.text, params.start)
        elif params.action == "record":
            fields, total = dict(params.fields), params.total_krw
            if params.rate:
                suggested, current_price = ledger.rate_fields(params.source, params.rate)
                fields = {**suggested, **fields}
                total = current_price if total is None else total
            candidate = ledger.record(fields, total, params.candidate)
            payload = {"recorded": candidate.id, "missing": ledger.missing(candidate)}
        elif params.action == "finish":
            ledger.finished = True
            payload = ledger.result()
        ledger.save(path)
        acceptance = ledger.acceptance()
        acceptance["evidence_path"] = str(path)
        if params.action != "finish":
            payload["progress"] = ledger.status(compact=params.action not in {"begin", "status"})
        if params.action in {"begin", "status"}:
            payload["guidance"] = (
                "Record available fields promptly (a new candidate needs name, room, price_quote and total_krw; omit candidate). "
                "Keep recorded fields. Use find with an observed label to locate missing information in saved sources; "
                "read only those line ranges before navigating again. "
                "Capture another page only for conditions absent from saved sources. "
                "After the required distinct properties are complete, call finish; its exact JSON is the final result."
            )
        return CapabilityResult(success=True, output=json.dumps(payload, ensure_ascii=False), metadata={
            "retrieval_acceptance": acceptance, "evidence_path": str(path),
            "source_count": len(ledger.sources), "required_fields": FIELDS,
            "url": payload.get("url", ""),
        })
    except (ValueError, KeyError) as exc:
        log.info("stay_evidence_rejected", action=params.action, error=str(exc)[:180])
        return CapabilityResult(success=False, error=str(exc), metadata={"action_status": "not_executed"})
    except Exception as exc:
        log.warning("stay_evidence_failed", action=params.action, error=str(exc)[:180])
        return CapabilityResult(success=False, error=f"Evidence collection failed: {exc}")


def register_stay_evidence(registry: CapabilityRegistry) -> None:
    from rune.capabilities.types import CapabilityDefinition
    from rune.types import Domain, RiskLevel

    registry.register(CapabilityDefinition(
        name="browser_stay_evidence", group="browser", domain=Domain.BROWSER, risk_level=RiskLevel.LOW,
        parameters_model=BrowserStayEvidenceParams, execute=browser_stay_evidence,
        description=("Collect accommodation comparison evidence without booking. NOL currently supported. "
                     "begin fixes the requested stay conditions and returns official search entry points; "
                     "capture preserves DOM room/rate boundaries and current versus struck-through prices on property detail/room-list pages. inspect_rate opens the selected rate's read-only price and policy dialogs, validates its identity and stores complete cancellation/tax/discount regions. "
                     "not search results, stores full page text and returns a bounded "
                     "header/trailing-detail preview with original line numbers (leave start unset); "
                     "record copies exact line ranges into a property/room/rate record; read/find retrieve saved lines without "
                     "visiting the page again; status lists candidates and missing fields; finish validates distinct properties and "
                     "returns the final JSON. Taxes and discount conditions need separate quotes. "
                     "For overseas rates, price/cancellation/discount evidence must come from the room list/detail view, "
                     "not the property's teaser or general coupon banner. "
                     "Use for multi-property comparisons to avoid losing evidence or retyping quotes. "
                     "Only inspect_rate clicks read-only details and closes them. Never logs in, claims coupons or makes reservations."),
    ))
