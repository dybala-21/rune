"""Independent checks against page text captured by the harness, never by the model.

These establish quote provenance and request consistency, not the meaning of
arbitrary cancellation prose or a guarantee of the final checkout amount.
"""

import hashlib
import json
import re
from datetime import date
from typing import Literal
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StayRequest(BaseModel):
    city: str
    city_labels: list[str] = Field(default_factory=list)
    market: Literal["domestic", "overseas"] = "domestic"
    check_in: date
    check_out: date
    adults: int = Field(gt=0)
    max_total_krw: int = Field(gt=0)
    count: int = Field(default=3, gt=0)


class Offer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    url: str
    room: str = Field(min_length=1)
    total_krw: int = Field(gt=0)
    price_quote: str = Field(min_length=1)
    stay_quote: str = Field(min_length=1)
    occupancy_quote: str = Field(min_length=1)
    location_quote: str = Field(min_length=1)
    cancellation_quote: str = Field(min_length=1)
    price_conditions_quote: str = Field(min_length=1)
    taxes_quote: str = Field(min_length=1)
    evidence_refs: dict[str, dict] = Field(default_factory=dict)


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    offers: list[Offer]
    limitations: list[str]


def normalized(value: str) -> str:
    return " ".join(value.split())


def nol_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.hostname == "nol.yanolja.com"
                and not parsed.username and not parsed.password and parsed.port in {None, 443})
    except ValueError:
        return False


def numbers(value: str) -> set[int]:
    # Parse integer amounts with optional thousands separators.
    return {int(item.replace(",", "")) for item in re.findall(r"(?<![\d.,])\d+(?:,\d{3})*(?!\d|[.,]\d)", value)}


def has_date(value: str, expected: date) -> bool:
    return any(expected.strftime(fmt) in value for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"))


def verify_comparison(answer: str, request: StayRequest, snapshots: list[dict]) -> dict:
    failures = []
    try:
        comparison = Comparison.model_validate(json.loads(answer.strip()))
    except (ValueError, ValidationError) as exc:
        return {"passed": False, "failures": [f"Invalid comparison JSON: {type(exc).__name__}"],
                "offers_checked": 0}
    if len(comparison.offers) != request.count:
        failures.append(f"Expected {request.count} offers, got {len(comparison.offers)}")
    if comparison.limitations:
        failures.append("Unresolved limitations: " + "; ".join(comparison.limitations))
    identities = set()
    for index, offer in enumerate(comparison.offers):
        prefix = f"Offer {index + 1}: "
        if not nol_url(offer.url):
            failures.append(prefix + "not an official property detail URL")
            continue
        identity = urlsplit(offer.url).path.rstrip("/")
        pattern = (r"(/stay/domestic/\d+)" if request.market == "domestic"
                   else r"(/stay/overseas/TDP-\d+)(?:/list)?")
        detail = re.fullmatch(pattern, identity)
        if detail is None:
            failures.append(prefix + "not an official detail URL for the requested market")
        else:
            identity = detail[1]
        if identity in identities:
            failures.append(prefix + "duplicate property")
        identities.add(identity)
        if offer.total_krw > request.max_total_krw:
            failures.append(prefix + "over budget")
        pages = [page for page in snapshots if page.get("url") == offer.url
                 and page.get("http_status", 200) < 400 and page.get("text")]
        if not pages:
            failures.append(prefix + "no independently captured detail page")
            continue
        quotes = [getattr(offer, field) for field in Offer.model_fields if field.endswith("_quote")]
        # Legacy quotes must coexist. Typed citations can join separate captures,
        # but each must independently match the same property and stay conditions.
        if offer.evidence_refs:
            fields = {"name": offer.name, "room": offer.room,
                      **{field: getattr(offer, field) for field in Offer.model_fields if field.endswith("_quote")}}
            if set(offer.evidence_refs) != set(fields):
                failures.append(prefix + "missing or unexpected evidence references")
            rates = {ref.get("rate_id") for ref in offer.evidence_refs.values()}
            if len(rates) != 1 or not next(iter(rates), None):
                failures.append(prefix + "field references disagree on the selected rate")
            for field, value in fields.items():
                ref = offer.evidence_refs.get(field, {})
                if (request.market == "overseas" and field in {"price_quote", "cancellation_quote", "price_conditions_quote"}
                        and not urlsplit(ref.get("url", "")).path.rstrip("/").endswith("/list")):
                    failures.append(prefix + f"{field} uses a property teaser instead of the room/rate view")
                candidates = [page for page in snapshots if page.get("url") == ref.get("url")
                              and page.get("http_status", 200) < 400 and page.get("text")]
                valid = False
                for page in candidates:
                    lines = page["text"].splitlines()
                    start, end = ref.get("start"), ref.get("end")
                    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
                        continue
                    quoted = "\n".join(lines[start - 1:end]).strip()
                    fingerprint = hashlib.sha256((page["url"] + "\n" + page["text"]).encode()).hexdigest()
                    path = urlsplit(page["url"]).path.rstrip("/").removesuffix("/list")
                    params = parse_qs(urlsplit(page["url"]).query)
                    scope = {"checkInDate": [str(request.check_in)], "checkOutDate": [str(request.check_out)],
                             "rooms[0][adult]": [str(request.adults)]}
                    valid |= (quoted == value and fingerprint == ref.get("source")
                              and path == identity == ref.get("property_id") and ref.get("room") == offer.room
                              and nol_url(page["url"]) and all(params.get(key) == value for key, value in scope.items())
                              and not any(key.startswith("rooms[") and key not in scope for key in params))
                if not valid:
                    failures.append(prefix + f"{field} lacks an exact independently captured scoped citation")
        else:
            matching = [page for page in pages if all(normalized(quote) in normalized(page["text"])
                                                     for quote in [offer.name, *quotes])]
            if not matching:
                failures.append(prefix + "quotes do not coexist in one observed page state")
        if normalized(offer.room) not in normalized(offer.price_quote):
            failures.append(prefix + "price quote does not identify the selected room")
        if offer.total_krw not in numbers(offer.price_quote):
            failures.append(prefix + "price is absent from the room quote")
        if not any(label in offer.location_quote for label in [request.city, *request.city_labels]):
            failures.append(prefix + "requested city is absent from location evidence")
        if request.adults not in numbers(offer.occupancy_quote):
            failures.append(prefix + "requested occupancy is absent from evidence")
        date_evidence = unquote(offer.url) + " " + offer.stay_quote
        if not all(has_date(date_evidence, day) for day in (request.check_in, request.check_out)):
            failures.append(prefix + "requested dates are absent from evidence")
    return {"passed": not failures, "failures": failures, "offers_checked": len(comparison.offers),
            "scope": "Observed quote provenance, distinct properties, dates, numeric occupancy and quoted price; "
                     "cancellation/fee semantics require review; no booking or checkout verification."}
