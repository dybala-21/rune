"""Keep exact observations outside model context and join only compatible stay scopes.

Design references: benchmarks/nol_comparison/hybrid-research.json. This checks
provenance and requested stay scope, not checkout availability or policy semantics.
"""

import hashlib
import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

QUOTE_FIELDS = ("price_quote", "stay_quote", "occupancy_quote", "location_quote",
                "cancellation_quote", "price_conditions_quote", "taxes_quote")
FIELDS = ("name", "room", *QUOTE_FIELDS)


def normalize(text: str) -> str:
    return " ".join(text.split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def numbers(text: str) -> set[int]:
    return {int(value.replace(",", "")) for value in re.findall(r"(?<![\d.,])\d+(?:,\d{3})*(?!\d|[.,]\d)", text)}


class StayQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=1)
    city_labels: list[str] = Field(default_factory=list, max_length=8)
    check_in: date
    check_out: date
    adults: int = Field(gt=0, le=20)
    rooms: Literal[1] = 1
    count: int = Field(default=3, ge=1, le=10)
    max_total_krw: int = Field(gt=0, description="Maximum displayed room charge for the stay; additional taxes are quoted separately.")

    @model_validator(mode="after")
    def dates_ordered(self) -> Self:
        if self.check_out <= self.check_in:
            raise ValueError("Check-out must follow check-in")
        return self


def nol_scope(url: str, query: StayQuery) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "nol.yanolja.com" or parsed.username
            or parsed.password or parsed.port not in {None, 443}):
        raise ValueError("Only an observed official NOL detail page is supported")
    match = re.fullmatch(r"/stay/(?:overseas/TDP-\d+|domestic/\d+)(?:/list)?", parsed.path.rstrip("/"))
    if not match:
        raise ValueError("Open a property detail page before capturing evidence")
    params = parse_qs(parsed.query)
    expected = {"checkInDate": str(query.check_in), "checkOutDate": str(query.check_out),
                "rooms[0][adult]": str(query.adults)}
    if any(params.get(key) != [value] for key, value in expected.items()):
        raise ValueError("The observed URL does not match the fixed dates and occupancy")
    if any(key.startswith("rooms[") and key not in expected for key in params):
        raise ValueError("The URL contains additional room/child conditions outside this contract")
    return parsed.path.rstrip("/").removesuffix("/list")


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    start: int = Field(ge=1, description="Inclusive 1-based line number from capture/read")
    end: int = Field(ge=1, description="Inclusive 1-based line number; copy the entire needed condition")


class Source(BaseModel):
    id: str
    url: str
    property_id: str
    observed_at: str
    text: str
    document_id: str
    interface: Literal["dom"] = "dom"

    def quote(self, span: Span) -> str:
        lines = self.text.splitlines()
        if span.end < span.start or span.end > len(lines) or span.end - span.start > 300:
            raise ValueError("Quote range is outside the captured source")
        value = "\n".join(lines[span.start - 1:span.end]).strip()
        if not value or len(value) > 16000:
            raise ValueError("Quote must be nonempty and at most 16000 characters")
        return value

    def fresh(self, now: datetime) -> bool:
        age = (now - datetime.fromisoformat(self.observed_at)).total_seconds()
        return 0 <= age <= 900


class Candidate(BaseModel):
    id: str
    property_id: str
    total_krw: int = Field(gt=0)
    fields: dict[str, Span] = Field(default_factory=dict)


class StayEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    owner: str
    run_id: str
    query: StayQuery
    sources: dict[str, Source] = Field(default_factory=dict)
    candidates: dict[str, Candidate] = Field(default_factory=dict)
    finished: bool = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(self.model_dump_json(indent=2))
        temporary.chmod(0o600)
        temporary.replace(path)

    @classmethod
    def restore(cls, path: Path, owner: str, run_id: str) -> Self:
        ledger = cls.model_validate_json(path.read_text())
        if ledger.owner != owner or ledger.run_id != run_id:
            raise ValueError("Evidence belongs to a different owner or run")
        return ledger

    def capture(self, url: str, text: str, document_id: str, now: datetime | None = None) -> Source:
        property_id = nol_scope(url, self.query)
        if len(text) > 80000 or not text.strip():
            raise ValueError("Page text is empty or too large; no truncated source is accepted")
        now = now or datetime.now(UTC)
        key = digest(url + "\n" + text)
        previous = self.sources.get(key)
        if previous and previous.fresh(now):
            return previous
        if len(self.sources) >= 60 and key not in self.sources:
            raise ValueError("Evidence source limit reached; finish with the missing fields reported")
        source = Source(id=key, url=url, text=text, property_id=property_id,
                        observed_at=now.isoformat(), document_id=document_id)
        self.sources[key] = source
        self.finished = False
        return source

    def read(self, source_id: str, start: int = 1, limit: int = 300) -> dict[str, Any]:
        source = self.sources[source_id]
        lines = source.text.splitlines()
        if start > len(lines):
            raise ValueError("Start is outside this source")
        selected: list[str] = []
        size = 0
        for index in range(start - 1, min(len(lines), start - 1 + limit)):
            line = f"{index + 1}: {lines[index]}"
            if len(line) > 14000:
                if selected:
                    break
                raise ValueError("This source line exceeds the read limit; use a narrower rendered view")
            if selected and size + len(line) > 14000:
                break
            selected.append(line)
            size += len(line)
        return {"source": source.id, "url": source.url, "property_id": source.property_id,
                "observed_at": source.observed_at, "lines": "\n".join(selected),
                "next_line": start + len(selected) if start + len(selected) <= len(lines) else None,
                "total_lines": len(lines)}

    def find(self, source_id: str, text: str, start: int = 1) -> dict[str, Any]:
        """Find literal text in a saved source and return numbered excerpts."""
        if not text.strip() or len(text) > 200:
            raise ValueError("Supply an exact observed label or phrase of 1–200 characters")
        source = self.sources[source_id]
        lines = source.text.splitlines()
        matches = [index for index in range(start - 1, len(lines)) if text.casefold() in lines[index].casefold()]
        excerpts: list[dict[str, Any]] = []
        size = consumed = 0
        for index in matches[:8]:
            excerpt = self.read(source_id, max(1, index - 2), 8)
            if excerpts and size + len(excerpt["lines"]) > 12000:
                break
            excerpts.append(excerpt)
            size += len(excerpt["lines"])
            consumed += 1
        return {"source": source.id, "url": source.url, "query": text,
                "matches": excerpts,
                "next_match_line": matches[consumed] + 1 if len(matches) > consumed else None,
                "guidance": "Use these exact line numbers to record missing fields; read a wider range only if needed."}

    def _quote(self, span: Span) -> str:
        if span.source not in self.sources:
            raise ValueError("Source was not captured by this run")
        return self.sources[span.source].quote(span)

    def record(self, fields: dict[str, Span], total_krw: int | None = None, candidate_id: str = "") -> Candidate:
        if not fields or set(fields) - set(FIELDS):
            raise ValueError("Use only the supplied evidence field names")
        values = {name: self._quote(span) for name, span in fields.items()}
        properties = {self.sources[span.source].property_id for span in fields.values()}
        if len(properties) != 1:
            raise ValueError("Fields from different properties cannot be joined")
        if candidate_id:
            if candidate_id not in self.candidates:
                raise ValueError("Unknown candidate ID. Omit candidate to create a record, or copy its issued ID from status.")
            original = self.candidates[candidate_id]
            candidate = original.model_copy(deep=True)
            if total_krw is not None and total_krw != candidate.total_krw:
                raise ValueError("A price change requires a new candidate, not overwriting this rate")
            if any(key in candidate.fields and self._quote(candidate.fields[key]) != value for key, value in values.items()):
                raise ValueError("A recorded field cannot be silently replaced; create a new candidate")
        else:
            if not {"name", "room", "price_quote"} <= fields.keys() or total_krw is None:
                raise ValueError("Start a candidate with name, room, price_quote and total_krw")
            key = digest(next(iter(properties)) + values["room"] + values["price_quote"])
            candidate = self.candidates.get(key, Candidate(id=key, property_id=next(iter(properties)), total_krw=total_krw)).model_copy(deep=True)
            if candidate.total_krw != total_krw or any(
                key in candidate.fields and self._quote(candidate.fields[key]) != value for key, value in values.items()
            ):
                raise ValueError("An existing rate cannot be overwritten; use its candidate ID to fill missing fields")
            if len(self.candidates) >= 30 and key not in self.candidates:
                raise ValueError("Candidate limit reached; fill the missing fields of existing candidates")
        if properties != {candidate.property_id}:
            raise ValueError("These fields do not belong to the selected property")
        candidate.fields.update(fields)
        name, room, price = (normalize(self._quote(candidate.fields[key])) for key in ("name", "room", "price_quote"))
        if room not in price or candidate.total_krw not in numbers(price):
            raise ValueError("The continuous price quote must contain the selected room and its price")
        for key, span in fields.items():
            source = self.sources[span.source]
            if not source.fresh(datetime.now(UTC)):
                raise ValueError("Source expired; recapture the current page")
            if (candidate.property_id.startswith("/stay/overseas/")
                    and key in {"price_quote", "cancellation_quote", "price_conditions_quote"}
                    and not urlsplit(source.url).path.rstrip("/").endswith("/list")):
                raise ValueError(f"{key} needs the selected overseas room's /list or room-detail view. "
                                 "The property teaser/coupon banner cannot establish its rate conditions. "
                                 "Reuse a saved room-list source if available; otherwise open the room details.")
            # Shared tax policies may omit the rate; rate-specific sources must show it.
            if key in {"price_quote", "cancellation_quote", "price_conditions_quote"}:
                if room not in normalize(source.text) or candidate.total_krw not in numbers(source.text):
                    raise ValueError("The rate-condition source does not show this room and price")
        if "location_quote" in candidate.fields and not any(
            label.casefold() in self._quote(candidate.fields["location_quote"]).casefold()
            for label in [self.query.city, *self.query.city_labels]
        ):
            raise ValueError("The location quote does not contain the requested city")
        if "occupancy_quote" in candidate.fields and self.query.adults not in numbers(self._quote(candidate.fields["occupancy_quote"])):
            raise ValueError("The occupancy quote does not contain the requested adult count")
        if candidate.total_krw > self.query.max_total_krw:
            raise ValueError("The displayed room charge exceeds the requested budget")
        if "stay_quote" in candidate.fields:
            quote = self._quote(candidate.fields["stay_quote"])
            if not all(any(day.strftime(fmt) in quote for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"))
                       for day in (self.query.check_in, self.query.check_out)):
                raise ValueError("Capture the visible full dates, including their year")
        self.candidates[candidate.id] = candidate
        self.finished = False
        return candidate

    def missing(self, candidate: Candidate) -> list[str]:
        now = datetime.now(UTC)
        return [key for key in FIELDS if key not in candidate.fields
                or not self.sources[candidate.fields[key].source].fresh(now)]

    def status(self) -> dict[str, Any]:
        selected: dict[str, Candidate] = {}
        for candidate in self.candidates.values():
            if not self.missing(candidate):
                selected.setdefault(candidate.property_id, candidate)
        return {"complete": len(selected) >= self.query.count, "required": self.query.count,
                "complete_properties": len(selected), "query": self.query.model_dump(mode="json"),
                "sources": [{"id": item.id, "url": item.url, "property_id": item.property_id,
                             "lines": len(item.text.splitlines())} for item in self.sources.values()],
                "candidates": [{"id": item.id, "property_id": item.property_id,
                                "name": self._quote(item.fields["name"]), "room": self._quote(item.fields["room"]),
                                "total_krw": item.total_krw, "missing": self.missing(item)}
                               for item in self.candidates.values()]}

    def result(self) -> dict[str, Any]:
        offers, used = [], set()
        for item in self.candidates.values():
            if item.property_id in used or self.missing(item):
                continue
            used.add(item.property_id)
            values = {key: self._quote(span) for key, span in item.fields.items()}
            refs = {key: {**span.model_dump(), "url": self.sources[span.source].url,
                          "observed_at": self.sources[span.source].observed_at,
                          "property_id": item.property_id, "room": values["room"], "rate_id": item.id}
                    for key, span in item.fields.items()}
            offers.append({**values, "url": self.sources[item.fields["price_quote"].source].url,
                           "total_krw": item.total_krw, "evidence_refs": refs})
            if len(offers) == self.query.count:
                break
        missing = self.query.count - len(offers)
        return {"offers": offers, "limitations": [f"{missing} properties still lack required fresh evidence"] if missing else []}

    def acceptance(self) -> dict[str, Any]:
        ready = self.finished and self.status()["complete"]
        times = [datetime.fromisoformat(source.observed_at) for source in self.sources.values()]
        partial = self.result()
        if not ready and not partial["limitations"]:
            partial["limitations"] = ["Evidence collected, but comparison has not been finalized"]
        return {"status": "pass" if ready else "incomplete", "required": True,
                "expires_at": (min(times) + timedelta(seconds=900)).isoformat() if times else "",
                "scope": "property/room/rate quote provenance and fixed stay scope; displayed charge, not final checkout total",
                "contract": digest(self.query.model_dump_json()), "progress": self.status(),
                "answer": json.dumps(self.result(), ensure_ascii=False) if self.finished else "",
                "partial_answer": json.dumps(partial, ensure_ascii=False)}
