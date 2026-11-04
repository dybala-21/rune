"""DOM rate boundaries, separate from language understanding and task routing.

Design evidence: benchmarks/nol_comparison/rate-dom-research.json.
A displayed conditional price is a quote, never proof of personal eligibility.
"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Lines(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=1)
    end: int = Field(ge=1)


class RateCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    room_id: str
    index: int = Field(ge=0)
    room: str
    room_lines: Lines
    lines: Lines
    text: str
    current_price: int = Field(gt=0)
    price_text: str
    price_label: str
    struck_prices: list[str] = Field(default_factory=list)
    signature: str


class RateRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rate_id: str
    kind: Literal["cancellation_quote", "price_conditions_quote", "taxes_quote", "name"]
    lines: Lines
    selector: str


def fingerprint(url: str, text: str, rates: list[RateCard], regions: list[RateRegion]) -> str:
    value = url + "\n" + text
    if rates or regions:
        value += "\n" + json.dumps({"rates": [r.model_dump() for r in rates],
                                   "regions": [r.model_dump() for r in regions]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def locate(text: str, excerpt: str, offset: int = 0) -> tuple[Lines, int]:
    """Locate a complete rendered DOM block without fuzzy text matching."""
    excerpt = excerpt.strip()
    at = text.find(excerpt, offset)
    if not excerpt or at < 0:
        raise ValueError("Rendered DOM block cannot be aligned to the captured text")
    end = at + len(excerpt)
    if text[text.rfind("\n", 0, at) + 1:at].strip() or text[end:text.find("\n", end) if "\n" in text[end:] else len(text)].strip():
        raise ValueError("DOM block does not occupy complete source lines")
    return Lines(start=text.count("\n", 0, at) + 1, end=text.count("\n", 0, end) + 1), end
