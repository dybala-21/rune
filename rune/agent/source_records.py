"""Compare JSON projections with individual retrieved records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

_MAX_CHARS = 30_000
_MAX_RECORDS = 512
_MAX_SOURCES = 8


@dataclass(frozen=True)
class SourceReview:
    status: str = "not_checked"
    checked: int = 0
    unchecked: int = 0
    detail: str = ""
    source_ids: tuple[str, ...] = ()

    def snapshot(self) -> dict:
        return {"status": self.status, "checked": self.checked, "unchecked": self.unchecked,
                "detail": self.detail, "sourceIds": list(self.source_ids), "scope": "record_fields"}


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _records(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    if len(text) > _MAX_CHARS or not text.startswith(("{", "[")):
        return []
    try:
        data = json.loads(text, parse_float=Decimal, parse_int=Decimal, parse_constant=_reject_constant)
    except (ValueError, RecursionError, InvalidOperation):
        return []
    rows = []

    def visit(node: Any, scope: dict, depth: int) -> None:
        if depth > 16 or len(rows) >= _MAX_RECORDS:
            raise ValueError("Record limit exceeded")
        if isinstance(node, list):
            for child in node:
                visit(child, scope, depth + 1)
        elif isinstance(node, dict):
            # Only ancestors supply context; sibling records cannot contribute fields.
            row = {**scope, **{key: value for key, value in node.items() if not isinstance(value, (dict, list))}}
            if len(row) > 64:
                raise ValueError("Field limit exceeded")
            if row:
                rows.append(row)
            for child in node.values():
                if isinstance(child, (dict, list)):
                    visit(child, row, depth + 1)

    try:
        visit(data, {}, 0)
    except ValueError:
        return []
    return rows


@dataclass
class SourceRecords:
    documents: list[list[dict[str, Any]]] = field(default_factory=list)
    complete: bool = True
    source_ids: list[str] = field(default_factory=list)

    def observe(self, text: str, source_id: str = "") -> None:
        rows = _records(text)
        if not rows:
            self.complete = False
        if rows and (source_id, rows) not in list(zip(self.source_ids, self.documents, strict=True)):
            if len(self.documents) == _MAX_SOURCES:
                self.complete = False
                return
            self.documents.append(rows)
            self.source_ids.append(source_id)

    def review(self, answer: str) -> SourceReview:
        claims = _records(answer)
        if not claims:
            return SourceReview(detail="This check covers structured record fields, not prose or derived claims.")
        projections = {}
        checked, unchecked, sources = 0, 0, set()
        for claim in claims:
            if len(claim) < 2:
                unchecked += 1
                continue
            keys = tuple(sorted(claim))
            if keys not in projections:
                candidates = [(source, row) for source, doc in zip(self.source_ids, self.documents, strict=True)
                              for row in doc if claim.keys() <= row.keys()]
                # Renamed fields and calculated outputs need semantic checks, not guessed mappings.
                numeric = any(isinstance(row[key], Decimal) for _, row in candidates for key in keys)
                projections[keys] = {}
                if numeric:
                    for source, row in candidates:
                        projections[keys].setdefault(tuple((type(row[key]), row[key]) for key in keys), set()).add(source)
            if not projections[keys]:
                unchecked += 1
                continue
            matches = projections[keys].get(tuple((type(claim[key]), claim[key]) for key in keys))
            if matches:
                checked += 1
                sources.update(source for source in matches if source)
                continue
            if not self.complete:
                unchecked += 1
                continue
            return SourceReview("contradicted", checked, unchecked, (
                "The JSON answer combines fields that do not match a single retrieved record. "
                "Check the entity, value, unit and observation time together, including inherited context. "
                "Use the source fields without mixing records or replacing observation time with retrieval time. "
                "Correct the answer from the existing evidence; retrieve another source only if needed."
            ), tuple(sorted(sources)))
        status = "supported" if checked and not unchecked else "insufficient" if checked or not self.complete else "not_checked"
        return SourceReview(status, checked, unchecked, source_ids=tuple(sorted(sources)))

    def blocker(self, answer: str) -> str | None:
        review = self.review(answer)
        return review.detail if review.status == "contradicted" else None
