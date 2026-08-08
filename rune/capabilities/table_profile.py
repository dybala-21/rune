"""Mechanically computed facts about a tabular file, shown at read time.

A model that reads a CSV and aggregates it in its head gets the sum
wrong in a specific, repeatable way: duplicate rows are counted twice,
silently, every time (measured 17/17 on the office bench). The model is
not lying — a duplicated row is invisible once the file scrolls past.

So when file_read returns a .csv/.tsv, the observation carries a footer
of facts computed by code over the WHOLE file, independent of whatever
window the read requested: row count, exact-duplicate rows, blank
cells, and for numeric columns the sum both with and without the
duplicates. The footer states observations, never instructions — which
total is right depends on what the duplicates mean, and that stays the
model's call to make with the facts in view.

Reads that go through the shell (`cat data.csv`) bypass this, the same
way they bypass every observation hook; the footer is an aid on the
common path, not a guard.
"""

from __future__ import annotations

import csv
import io
import os
from collections import Counter
from decimal import Decimal, InvalidOperation

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_TABLE_PROFILE"
_EXTENSIONS = {".csv": ",", ".tsv": "\t"}
_MAX_ROWS = 200_000
_MAX_COLUMNS = 15
_MAX_DUP_EXAMPLES = 3


def profile_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def _decimal(value: str) -> Decimal | None:
    try:
        d = Decimal(value.strip())
    except InvalidOperation:
        return None
    # Decimal happily parses "NaN" and "Infinity"; a column of those is
    # not numeric data and a footer must never claim sum=NaN as a fact.
    return d if d.is_finite() else None


def _fmt(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def profile_table(text: str, filename: str) -> str:
    """The footer for *text* read from *filename*, or "" when not a table."""
    if not profile_enabled():
        return ""
    ext = os.path.splitext(filename)[1].lower()
    delimiter = _EXTENSIONS.get(ext)
    if delimiter is None or not text.strip():
        return ""

    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        log.debug("table_profile_parse_failed", file=filename,
                  error=str(exc)[:120])
        return ""
    if len(rows) < 2:
        return ""
    header, data = rows[0], rows[1:]
    if len(header) < 2:
        return ""
    if len(data) > _MAX_ROWS:
        return (
            f"\n[table profile skipped: {len(data)} data rows exceeds "
            f"{_MAX_ROWS}]"
        )

    counts = Counter(tuple(r) for r in data)
    extra_copies = sum(c - 1 for c in counts.values() if c > 1)

    if extra_copies:
        seen: set[tuple[str, ...]] = set()
        dup_lines: list[int] = []
        for i, row in enumerate(data):
            key = tuple(row)
            if key in seen and counts[key] > 1:
                dup_lines.append(i + 2)  # 1-based, after the header line
                if len(dup_lines) >= _MAX_DUP_EXAMPLES:
                    break
            seen.add(key)
        examples = ", ".join(str(n) for n in dup_lines)
        suffix = ", …" if extra_copies > len(dup_lines) else ""
        dup_note = (
            f"{extra_copies} exact duplicate row(s) — repeat(s) of an earlier "
            f"row (line {examples}{suffix})"
        )
    else:
        dup_note = "no exact duplicate rows"

    blank_notes: list[str] = []
    sum_notes: list[str] = []
    for col in range(min(len(header), _MAX_COLUMNS)):
        name = header[col].strip() or f"column {col + 1}"
        values = [r[col] for r in data if col < len(r)]
        blanks = sum(1 for v in values if not v.strip())
        if blanks:
            blank_notes.append(f"{name}: {blanks}")
        present = [v for v in values if v.strip()]
        parsed = [_decimal(v) for v in present]
        if present and all(p is not None for p in parsed):
            total = sum(p for p in parsed if p is not None)
            note = f"{name} sum={_fmt(total)}"
            if extra_copies:
                unique_total = Decimal(0)
                for key in counts:
                    if col < len(key) and key[col].strip():
                        d = _decimal(key[col])
                        if d is not None:
                            unique_total += d
                note += f" ({_fmt(unique_total)} without the duplicate rows)"
            sum_notes.append(note)

    lines = [
        "",
        "[table profile — computed by code over the whole file, not by "
        "the model]",
        f"data rows: {len(data)} (plus header); {dup_note}",
    ]
    if blank_notes:
        lines.append("blank cells: " + "; ".join(blank_notes))
    if sum_notes:
        lines.append("numeric column sums: " + "; ".join(sum_notes))
    if len(header) > _MAX_COLUMNS:
        lines.append(
            f"(first {_MAX_COLUMNS} of {len(header)} columns profiled)"
        )
    log.info(
        "table_profile",
        file=filename,
        rows=len(data),
        duplicates=extra_copies,
    )
    return "\n".join(lines)
