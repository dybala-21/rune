"""Build markdown tables from data.

Port of shared/markdown-table.ts - generates aligned ASCII markdown
tables from header + rows data.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum


class Align(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    alignments: Sequence[Align] | None = None,
    pad: int = 1,
) -> str:
    """Build a markdown table string.

    Parameters
    ----------
    headers:
        Column header strings.
    rows:
        Iterable of row data (each row is a sequence of cell strings).
    alignments:
        Per-column alignment. Defaults to left for every column.
    pad:
        Number of spaces on each side of cell content.

    Returns
    -------
    str
        The rendered markdown table.
    """
    col_count = len(headers)
    aligns = list(alignments) if alignments else [Align.LEFT] * col_count
    while len(aligns) < col_count:
        aligns.append(Align.LEFT)

    # Normalise rows to strings and ensure column count matches
    str_rows: list[list[str]] = []
    for row in rows:
        cells = [str(c) for c in row]
        while len(cells) < col_count:
            cells.append("")
        str_rows.append(cells[:col_count])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Ensure minimum width of 3 for the separator dashes
    widths = [max(w, 3) for w in widths]

    def _pad_cell(text: str, width: int, align: Align) -> str:
        space = " " * pad
        if align == Align.RIGHT:
            return space + text.rjust(width) + space
        if align == Align.CENTER:
            return space + text.center(width) + space
        return space + text.ljust(width) + space

    def _separator(width: int, align: Align) -> str:
        dashes = "-" * (width + 2 * pad)
        if align == Align.CENTER:
            return ":" + dashes[1:-1] + ":"
        if align == Align.RIGHT:
            return dashes[:-1] + ":"
        return dashes

    # Build lines
    header_line = "|" + "|".join(
        _pad_cell(h, widths[i], aligns[i]) for i, h in enumerate(headers)
    ) + "|"

    sep_line = "|" + "|".join(
        _separator(widths[i], aligns[i]) for i in range(col_count)
    ) + "|"

    data_lines: list[str] = []
    for row in str_rows:
        line = "|" + "|".join(
            _pad_cell(cell, widths[i], aligns[i]) for i, cell in enumerate(row)
        ) + "|"
        data_lines.append(line)

    return "\n".join([header_line, sep_line, *data_lines])
