"""Fault-tolerant search-block matching for file_edit.

Match ladder: exact substring → unique line-trimmed match → unique
whitespace-normalized match → no match, with a closest-section hint.
Ambiguous fuzzy matches refuse; replacements are re-indented to the matched
block's indentation.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

# Cap for hint scanning on huge files (lines).
_HINT_SCAN_MAX_LINES = 20_000


@dataclass(slots=True)
class BlockMatch:
    """A unique fuzzy match: [start, end) line indices and the strategy."""

    start: int
    end: int
    strategy: str  # "trimmed" | "ws-normalized"
    indent_delta: str  # prefix added to (or "" ) each replace line


def _norm_trim(line: str) -> str:
    return line.strip()


def _norm_ws(line: str) -> str:
    return " ".join(line.split())


def _find_unique_window(
    content_lines: list[str], search_lines: list[str], norm
) -> tuple[int, str] | None:
    """Start index of the unique normalized window match, else None.

    Returns (start, indent_prefix). Ambiguous (2+) matches return None.
    """
    n = len(search_lines)
    if n == 0 or n > len(content_lines):
        return None
    normalized_search = [norm(line) for line in search_lines]
    found: list[int] = []
    for i in range(len(content_lines) - n + 1):
        if [norm(line) for line in content_lines[i:i + n]] == normalized_search:
            found.append(i)
            if len(found) > 1:
                return None
    if len(found) != 1:
        return None
    start = found[0]
    # Indent delta: matched first non-empty line's leading blanks minus the
    # search's — applied to replacement lines so shape follows the file.
    delta = ""
    for cl, sl in zip(
        content_lines[start:start + n], search_lines, strict=True
    ):
        if cl.strip():
            c_ind = cl[: len(cl) - len(cl.lstrip())]
            s_ind = sl[: len(sl) - len(sl.lstrip())]
            if c_ind.startswith(s_ind):
                delta = c_ind[len(s_ind):]
            break
    return start, delta


def find_block(content: str, search: str) -> BlockMatch | None:
    """Locate *search* in *content* via the fuzzy ladder (exact is handled by
    the caller). Returns None when nothing matches uniquely."""
    content_lines = content.splitlines()
    search_lines = search.splitlines()
    # Drop leading/trailing blank search lines — models pad blocks unevenly.
    while search_lines and not search_lines[0].strip():
        search_lines.pop(0)
    while search_lines and not search_lines[-1].strip():
        search_lines.pop()
    if not search_lines:
        return None
    for strategy, norm in (("trimmed", _norm_trim), ("ws-normalized", _norm_ws)):
        hit = _find_unique_window(content_lines, search_lines, norm)
        if hit is not None:
            start, delta = hit
            return BlockMatch(
                start=start, end=start + len(search_lines),
                strategy=strategy, indent_delta=delta,
            )
    return None


def apply_block(content: str, match: BlockMatch, replace: str) -> str:
    """Replace the matched line span with *replace*, re-indented."""
    content_lines = content.splitlines(keepends=True)
    replace_lines = replace.splitlines()
    if match.indent_delta:
        replace_lines = [
            (match.indent_delta + line) if line.strip() else line
            for line in replace_lines
        ]
    # Preserve the trailing newline shape of the replaced span.
    span = content_lines[match.start:match.end]
    trailing_nl = span[-1].endswith("\n") if span else True
    new_block = "\n".join(replace_lines)
    if trailing_nl and (not new_block.endswith("\n")):
        new_block += "\n"
    return "".join(content_lines[: match.start]) + new_block + "".join(
        content_lines[match.end:]
    )


def closest_section_hint(content: str, search: str, context: int = 4) -> str:
    """A short 'did you mean' hint: the file window most similar to *search*.

    Gives the model REAL lines (with numbers) to build its next search block
    from, instead of letting it retry blind variations of a stale snippet.
    """
    content_lines = content.splitlines()[:_HINT_SCAN_MAX_LINES]
    search_lines = [line.strip() for line in search.splitlines() if line.strip()]
    if not search_lines or not content_lines:
        return ""
    n = max(1, min(len(search_lines), 12))
    target = "\n".join(search_lines[:n])
    best_i, best_r = 0, 0.0
    matcher = difflib.SequenceMatcher(a=target)
    for i in range(0, len(content_lines) - n + 1):
        window = "\n".join(
            line.strip() for line in content_lines[i:i + n]
        )
        matcher.set_seq2(window)
        r = matcher.quick_ratio()
        if r > best_r:
            best_r, best_i = r, i
    if best_r < 0.4:
        return ""
    lo = max(0, best_i - 1)
    hi = min(len(content_lines), best_i + n + context - 3)
    shown = "\n".join(
        f"{idx + 1}: {content_lines[idx]}" for idx in range(lo, hi)
    )
    return f"Closest matching section (line numbers shown):\n{shown}"


# --- per-file consecutive edit-failure counter (escalation ladder) -----------

_fail_counts: dict[str, int] = {}
_FAIL_COUNTS_CAP = 512


def record_edit_failure(path: str) -> int:
    """Bump and return the consecutive failure count for *path*."""
    if len(_fail_counts) > _FAIL_COUNTS_CAP:
        _fail_counts.clear()
    _fail_counts[path] = _fail_counts.get(path, 0) + 1
    return _fail_counts[path]


def record_edit_success(path: str) -> None:
    _fail_counts.pop(path, None)


def escalation_hint(path: str, failures: int) -> str:
    """After repeated failures on one file, prescribe the way out."""
    if failures < 2:
        return ""
    return (
        f"\nThis is consecutive failure #{failures} editing {path}. Stop "
        "retrying variations of the same search string. Either: (1) re-read "
        "the file with file_read to get its EXACT current contents, (2) use "
        "a longer, unique search snippet copied verbatim from that read, or "
        "(3) rewrite the whole file with file_write."
    )
