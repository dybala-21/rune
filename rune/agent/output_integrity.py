"""Output-integrity checks: deterministic, model-free verification of the output.

Citation integrity: every URL cited in the produced output must appear in a tool
result (search result or fetched page) or a web_fetch call argument. A cited URL
that was never retrieved is ungrounded. Agent-generated content (assistant text,
file_write arguments) is not counted as retrieval. Conservative: when no
retrieved URLs can be determined, the check skips (never blocks).

Enabled via RUNE_OUTPUT_INTEGRITY (off by default).
"""

from __future__ import annotations

import os
import re
from urllib.parse import unquote

from rune.utils.logger import get_logger

log = get_logger(__name__)

_OUTPUT_INTEGRITY_ENV = "RUNE_OUTPUT_INTEGRITY"
_OFF = frozenset({"0", "false", "no", "off"})
_URL_RE = re.compile(r"""https?://[^\s)\]}>"'`]+""")


def output_integrity_enabled() -> bool:
    """On unless explicitly disabled: no model call, and it nudges not blocks.

    A floor, not a guarantee — it catches a citation the run never fetched.
    Link validity runs above 94% while the facts hung off those links are
    right only 39-77% of the time (arXiv 2605.06635).
    """
    return os.environ.get(_OUTPUT_INTEGRITY_ENV, "1").strip().lower() not in _OFF


def _norm(url: str) -> str:
    """Normalize a URL for comparison. Percent-decode so a citation written in
    decoded form (e.g. non-ASCII path) matches the same URL retrieved in
    percent-encoded form; without this, a legitimately retrieved non-ASCII URL is
    falsely flagged as ungrounded."""
    return unquote(url).rstrip(".,;:!?")


def _urls(text: str) -> set[str]:
    return {_norm(u) for u in _URL_RE.findall(text or "")}


def retrieved_urls(messages: list) -> set[str]:
    """URLs the system actually surfaced: tool-result contents and web_fetch call
    arguments. Excludes agent-generated content (assistant text, file_write)."""
    seen: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if (m.get("role") or m.get("type")) == "tool":
            seen |= _urls(str(m.get("content", "")))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {} if isinstance(tc, dict) else {}
            name = str(fn.get("name", "")).lower()
            if "fetch" in name or "search" in name:
                seen |= _urls(str(fn.get("arguments", "")))
    return seen


def fabricated_citations(output: str, messages: list) -> list[str]:
    """Cited URLs that were never retrieved. Empty when nothing is cited or when
    no retrieval can be determined (conservative: do not block)."""
    cited = _urls(output)
    seen = retrieved_urls(messages)
    log.info("output_integrity_check", cited=len(cited), retrieved=len(seen))
    if not cited or not seen:
        return []
    return sorted(u for u in cited if u not in seen)


def build_nudge(urls: list[str]) -> str:
    return (
        "[Output Integrity] These URLs are cited but never appeared in any search "
        "result or page you fetched. Remove them, or actually retrieve them before "
        "citing:\n" + "\n".join(f"- {u}" for u in urls)
    )


# Numeric claims


# A measurement: optional sign, thousands separators, decimals, and a unit
# that makes it a quantity. Bare integers are excluded — "section 3" is not
# a claim.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                       # not mid-identifier (gpt-5.6, v1.2)
    (?:\$|₩|€|£)?\s*                  # currency, if any
    (?P<num>[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,016.59
             |[+-]?\d+\.\d+                      # 6.10
             |[+-]?\d+)                           # 22
    \s*(?P<unit>%|percent|퍼센트|원|달러|dollars?)?
    """,
    re.VERBOSE | re.IGNORECASE,
)
# yyyy-mm-dd, yyyy/mm/dd and yyyy년 mm월: dates are not measurements.
_DATE_RE = re.compile(r"\d{4}\s*[-/년]\s*\d{1,2}\s*[-/월]?\s*\d{0,2}")
_TICKER_RE = re.compile(r"\([A-Z]{1,5}\)")
# A number welded into a name — gpt-5.6-sol, claude-opus-4-5, v1.2.3 — is a
# version, not a quantity. The lookbehind cannot see the letters before the
# hyphen, so these are removed before matching.
_IDENTIFIER_RE = re.compile(r"[A-Za-z][\w.]*-[\w.-]*\d[\w.-]*")


def _canon(raw: str) -> str:
    """One spelling per value, so 6.1 and 6.10 and 1,016.59 compare equal."""
    text = raw.replace(",", "").lstrip("+")
    try:
        value = float(text)
    except ValueError:
        return text
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def numeric_claims(text: str) -> set[str]:
    """Quantities an answer asserts: percentages, prices, money.

    An unmarked number counts only with a sign or a decimal point, which is
    what separates "+22" and "6.10" from "section 3".
    """
    if not text:
        return set()
    cleaned = _IDENTIFIER_RE.sub(" ", _TICKER_RE.sub(" ", _DATE_RE.sub(" ", text)))
    out: set[str] = set()
    for match in _NUMBER_RE.finditer(cleaned):
        raw = match.group("num")
        if not match.group("unit") and not (raw[0] in "+-" or "." in raw):
            continue
        out.add(_canon(raw))
    return out


def _rounded_from(target: str, known: set[str]) -> bool:
    """Whether *target* is a retrieved value written to fewer decimals.

    "about 6.7%" for a source saying 6.67% is quoting, not inventing.
    """
    try:
        goal = float(target)
    except ValueError:
        return False
    decimals = len(target.split(".")[1]) if "." in target else 0
    for item in known:
        try:
            value = float(item)
        except ValueError:
            continue
        if round(value, decimals) == goal:
            return True
    return False


# Pair scanning is quadratic. A page of quotes carries a handful of prices;
# among twenty thousand numbers a matching pair is coincidence, not
# derivation — so the bound serves accuracy as much as speed.
_DERIVE_MAX_VALUES = 120
# Below this a derived percentage is noise: with enough values some pair is
# always within a rounding error of zero.
_DERIVE_MIN_PERCENT = 0.1


def _derivable(target: str, known: set[str]) -> bool:
    """Whether *target* follows arithmetically from numbers already retrieved.

    A percentage computed from two quoted prices appears verbatim in no
    source, and flagging it would punish exactly the behaviour we want. Only
    percent change between retrieved pairs is modelled; anything richer is
    left to a human.
    """
    try:
        goal = abs(float(target))
    except ValueError:
        return False
    if goal < _DERIVE_MIN_PERCENT:
        return False

    values: list[float] = []
    for item in known:
        try:
            value = float(item)
        except ValueError:
            continue
        if value:
            values.append(value)
        if len(values) >= _DERIVE_MAX_VALUES:
            break

    tolerance = max(0.05, goal * 0.005)
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            for hi, lo in ((a, b), (b, a)):
                change = abs((hi - lo) / lo) * 100
                if abs(change - goal) <= tolerance:
                    return True
    return False


def unsourced_numbers(output: str, messages: list) -> list[str]:
    """Quantities in *output* that appear in nothing the run retrieved.

    Conservative: nothing retrieved means nothing to check, formatting
    differences do not count as missing, and a figure derivable from
    retrieved values counts as supported. Flags for a human, never blocks —
    a wrongly refused answer costs more than a number with a caveat.
    """
    claimed = numeric_claims(output)
    if not claimed:
        return []
    retrieved_text = _retrieved_text(messages)
    if not retrieved_text.strip():
        return []
    # Looser rule for sources: a bare integer in an article ("Rises 8")
    # grounds a claim of 8%, though it would not count as a claim itself.
    known = {_canon(m.group("num")) for m in _NUMBER_RE.finditer(retrieved_text)}
    missing = [
        c
        for c in sorted(claimed)
        if c not in known
        and not _rounded_from(c, known)
        and not _derivable(c, known)
    ]
    if missing:
        log.info("unsourced_numbers", count=len(missing), values=missing[:5])
    return missing


# Scanning is linear in retrieved text and a run can carry several pages.
# Past this the check is reading a data dump, not an answer's sources.
_MAX_SOURCE_CHARS = 400_000


def _retrieved_text(messages: list) -> str:
    """Everything the run actually surfaced: tool results and search arguments."""
    parts: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if (m.get("role") or m.get("type")) == "tool":
            parts.append(str(m.get("content", "")))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {} if isinstance(tc, dict) else {}
            if "fetch" in str(fn.get("name", "")).lower() or "search" in str(
                fn.get("name", "")
            ).lower():
                parts.append(str(fn.get("arguments", "")))
    joined = "\n".join(parts)
    if len(joined) > _MAX_SOURCE_CHARS:
        log.debug("unsourced_numbers_source_truncated", chars=len(joined))
        return joined[:_MAX_SOURCE_CHARS]
    return joined
