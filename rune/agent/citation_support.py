"""Does a cited source actually support the claim it's attached to?

``output_integrity`` already checks that a cited URL was retrieved. This checks the
next step: the page was retrieved, but does it back the claim, or is it a
misattribution? Only inline prose claims are checked against the page RUNE already
fetched — bibliography/reference lists have no specific claim to verify and are
skipped, and a citation seen only as a search snippet is skipped too (verifying
against one sentence is where most false flags come from).

The check is bounded: at most ``CAP`` of the least-covered citations are verified,
in parallel, and a keyword pre-filter skips the ones the page obviously supports
without a model call. It only flags (soft note), never rewrites. The verifier is
injected as ``averify_fn(claim, evidence) -> bool`` so the parsing here stays
model-free and testable. Opt-in via RUNE_CITATION_SUPPORT.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from urllib.parse import unquote

from rune.utils.env import env_flag
from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV = "RUNE_CITATION_SUPPORT"
_URL_RE = re.compile(r"""https?://[^\s)\]}>"'`]+""")
# Split on every sentence boundary. Gluing two sentences into one claim makes the
# verifier reject it and flags a good citation; dropping a claim only misses one.
# So we'd rather over-split than under-split, and skip an abbreviation exception list.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+", re.UNICODE)  # unicode-aware: works for CJK/non-Latin claims too
# A word that names a bibliography section (matched inside a short heading line).
_BIB_WORD = re.compile(
    r"(?i)\b(sources?|references?|citations?|bibliography|works cited"
    r"|further reading|external links|see also|footnotes?|end ?notes?)\b"
)
_HEADING = re.compile(r"^\s*#{1,6}\s+\S")  # any markdown heading (ends a biblio section)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
# script/style/head/noscript blocks whose text (JS/CSS) must not leak into evidence.
_BLOCK_RE = re.compile(r"(?is)<(script|style|head|noscript)\b[^>]*>.*?</\1>")

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in",
    "on", "at", "by", "for", "and", "or", "as", "it", "its", "this", "that", "with",
    "from", "into", "via", "see", "which", "has", "have", "had", "can", "will",
    "default", "value", "used", "using", "when", "than",
}

CAP = 8          # max citations verified per run (bounds cost)
CONCURRENCY = 4  # parallel verifier calls
_MAX_CONTENT = 60_000  # cap before regex work (bounds cost); evidence is windowed anyway
_MIN_PAGE = 400  # content shorter than this is treated as a snippet -> skip

VerifyFn = Callable[[str, str], bool]
AVerifyFn = Callable[[str, str], Awaitable[bool]]


def citation_support_enabled() -> bool:
    return env_flag(_ENV)


def _norm(url: str) -> str:
    url = unquote(url).rstrip(".,;:!?")
    if url.count("(") > url.count(")"):  # repair paren-truncated URLs (Wikipedia)
        url += ")"
    return url


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _is_bib_heading(line: str) -> bool:
    """Whether a line is a bibliography-section heading, e.g. "## Sources",
    "## Annotated Bibliography", "Notes and references", "**Sources**".

    A short heading-like line (no URL, no sentence punctuation) that mentions a
    bibliography word counts. We don't try to separate a real reference heading from
    a content heading that happens to use the word ("Primary Sources Reveal Fraud")
    — telling those apart needs real parsing. Treating both as bibliography can skip
    a content section by mistake, which is fine; the alternative would leak a
    reference list into the claims and flag a legitimate citation, which is not."""
    s = line.strip()
    if not s or _URL_RE.search(s):  # a line with a URL is content, not a heading
        return False
    core = s.strip("#*_ ").rstrip(":").strip("*_ ").strip()
    core = re.sub(r"[\(\[][^)\]]*[\)\]]", " ", core)  # drop "(peer-reviewed)", "[1]"
    core = re.sub(r"\s+", " ", core).strip()
    if not core or len(core.split()) > 8 or core[-1] in ".!?":
        return False
    return bool(_BIB_WORD.search(core))


def _split_sentences(line: str) -> list[str]:
    return _SENT_SPLIT.split(line)


def inline_claim_citations(output: str) -> list[tuple[str, str]]:
    """(claim, url) for inline citations — a URL sitting in a prose sentence that
    makes a factual claim. Bibliography lists and sentences with nothing specific to
    check are skipped."""
    pairs: list[tuple[str, str]] = []
    in_biblio = False
    for line in (output or "").splitlines():
        if _is_bib_heading(line):
            in_biblio = True
            continue
        if _HEADING.match(line):
            in_biblio = False  # a new (non-biblio) heading ends the bibliography section
        if in_biblio:
            continue  # everything under a Sources/References heading is a list
        if _LIST_MARKER.match(line):
            # Skip list items — a "title + URL" reference bullet and a short factual
            # bullet look the same without real parsing, and a real inline claim is
            # almost always prose anyway. Better to miss a bullet than flag a
            # reference list.
            continue
        for sent in _split_sentences(line):
            urls = _URL_RE.findall(sent)
            if not urls:
                continue
            claim = _URL_RE.sub("", sent)
            claim = re.sub(r"\[([^\]]*)\]\(\s*\)", r"\1", claim)
            claim = re.sub(r"[\(\[\]\)<>]", " ", claim)
            claim = re.sub(r"\s+", " ", claim).strip(" -•*|#")
            if len(_content_words(claim)) < 3:  # not a verifiable factual claim
                continue
            for u in urls:
                pairs.append((claim[:400], _norm(u)))
    return pairs


def build_url_content_map(messages: list) -> dict[str, str]:
    """{normalized_url: best retrieved content} over tool results. Prefers the
    LONGEST content for a URL (the full fetched page over a search snippet)."""
    out: dict[str, str] = {}
    for m in messages or []:
        if not isinstance(m, dict) or (m.get("role") or m.get("type")) != "tool":
            continue
        content = str(m.get("content", ""))
        for u in _URL_RE.findall(content):
            nu = _norm(u)
            if len(content) > len(out.get(nu, "")):
                out[nu] = content
    return out


def relevant_windows(content: str, claim: str, k: int = 2, window: int = 1200) -> list[str]:
    """Top-k disjoint, boilerplate-stripped windows of ``content`` most relevant to
    the claim, scored by stop-word-filtered, word-boundary keyword overlap (not
    first-match / substring). Content is length-capped first to bound regex cost."""
    content = content[:_MAX_CONTENT]  # bounds cost of the strip regexes below
    text = _BLOCK_RE.sub(" ", content)  # drop script/style/head text (JS/CSS junk)
    # `[^<>]` (not `[^>]`) so an unclosed '<' can't scan to EOF — keeps this linear.
    text = re.sub(r"<[^<>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= window:
        return [text]
    keys = set(_content_words(claim))
    if not keys:
        return [text[:window]]
    step = max(1, window // 2)
    starts = list(range(0, max(1, len(text) - window + 1), step))
    tail = len(text) - window  # always cover the final window (evidence at page end)
    if tail > 0 and starts[-1] != tail:
        starts.append(tail)
    scored: list[tuple[int, int]] = []
    for i in starts:
        w_words = set(_WORD.findall(text[i:i + window].lower()))
        score = len(keys & w_words)  # word-boundary overlap, distinct keys
        scored.append((score, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked: list[str] = []
    used: list[int] = []
    for score, i in scored:
        if score <= 0 and picked:
            break
        if any(abs(i - j) < window for j in used):
            continue  # keep windows disjoint
        picked.append(text[i:i + window])
        used.append(i)
        if len(picked) >= k:
            break
    return picked


def _coverage(evidence: str, claim: str) -> float:
    keys = set(_content_words(claim))
    if not keys:
        return 1.0
    ev = set(_WORD.findall(evidence.lower()))
    return len(keys & ev) / len(keys)


def _evidence_for(url: str, claim: str, url_map: dict[str, str]) -> str | None:
    """Joined top-k relevant windows of the retrieved full page for this citation,
    or None when there is no usable page-level evidence (snippet-only -> skip)."""
    content = url_map.get(_norm(url), "")
    if len(content) < _MIN_PAGE:
        return None  # snippet only (or nothing) -> can't fairly verify -> skip
    windows = relevant_windows(content, claim)
    if not windows:
        return None
    return "\n---\n".join(windows)[:3000]


async def averify_unsupported(
    output: str, messages: list, averify_fn: AVerifyFn,
    *, cap: int = CAP, concurrency: int = CONCURRENCY,
) -> list[tuple[str, str]]:
    """(claim, url) pairs whose retrieved full page does NOT support the claim.

    Conservative and bounded: only inline claims; only page-level (non-snippet)
    evidence; a keyword pre-filter skips obviously-supported citations; at most
    ``cap`` most-suspicious citations are model-verified, concurrently. Never
    raises — a verifier error leaves the citation unflagged."""
    pairs = inline_claim_citations(output)
    if not pairs:
        return []
    url_map = build_url_content_map(messages)

    # Build (claim, url, evidence, coverage) for citations with usable evidence.
    candidates: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for claim, url in pairs:
        if (claim, url) in seen:
            continue
        seen.add((claim, url))
        evidence = _evidence_for(url, claim, url_map)
        if evidence is None:
            continue  # snippet-only / not retrieved -> grounding's job, not flagged
        cov = _coverage(evidence, claim)
        if cov >= 0.999:
            continue  # every claim keyword present -> pre-filter as supported
        candidates.append((claim, url, evidence, cov))

    if not candidates:
        return []
    # Most-suspicious first (lowest keyword coverage), capped.
    candidates.sort(key=lambda c: c[3])
    candidates = candidates[:cap]

    sem = asyncio.Semaphore(concurrency)

    async def _check(claim: str, url: str, evidence: str):
        async with sem:
            try:
                supported = await averify_fn(claim, evidence)
            except Exception as exc:
                log.debug("citation_support_verify_error", error=str(exc)[:120])
                return None  # error -> leave unflagged (fail-safe for the run)
            return None if supported else (claim, url)

    results = await asyncio.gather(
        *(_check(c, u, e) for c, u, e, _ in candidates)
    )
    bad = [r for r in results if r is not None]
    log.info("citation_support_check",
             inline=len(pairs), checked=len(candidates), unsupported=len(bad))
    return bad


def unsupported_citations(
    output: str, messages: list, verify_fn: VerifyFn, *, cap: int = CAP,
) -> list[tuple[str, str]]:
    """Synchronous variant (for tests / non-async callers) with a sync verifier."""
    async def _a(claim: str, evidence: str) -> bool:
        return verify_fn(claim, evidence)
    return asyncio.run(averify_unsupported(output, messages, _a, cap=cap))


def build_support_note(bad: list[tuple[str, str]]) -> str:
    """Advisory note listing the citations whose page didn't seem to back the claim."""
    lines = [
        "[Citation Support] These citations may not be backed by their source — the "
        "retrieved page doesn't appear to state the claim. Re-check them, cite a page "
        "that does support the claim, or say the source is uncertain:"
    ]
    for claim, url in bad:
        safe = re.sub(r"[<>]", "", claim)[:80]
        lines.append(f'- "{safe}" -> {url}')
    return "\n".join(lines)
