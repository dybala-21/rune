"""Output-integrity checks: deterministic, model-free verification of the output.

Citation integrity: every URL cited in the produced output must appear in a tool
result (search result or fetched page) or a web_fetch call argument. A cited URL
that was never retrieved is ungrounded. Agent-generated content (assistant text,
file_write arguments) is not counted as retrieval. Conservative: when no
retrieved URLs can be determined, the check skips (never blocks).

Enabled via RUNE_OUTPUT_INTEGRITY (off by default).
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from rune.utils.env import env_flag
from rune.utils.logger import get_logger

log = get_logger(__name__)

_OUTPUT_INTEGRITY_ENV = "RUNE_OUTPUT_INTEGRITY"
_URL_RE = re.compile(r"""https?://[^\s)\]}>"'`]+""")


def output_integrity_enabled() -> bool:
    return env_flag(_OUTPUT_INTEGRITY_ENV)


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
