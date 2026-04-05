"""Rehydration retrieval — find and format relevant compacted records.

Uses the existing hybrid_search pipeline (RRF + temporal decay + MMR).
No new ranking logic.  Query synthesis is structural composition only
(no regex on natural language, no language-specific rules).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from rune.agent.rehydration.protocols import LoopStateView, SignalReading
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Query synthesis
def synthesize_query(state: LoopStateView, reading: SignalReading) -> str:
    """Build a retrieval query from loop state.

    Pure structural composition.  The embedding model handles all
    semantic matching, so this text concentrates relevant keywords.
    No language-specific patterns.
    """
    parts = [state.goal()]
    recent = state.recent_tool_names(n=3)
    if recent:
        parts.append("recent: " + " ".join(recent))
    parts.append(f"phase: {state.activity_phase}")
    return " | ".join(parts)


# Retrieval result
@dataclass(slots=True)
class RehydrationResult:
    step: int
    activity_phase: str
    tool_name: str
    role: str
    summary: str
    score: float


# Rehydrate
def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()[:200]).hexdigest()[:16]


async def rehydrate(
    state: LoopStateView,
    reading: SignalReading,
    session_id: str,
    *,
    k: int = 3,
    existing_context_hashes: set[str] | None = None,
) -> list[RehydrationResult]:
    """Find compacted records relevant to current state.

    Filters by session_id and deduplicates against current context.
    """
    from rune.memory.manager import get_memory_manager

    mgr = get_memory_manager()
    query = synthesize_query(state, reading)

    results = await mgr.search(
        query,
        k=k * 2,  # overfetch for dedup/filter
        type_filter="compacted_turn",
    )

    # Session filter — no cross-session bleed
    filtered: list[Any] = []
    for r in results:
        extra = r.metadata.extra
        if extra.get("session_id") == session_id:
            filtered.append(r)

    # Dedup against current context
    if existing_context_hashes:
        filtered = [
            r
            for r in filtered
            if _content_hash(r.metadata.summary) not in existing_context_hashes
        ]

    # Convert to RehydrationResult
    output: list[RehydrationResult] = []
    for r in filtered[:k]:
        extra = r.metadata.extra
        output.append(
            RehydrationResult(
                step=extra.get("step", 0),
                activity_phase=extra.get("activity_phase", ""),
                tool_name=extra.get("tool_name", ""),
                role=extra.get("role", ""),
                summary=r.metadata.summary,
                score=r.score,
            )
        )

    log.debug(
        "rehydration_retrieved",
        query_len=len(query),
        candidates=len(results),
        session_filtered=len(filtered),
        returned=len(output),
    )
    return output


# Format injection
_SENTINEL_OPEN = "[[REHYDRATED_CONTEXT"
_SENTINEL_CLOSE = "[[/REHYDRATED_CONTEXT]]"


def format_injection(
    results: list[RehydrationResult], reading: SignalReading
) -> str:
    """Format rehydrated results as a system message.

    Marked with a sentinel tag so Evidence Gate and Completion Gate
    can recognize and exclude it from evidence counting.
    """
    if not results:
        return ""

    lines = [
        f"{_SENTINEL_OPEN} trigger={reading.name}]]",
        "Relevant earlier context from this session:",
        "",
    ]
    for i, r in enumerate(results, 1):
        label = r.tool_name or r.role
        lines.append(
            f"{i}. (step {r.step}, {r.activity_phase} phase, {label}) "
            f"{r.summary[:400]}"
        )
    lines.append("")
    lines.append(_SENTINEL_CLOSE)
    lines.append("This is recall, not new work. Continue your current task.")
    return "\n".join(lines)
