"""Episode promotion engine for tiered memory.

Ported from src/memory/promotion-engine.ts - scores, filters, compacts,
and promotes episode candidates based on quality metrics and rollout mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC
from typing import Literal

from rune.memory.store import Episode
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

MemoryPolicyMode = Literal["shadow", "balanced", "strict", "legacy"]


@dataclass(slots=True)
class EpisodeCandidate:
    episode: Episode
    score: float


@dataclass(slots=True)
class PromotionDiagnostics:
    baseline_count: int = 0
    selected_count: int = 0
    shadow_agreement: float = 0.0
    compression_gain: float = 0.0
    token_reduction_rate: float = 0.0


@dataclass(slots=True)
class PromotionResult:
    selected: list[EpisodeCandidate] = field(default_factory=list)
    diagnostics: PromotionDiagnostics = field(default_factory=PromotionDiagnostics)


# Helpers

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _estimate_episode_tokens(episode: Episode) -> int:
    lessons = episode.lessons or ""
    result = episode.result or ""
    return _estimate_tokens(f"{episode.task_summary}\n{lessons}\n{result}")


def _jaccard_similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compact_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


# Quality scoring

def calculate_episode_quality(episode: Episode, now_ms: float | None = None) -> float:
    """Calculate a quality score for an episode (0..1).

    Factors: importance, success, execution depth, lesson density, recency.
    """
    import time

    if now_ms is None:
        now_ms = time.time() * 1000

    importance = _clamp01(episode.importance)

    # Derive success from result text heuristic
    result_text = (episode.result or "").lower()
    success = 1.0 if ("success" in result_text or "completed" in result_text) else 0.0

    # Lesson density: count lines in lessons field
    lesson_count = len((episode.lessons or "").strip().splitlines())
    lesson_density = _clamp01(lesson_count / 5)

    # Execution depth: heuristic from result length
    execution_depth = _clamp01(min(len(episode.result or ""), 400) / 400)

    # Recency
    try:
        from datetime import datetime

        ep_dt = datetime.fromisoformat(episode.timestamp)
        if ep_dt.tzinfo is None:
            ep_dt = ep_dt.replace(tzinfo=UTC)
        age_days = max(0.0, (datetime.now(UTC) - ep_dt).total_seconds() / 86400)
    except (ValueError, TypeError):
        age_days = 30.0
    recency = _clamp01(math.exp(-0.01 * age_days))

    return _clamp01(
        importance * 0.25
        + success * 0.25
        + execution_depth * 0.20
        + lesson_density * 0.15
        + recency * 0.15
    )


# Compaction

def compact_episode_for_mode(episode: Episode, mode: MemoryPolicyMode) -> Episode:
    """Return a compacted copy of the episode based on the rollout mode.

    Legacy/shadow modes return the episode unchanged.
    Balanced/strict modes truncate summaries and lessons.
    """
    if mode in ("legacy", "shadow"):
        return episode

    if mode == "strict":
        summary_limit = 120
        lesson_limit = 100
    else:  # balanced
        summary_limit = 180
        lesson_limit = 160

    return Episode(
        id=episode.id,
        timestamp=episode.timestamp,
        task_summary=_compact_text(episode.task_summary, summary_limit),
        intent=episode.intent,
        plan=episode.plan,
        result=_compact_text(episode.result, lesson_limit * 3) if episode.result else "",
        lessons=_compact_text(episode.lessons, lesson_limit * 2) if episode.lessons else "",
        embedding=episode.embedding,
        conversation_id=episode.conversation_id,
        importance=episode.importance,
    )


# Promotion

def promote_episode_candidates(
    baseline: list[EpisodeCandidate],
    mode: MemoryPolicyMode,
    limit: int,
    now_ms: float | None = None,
) -> PromotionResult:
    """Promote, rescore, filter, and compact episode candidates.

    For legacy/shadow modes the baseline is returned as-is.
    For balanced/strict modes, episodes are rescored by quality,
    filtered by thresholds, compacted, and capped at *limit*.
    """
    import time

    if now_ms is None:
        now_ms = time.time() * 1000

    baseline_top = baseline[:limit]
    baseline_ids = [c.episode.id for c in baseline_top]
    baseline_token_estimate = sum(_estimate_episode_tokens(c.episode) for c in baseline_top)

    # Legacy / shadow: pass through
    if mode in ("legacy", "shadow"):
        return PromotionResult(
            selected=baseline_top,
            diagnostics=PromotionDiagnostics(
                baseline_count=len(baseline_top),
                selected_count=len(baseline_top),
                shadow_agreement=1.0,
                compression_gain=0.0,
                token_reduction_rate=0.0,
            ),
        )

    # Rescore
    rescored: list[tuple[EpisodeCandidate, float, float]] = []  # (candidate, quality, promoted_score)
    for candidate in baseline_top:
        quality = calculate_episode_quality(candidate.episode, now_ms)
        if mode == "strict":
            promoted_score = candidate.score * 0.7 + quality * 0.3
        else:
            promoted_score = candidate.score * 0.8 + quality * 0.2
        rescored.append((candidate, quality, promoted_score))

    # Filter
    if mode == "strict":
        filtered = [
            (c, q, ps) for c, q, ps in rescored
            if c.score >= 0.45 and q >= 0.45
        ]
    else:
        filtered = [
            (c, q, ps) for c, q, ps in rescored
            if c.score >= 0.4 and (q >= 0.3 or ps >= 0.48)
        ]

    # Fallback to rescored if nothing passed the filter
    source = filtered if filtered else rescored
    source.sort(key=lambda x: x[2], reverse=True)
    source = source[: max(1, limit)]

    # Compact
    compacted = [
        EpisodeCandidate(
            episode=compact_episode_for_mode(c.episode, mode),
            score=ps,
        )
        for c, _, ps in source
    ]

    selected_ids = [c.episode.id for c in compacted]
    selected_token_estimate = sum(_estimate_episode_tokens(c.episode) for c in compacted)

    return PromotionResult(
        selected=compacted,
        diagnostics=PromotionDiagnostics(
            baseline_count=len(baseline_top),
            selected_count=len(compacted),
            shadow_agreement=_jaccard_similarity(baseline_ids, selected_ids),
            compression_gain=(
                _clamp01(1 - len(compacted) / len(baseline_top))
                if baseline_top else 0.0
            ),
            token_reduction_rate=(
                _clamp01(1 - selected_token_estimate / baseline_token_estimate)
                if baseline_token_estimate > 0 else 0.0
            ),
        ),
    )
