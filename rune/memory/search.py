"""Hybrid search for RUNE memory.

5-stage pipeline:
  1. Candidate retrieval (FAISS dense + keyword, parallel)
  2. RRF fusion (rank-based, scale-invariant)
  3. Temporal decay (30-day half-life, evergreen exemptions)
  4. Source boost (project > global > learned > episodes)
  5. MMR diversity re-ranking (prevent near-duplicate results)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from rune.memory.tuning import get_tuning_config
from rune.memory.types import Fact, SearchResult, VectorMetadata
from rune.memory.vector import KeywordIndex, VectorStore
from rune.utils.logger import get_logger

log = get_logger(__name__)

# RRF constant (standard value from Cormack et al.)
_RRF_K = 60

# Rank assigned to documents missing from a result list
_MISSING_RANK = 1000

# Temporal decay: ln(2)/30 gives 30-day half-life
_DECAY_LAMBDA = math.log(2) / 30.0

# Source boost multipliers
_SOURCE_BOOST: dict[str, float] = {
    "project_md": 1.3,
    "md_fact": 1.1,      # global MEMORY.md + learned.md
    "md_daily": 1.0,
    "md_monthly": 0.7,
    "md_rule": 1.0,
    "md_profile": 0.8,
    "episode": 0.9,
    "durable_fact": 1.0,
}

# Types exempt from temporal decay (evergreen)
_EVERGREEN_TYPES = {"md_fact", "durable_fact", "md_rule", "md_profile", "project_md"}

# Candidate pool multiplier
_CANDIDATE_MULTIPLIER = 4


async def hybrid_search(
    query: str,
    vectors: VectorStore,
    keywords: KeywordIndex,
    k: int = 5,
    type_filter: str | None = None,
) -> list[SearchResult]:
    """Run the 5-stage hybrid search pipeline.

    Returns up to k results, ranked by fused relevance with diversity.
    """
    fetch_k = k * _CANDIDATE_MULTIPLIER

    # Stage 1: Candidate retrieval
    vector_results = _retrieve_vectors(query, vectors, fetch_k, type_filter)
    if isinstance(vector_results, list):
        pass  # sync path (mock/test)
    else:
        vector_results = await vector_results

    keyword_results = _retrieve_keywords(query, keywords, fetch_k, type_filter)

    if not vector_results and not keyword_results:
        return []

    # Single-source fallback (no fusion needed)
    tuning = get_tuning_config()
    min_score = float(tuning["semantic_min_score"])

    if not vector_results:
        return [r for r in keyword_results if r.score >= min_score][:k]
    if not keyword_results:
        return [r for r in vector_results if r.score >= min_score][:k]

    # Stage 2: RRF fusion
    fused = _rrf_fuse(vector_results, keyword_results)

    # Stage 3: Temporal decay
    _apply_temporal_decay(fused)

    # Stage 4: Source boost
    _apply_source_boost(fused)

    # Re-sort after decay + boost
    fused.sort(key=lambda r: r.score, reverse=True)

    # Stage 5: MMR diversity re-ranking
    top_pool = fused[:k * _CANDIDATE_MULTIPLIER]
    selected = _mmr_select(top_pool, k=k, lambda_=0.7)

    return selected


# Stage 1: Retrieval

async def _retrieve_vectors(
    query: str,
    vectors: VectorStore,
    fetch_k: int,
    type_filter: str | None,
) -> list[SearchResult]:
    try:
        from rune.llm.local_embedding import get_embedding_provider
        provider = get_embedding_provider()
        embedding = await provider.embed_single(query)
        return vectors.search(embedding, k=fetch_k, type_filter=type_filter)
    except Exception as exc:
        log.warning("vector_search_failed", error=str(exc))
        return []


def _retrieve_keywords(
    query: str,
    keywords: KeywordIndex,
    fetch_k: int,
    type_filter: str | None,
) -> list[SearchResult]:
    try:
        return keywords.search(query, k=fetch_k, type_filter=type_filter)
    except Exception as exc:
        log.warning("keyword_search_failed", error=str(exc))
        return []


# Stage 2: RRF Fusion

def _rrf_fuse(*result_lists: list[SearchResult]) -> list[SearchResult]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    score(d) = sum(1 / (k + rank_i(d))) for each list i.
    Documents missing from a list get rank = _MISSING_RANK.
    """
    scores: dict[str, float] = {}
    meta_lookup: dict[str, VectorMetadata] = {}
    text_lookup: dict[str, str] = {}

    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            meta_lookup.setdefault(r.id, r.metadata)
            if r.text and not text_lookup.get(r.id):
                text_lookup[r.id] = r.text

    # Add penalty for documents missing from some lists
    all_ids = set(scores)
    for results in result_lists:
        present = {r.id for r in results}
        for rid in all_ids - present:
            scores[rid] += 1.0 / (_RRF_K + _MISSING_RANK)

    fused = [
        SearchResult(
            id=rid,
            score=score,
            metadata=meta_lookup[rid],
            text=text_lookup.get(rid, ""),
        )
        for rid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


# Stage 3: Temporal Decay

def _apply_temporal_decay(results: list[SearchResult]) -> None:
    """Apply exponential decay to non-evergreen results in place.

    Decay formula: score *= exp(-lambda * age_days)
    30-day half-life: at 30 days, score is halved.
    MEMORY.md facts and learned.md facts are exempt (evergreen).
    """
    now = datetime.now(UTC)
    for r in results:
        if r.metadata.type in _EVERGREEN_TYPES:
            continue
        if not r.metadata.timestamp:
            continue
        try:
            ts = datetime.fromisoformat(r.metadata.timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            r.score *= math.exp(-_DECAY_LAMBDA * age_days)
        except (ValueError, TypeError):
            pass


# Stage 4: Source Boost

def _apply_source_boost(results: list[SearchResult]) -> None:
    """Multiply scores by source-type boost factors in place."""
    for r in results:
        source_type = r.metadata.type
        # Detect monthly summaries
        if source_type == "md_daily" and r.id and ".md::" not in r.id:
            source_type = "md_monthly"
        # Detect project-scoped items
        if r.metadata.category == "project" or (
            hasattr(r.metadata, "source_file")
            and "project::" in getattr(r.metadata, "source_file", "")
        ):
            source_type = "project_md"
        boost = _SOURCE_BOOST.get(source_type, 1.0)
        r.score *= boost


# Stage 5: MMR Diversity

def _mmr_select(
    candidates: list[SearchResult],
    k: int,
    lambda_: float = 0.7,
) -> list[SearchResult]:
    """Maximal Marginal Relevance re-ranking.

    Selects k results balancing relevance and diversity:
        mmr(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, selected)

    Uses Jaccard similarity on tokenized text to measure redundancy.
    """
    if len(candidates) <= k:
        return candidates

    selected: list[SearchResult] = []
    remaining = list(candidates)

    # Pre-tokenize for Jaccard
    token_cache: dict[str, set[str]] = {}
    for r in remaining:
        text = (r.metadata.summary or r.text or r.id).lower()
        token_cache[r.id] = set(text.split())

    while len(selected) < k and remaining:
        best_idx = -1
        best_mmr = -float("inf")

        for i, cand in enumerate(remaining):
            relevance = cand.score

            # Max similarity to any already-selected result
            max_sim = 0.0
            cand_tokens = token_cache.get(cand.id, set())
            for sel in selected:
                sel_tokens = token_cache.get(sel.id, set())
                if cand_tokens or sel_tokens:
                    intersection = len(cand_tokens & sel_tokens)
                    union = len(cand_tokens | sel_tokens)
                    if union > 0:
                        sim = intersection / union
                        max_sim = max(max_sim, sim)

            mmr = lambda_ * relevance - (1 - lambda_) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


# Durable fact overlay (unchanged)

def append_durable_facts(
    results: list[SearchResult],
    query: str,
    durable_facts: list[Fact],
) -> list[SearchResult]:
    """Append matching durable facts to search results."""
    query_lower = query.lower()
    for fact in durable_facts:
        if (
            query_lower in fact.key.lower()
            or query_lower in fact.value.lower()
        ):
            results.append(SearchResult(
                id=f"durable:{fact.category}:{fact.key}",
                score=fact.confidence,
                metadata=VectorMetadata(
                    type="durable_fact",
                    id=f"durable:{fact.category}:{fact.key}",
                    timestamp=fact.last_verified,
                    summary=f"[{fact.category}] {fact.key}: {fact.value}",
                ),
            ))
    return results
