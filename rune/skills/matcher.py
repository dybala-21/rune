"""Skill matching for RUNE.

Ranks skills against a query by keyword overlap, fuzzy similarity, and semantic
embedding similarity. The semantic signal catches reworded same-intent goals
that lexical matching misses, and degrades to lexical-only when embeddings are
unavailable.
"""

from __future__ import annotations

import math
import os
from difflib import SequenceMatcher

from rune.skills.types import Skill, SkillMatch

# Cosine at/above this counts as a semantic match; below it only lexical is used
# (unrelated short texts still cosine ~0.3). Mirrors RUNE_RULE_SIM_THRESHOLD.
_SKILL_SIM_THRESHOLD_ENV = "RUNE_SKILL_SIM_THRESHOLD"
_DEFAULT_SKILL_SIM_THRESHOLD = 0.55


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _skill_embed_text(skill: Skill) -> str:
    return f"{skill.name}. {skill.description}"


def _semantic_scores(query: str, skills: list[Skill]) -> dict[str, float]:
    """Cosine(query, skill) keyed by skill name; {} on failure so the caller
    falls back to lexical-only. One batched embed call."""
    if not skills:
        return {}
    try:
        from rune.llm.local_embedding import get_embedding_provider

        provider = get_embedding_provider()
        texts = [query] + [_skill_embed_text(s) for s in skills]
        vecs = provider.embed_sync(texts)
        qv = vecs[0]
        return {skills[i].name: _cosine(qv, vecs[i + 1]) for i in range(len(skills))}
    except Exception:
        return {}


def _sim_threshold() -> float:
    try:
        return float(os.environ.get(_SKILL_SIM_THRESHOLD_ENV, "") or _DEFAULT_SKILL_SIM_THRESHOLD)
    except ValueError:
        return _DEFAULT_SKILL_SIM_THRESHOLD


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return {w.lower() for w in text.split() if len(w) > 1}


def _compute_similarity(query: str, skill: Skill) -> float:
    """Compute a relevance score (0.0–1.0) between a query and a skill.

    Combines:
    - Keyword overlap (Jaccard-like) between query and skill name + description
    - Fuzzy sequence matching against the skill name
    - Fuzzy sequence matching against the skill description
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return 0.0

    query_tokens = _tokenize(query_lower)
    skill_text = f"{skill.name} {skill.description}".lower()
    skill_tokens = _tokenize(skill_text)

    # Keyword overlap (Jaccard-style)
    if query_tokens and skill_tokens:
        intersection = query_tokens & skill_tokens
        union = query_tokens | skill_tokens
        keyword_score = len(intersection) / len(union)
    else:
        keyword_score = 0.0

    # Fuzzy matching against name
    name_ratio = SequenceMatcher(None, query_lower, skill.name.lower()).ratio()

    # Fuzzy matching against description
    desc_ratio = SequenceMatcher(
        None, query_lower, skill.description.lower()[:200],
    ).ratio()

    # Weighted combination
    score = (keyword_score * 0.4) + (name_ratio * 0.35) + (desc_ratio * 0.25)

    return round(min(1.0, score), 4)


def match_skills(query: str, skills: list[Skill]) -> list[SkillMatch]:
    """Match a query against skills, score = max(lexical, semantic) where
    semantic counts only above the threshold. Sorted desc, zero-scores dropped."""
    if not query.strip():
        return []

    semantic = _semantic_scores(query, skills)
    threshold = _sim_threshold()
    q_lower = query.lower()

    matches: list[SkillMatch] = []
    for skill in skills:
        lexical = _compute_similarity(query, skill)
        sem = semantic.get(skill.name, 0.0)
        sem_qualifies = sem >= threshold
        score = max(lexical, sem if sem_qualifies else 0.0)
        if score > 0.05:
            if q_lower in skill.name.lower():
                reason = f"Name contains '{query}'"
            elif q_lower in skill.description.lower():
                reason = f"Description matches '{query}'"
            elif sem_qualifies and sem >= lexical:
                reason = f"Semantic match: {sem:.0%}"
            else:
                reason = f"Similarity: {score:.0%}"
            matches.append(SkillMatch(skill=skill, score=round(score, 4), reason=reason))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches
