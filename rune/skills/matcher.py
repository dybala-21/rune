"""Skill matching for RUNE.

Matches a user query against registered skills using keyword overlap
and fuzzy string similarity.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from rune.skills.types import Skill, SkillMatch


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
    """Match a query against a list of skills.

    Returns matches sorted by descending score, filtering out zero-score results.
    """
    matches: list[SkillMatch] = []

    for skill in skills:
        score = _compute_similarity(query, skill)
        if score > 0.05:
            # Build a brief reason
            if query.lower() in skill.name.lower():
                reason = f"Name contains '{query}'"
            elif query.lower() in skill.description.lower():
                reason = f"Description matches '{query}'"
            else:
                reason = f"Similarity: {score:.0%}"
            matches.append(SkillMatch(skill=skill, score=score, reason=reason))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches
