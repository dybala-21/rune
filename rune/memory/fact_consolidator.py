"""Mem0-style fact extraction and 4-op consolidation.

Ported from src/memory/fact-consolidator.ts - extracts durable facts
from task execution results, merges with existing memory using 4 operations:
ADD (new), UPDATE (changed), DELETE (contradicted), NOOP (identical).

Two-tier extraction:
  - Tier 1: Heuristic pattern matching (<1ms, zero cost, always runs)
  - Tier 2: LLM extraction (via router, optional, graceful failure)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rune.memory.store import Fact
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

FactCategory = Literal["preference", "environment", "user"]


@dataclass(slots=True)
class ExtractedFact:
    key: str
    value: str
    category: FactCategory


@dataclass(slots=True, frozen=True)
class _FactPattern:
    pattern: re.Pattern[str]
    category: FactCategory
    key: str
    value_group: int


# Tier 1: Heuristic Pattern Extraction

_FACT_PATTERNS: list[_FactPattern] = [
    # Package manager
    _FactPattern(
        re.compile(r"(?:using|chose|prefer)\s+(pnpm|yarn|npm)\b", re.I),
        "preference", "package_manager", 1,
    ),
    # Test runner
    _FactPattern(
        re.compile(r"(?:using)\s+(vitest|jest|pytest|mocha|cargo\s+test)\b", re.I),
        "preference", "test_runner", 1,
    ),
    # Framework
    _FactPattern(
        re.compile(
            r"(?:using|built\s+with)\s+"
            r"(react|vue|svelte|angular|next\.?js|nuxt|express|fastify|hono|actix|axum|gin|echo)\b",
            re.I,
        ),
        "environment", "framework", 1,
    ),
    # Database
    _FactPattern(
        re.compile(
            r"(?:using|connect)\s+(?:to\s+)?"
            r"(postgresql|postgres|mysql|mongodb|sqlite|redis|dynamodb)\b",
            re.I,
        ),
        "environment", "database", 1,
    ),
    # Language preference
    _FactPattern(
        re.compile(
            r"(?:prefer|use)\s+(typescript|python|rust|go|java|kotlin|swift)\s+(?:for)",
            re.I,
        ),
        "preference", "preferred_language", 1,
    ),
    # Bundler
    _FactPattern(
        re.compile(r"(?:using)\s+(vite|webpack|esbuild|tsup|rollup|turbopack)\b", re.I),
        "preference", "bundler", 1,
    ),
    # Port
    _FactPattern(
        re.compile(r"(?:port)\s*[:=]?\s*(\d{4,5})\b", re.I),
        "environment", "default_port", 1,
    ),
    # Node version
    _FactPattern(
        re.compile(r"node\s*(?:version|v)?\s*[:=]?\s*(v?\d{2,}(?:\.\d+)*)", re.I),
        "environment", "node_version", 1,
    ),
    # Deploy target
    _FactPattern(
        re.compile(
            r"(?:deploy)\s+(?:to|target)\s+"
            r"(vercel|netlify|aws|gcp|cloudflare|fly\.io|railway|heroku)\b",
            re.I,
        ),
        "environment", "deploy_target", 1,
    ),
    # --- Korean patterns ---
    # Preference (선호/사용/좋아하는)
    _FactPattern(
        re.compile(r"(?:선호하는|사용하는|좋아하는)\s*(?:.*?)\s*(pnpm|yarn|npm|vite|webpack)\b", re.I),
        "preference", "package_manager_ko", 1,
    ),
    _FactPattern(
        re.compile(
            r"(?:선호하는|사용하는|좋아하는)\s*(?:.*?)\s*"
            r"(react|vue|svelte|angular|next\.?js|nuxt|express|fastify|django|flask|fastapi)\b",
            re.I,
        ),
        "preference", "framework_ko", 1,
    ),
    _FactPattern(
        re.compile(
            r"(?:선호하는|사용하는|좋아하는)\s*(?:.*?)\s*"
            r"(typescript|python|rust|go|java|kotlin|swift)\b",
            re.I,
        ),
        "preference", "preferred_language_ko", 1,
    ),
    # Habit (항상/보통/주로)
    _FactPattern(
        re.compile(r"(?:항상|보통|주로)\s*(?:.*?)\s*(pnpm|yarn|npm)\b", re.I),
        "preference", "package_manager_habit_ko", 1,
    ),
    _FactPattern(
        re.compile(r"(?:항상|보통|주로)\s*(?:.*?)\s*(vitest|jest|pytest|mocha)\b", re.I),
        "preference", "test_runner_habit_ko", 1,
    ),
    # Work context (프로젝트/작업/개발)
    _FactPattern(
        re.compile(
            r"(?:프로젝트|작업|개발).*(?:에서|으로)\s*"
            r"(typescript|python|rust|go|java|kotlin|swift)\b",
            re.I,
        ),
        "environment", "project_language_ko", 1,
    ),
    _FactPattern(
        re.compile(
            r"(?:프로젝트|작업|개발).*(?:에서|으로)\s*"
            r"(postgresql|postgres|mysql|mongodb|sqlite|redis)\b",
            re.I,
        ),
        "environment", "project_database_ko", 1,
    ),
]


def extract_facts_heuristic(goal: str, answer: str) -> list[ExtractedFact]:
    """Heuristic fact extraction (<1ms, zero cost).

    Scans *goal* and *answer* text for known patterns and returns
    deduplicated extracted facts.
    """
    text = f"{goal}\n{answer}"
    facts: list[ExtractedFact] = []
    seen: set[str] = set()

    for fp in _FACT_PATTERNS:
        match = fp.pattern.search(text)
        if match and match.group(fp.value_group):
            if fp.key in seen:
                continue
            seen.add(fp.key)
            facts.append(ExtractedFact(
                key=fp.key,
                value=match.group(fp.value_group).lower().strip(),
                category=fp.category,
            ))

    return facts


# Tier 2: LLM Fact Extraction (optional)

async def extract_facts_with_llm(goal: str, answer: str) -> list[ExtractedFact]:
    """Extract facts using an LLM via the router (optional, graceful failure).

    Returns up to 3 facts. On any failure returns an empty list.
    """
    try:
        from rune.llm.router import get_llm_router

        router = get_llm_router()
        input_text = f"Task: {goal[:200]}\nResult: {answer[:500]}"

        system_prompt = (
            "Extract durable facts worth remembering from this task execution.\n"
            "Output a JSON array of objects with {key, value, category}.\n"
            "Categories: preference, environment, user.\n"
            "Keys: lowercase_snake_case. Values: concise (<100 chars).\n"
            "Only extract PERSISTENT information (preferences, decisions, environment details).\n"
            "Skip temporary task details, error messages, file contents.\n"
            "Max 3 facts. If nothing worth remembering, output []."
        )

        result = await router.generate(
            system=system_prompt,
            messages=[{"role": "user", "content": input_text}],
            complexity="simple",
            action="classify",
            max_tokens=300,
        )

        json_match = re.search(r"\[[\s\S]*?\]", result.text)
        if not json_match:
            return []

        parsed = json_decode(json_match.group())
        valid_categories = {"preference", "environment", "user"}
        return [
            ExtractedFact(key=f["key"], value=f["value"], category=f["category"])
            for f in parsed
            if f.get("key") and f.get("value") and f.get("category") in valid_categories
        ][:3]
    except Exception:
        return []


# 4-op Consolidation

async def apply_fact_operations(facts: list[ExtractedFact]) -> None:
    """Apply extracted facts using 4-op consolidation against markdown.

    For each fact:
      - SKIP if key is suppressed (user deleted it before)
      - SKIP if key exists in MEMORY.md (user-curated wins)
      - NOOP if identical value already exists in learned.md
      - UPDATE if key exists in learned.md with a different value
      - ADD if key is new
    """
    from rune.memory.manager import get_memory_manager
    from rune.memory.markdown_store import learned_md_has_key, memory_md_has_key
    from rune.memory.state import is_suppressed, record_conflict

    manager = get_memory_manager()
    await manager.initialize()

    for fact in facts:
        try:
            if is_suppressed(fact.key):
                log.debug("fact_suppressed", key=fact.key)
                continue

            if memory_md_has_key(fact.key):
                log.debug("fact_in_memory_md", key=fact.key)
                continue

            existing_value = learned_md_has_key(fact.key)

            if existing_value == fact.value:
                continue

            if existing_value is not None:
                log.debug("fact_update", key=fact.key, old=existing_value, new=fact.value)
                record_conflict(
                    key=fact.key,
                    old_value=existing_value,
                    new_value=fact.value,
                    old_source="learned_md",
                    new_source="consolidated",
                )
            else:
                log.debug("fact_add", key=fact.key, value=fact.value)

            await manager.save_fact(Fact(
                category=fact.category,
                key=fact.key,
                value=fact.value,
                source="consolidated",
                confidence=0.8 if existing_value else 0.6,
            ))
        except Exception as exc:
            log.debug("fact_operation_failed", key=fact.key, error=str(exc))


# Public API

async def consolidate_facts(goal: str, answer: str, *, use_llm: bool = True) -> int:
    """Extract and consolidate facts from a task execution.

    Runs Tier 1 (heuristic) always, and Tier 2 (LLM) optionally.
    Returns the number of facts applied.
    """
    # Tier 1: heuristic
    facts = extract_facts_heuristic(goal, answer)

    # Tier 2: LLM (if enabled and heuristic didn't find much)
    if use_llm and len(facts) < 2:
        llm_facts = await extract_facts_with_llm(goal, answer)
        # Merge, preferring heuristic facts (they're cheaper / more reliable)
        seen_keys = {f.key for f in facts}
        for f in llm_facts:
            if f.key not in seen_keys:
                facts.append(f)
                seen_keys.add(f.key)

    if facts:
        await apply_fact_operations(facts)
        log.info("facts_consolidated", count=len(facts))

    return len(facts)
