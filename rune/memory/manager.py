"""Memory manager for RUNE.

Ported from src/memory/manager.ts - orchestrates store, vector search,
working memory, and episode promotion.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rune.memory.store import MemoryStore, get_memory_store
from rune.memory.tiered_memory import TieredMemoryManager
from rune.memory.types import (
    Episode,
    Fact,
    SearchResult,
    VectorMetadata,
    WorkingMemory,
)
from rune.memory.vector import (
    KeywordIndex,
    VectorStore,
    get_vector_store,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)


class MemoryManager:
    """Orchestrates memory subsystems: store, vectors, working memory."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._store = store or get_memory_store()
        self._vectors = vector_store or get_vector_store()
        self._keywords = KeywordIndex()
        self._working = WorkingMemory()
        self._tiered = TieredMemoryManager(store=self._store)
        self._initialized = False

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def working(self) -> WorkingMemory:
        return self._working

    @property
    def tiered(self) -> TieredMemoryManager:
        """Direct access to the tiered memory subsystem."""
        return self._tiered

    async def initialize(self) -> None:
        """Load working memory from markdown files and SQLite."""
        if self._initialized:
            return
        self._initialized = True

        from rune.memory.markdown_store import (
            ensure_memory_structure,
            parse_learned_md,
            parse_memory_md,
        )

        loop = asyncio.get_running_loop()

        # Ensure markdown structure exists on first run
        await loop.run_in_executor(None, ensure_memory_structure)

        # Load facts from MEMORY.md (Zone 1)
        sections = await loop.run_in_executor(None, parse_memory_md)
        for lines in sections.values():
            for line in lines:
                if ":" in line:
                    key, _, value = line.partition(":")
                    self._working.facts[key.strip()] = value.strip()

        # Load facts from learned.md (Zone 2)
        learned = await loop.run_in_executor(None, parse_learned_md)
        for fact in learned:
            # MEMORY.md wins on conflict (already loaded)
            if fact["key"] not in self._working.facts:
                self._working.facts[fact["key"]] = fact["value"]

        # Load safety rules from project rules.md if exists
        from rune.memory.markdown_store import parse_rules_md

        project_rules = Path.cwd() / ".rune" / "memory" / "rules.md"
        if project_rules.exists():
            rules = await loop.run_in_executor(None, parse_rules_md, project_rules)
            self._working.safety_rules = [
                {"type": r.get("type", ""), "pattern": r.get("pattern", ""), "reason": r.get("reason", "")}
                for r in rules
            ]

        # Also load global rules.md if it exists
        global_rules = memory_dir_path / "rules.md" if (memory_dir_path := Path.home() / ".rune" / "memory").exists() else None
        if not self._working.safety_rules and global_rules and global_rules.exists():
            try:
                rules = await loop.run_in_executor(None, parse_rules_md, global_rules)
                self._working.safety_rules = [
                    {"type": r.get("type", ""), "pattern": r.get("pattern", ""), "reason": r.get("reason", "")}
                    for r in rules
                ]
            except Exception:
                pass

        # Load recent commands from SQLite (operational data stays in DB)
        commands = await loop.run_in_executor(None, self._store.get_recent_commands, 20)
        self._working.recent_commands = [c["command"] for c in commands]

        # Reconcile: sync fact-meta.json with learned.md values.
        # If meta has eval data (eval_count > 0), meta wins — it has been
        # validated by actual task outcomes.  Otherwise learned.md wins.
        from rune.memory.state import load_fact_meta, save_fact_meta
        meta = await loop.run_in_executor(None, load_fact_meta)
        meta_changed = False
        for fact in learned:
            key = fact["key"]
            if key not in meta:
                continue
            if meta[key].get("eval_count", 0) > 0:
                continue  # meta has outcome data — don't overwrite
            if meta[key].get("confidence") != fact["confidence"]:
                meta[key]["confidence"] = fact["confidence"]
                meta_changed = True
        if meta_changed:
            await loop.run_in_executor(None, save_fact_meta, meta)

        log.info(
            "memory_initialized",
            facts=len(self._working.facts),
            rules=len(self._working.safety_rules),
            commands=len(self._working.recent_commands),
            source="markdown",
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured embedding provider."""
        from rune.llm.local_embedding import get_embedding_provider

        provider = get_embedding_provider()
        return await provider.embed(texts)

    async def search(
        self,
        query: str,
        k: int = 5,
        type_filter: str | None = None,
        include_durable: bool = False,
        # Deprecated: kept for backward compat, ignored by RRF pipeline
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[SearchResult]:
        """Hybrid search using RRF fusion with temporal decay and diversity."""
        from rune.memory.search import append_durable_facts, hybrid_search

        results = await hybrid_search(
            query,
            self._vectors,
            self._keywords,
            k=k,
            type_filter=type_filter,
        )

        if include_durable:
            durable_facts = []
            for category in ("preference", "project"):
                durable_facts.extend(self._tiered.get_durable_facts(category))
            results = append_durable_facts(results, query, durable_facts)

        return results

    async def promote_memories(self) -> None:
        """Consolidate session memories into the daily tier.

        Intended to be called periodically (e.g. at end of session or
        on a timer) to flush ephemeral session data into daily summaries.
        """
        daily = await self._tiered.consolidate_daily()
        if daily:
            log.info(
                "memories_promoted",
                date=daily.date,
                tasks=daily.total_tasks,
            )

    async def get_tiered_context(self, goal: str | None = None) -> str:
        """Build combined context from all memory tiers for prompt injection.

        Merges session, daily, and durable memory into a single string
        suitable for inclusion in the agent system prompt.
        """
        from datetime import datetime

        parts: list[str] = []

        # --- Session context ---
        session_ctx = self._tiered.get_session_context()
        if session_ctx:
            parts.append("## Session Memory")
            for key, value in list(session_ctx.items())[:10]:
                parts.append(f"- {key}: {value}")
            parts.append("")

        # --- Daily context (today's summary) ---
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        daily = self._tiered.get_daily_summary(today)
        if daily:
            parts.append("## Today's Progress")
            if daily.goal_summaries:
                parts.append(f"- Tasks completed: {daily.successful_tasks}/{daily.total_tasks}")
                for g in daily.goal_summaries[:5]:
                    parts.append(f"  - {g}")
            if daily.key_decisions:
                parts.append("- Key decisions:")
                for d in daily.key_decisions[:5]:
                    parts.append(f"  - {d}")
            if daily.patterns_learned:
                parts.append("- Patterns learned:")
                for p in daily.patterns_learned[:5]:
                    parts.append(f"  - {p}")
            parts.append("")

        # --- Durable context ---
        durable_facts = self._tiered.get_durable_facts("preference")
        durable_facts += self._tiered.get_durable_facts("project")
        if durable_facts:
            parts.append("## Durable Knowledge")
            for fact in durable_facts[:10]:
                parts.append(f"- [{fact.category}] {fact.key}: {fact.value}")
            parts.append("")

        if not parts:
            return ""

        return "\n".join(parts).rstrip("\n") + "\n"

    # Episode scoring & similarity

    async def score_episodes(
        self, query: str, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Score episodes using vector similarity + importance + recency decay.

        Each result dict contains:
        - ``episode``: the :class:`Episode` object
        - ``score``: composite score (0-1)
        - ``breakdown``: dict with ``vector``, ``importance``, ``recency`` sub-scores

        Recency decay uses ``exp(-days_old / 30)`` (30-day half-life).
        Composite: ``0.5 * vector + 0.3 * importance + 0.2 * recency``.
        """
        loop = asyncio.get_running_loop()
        episodes = await loop.run_in_executor(
            None, self._store.get_recent_episodes, max(limit * 5, 50),
        )
        if not episodes:
            return []

        # Try vector search for similarity scores
        vector_scores: dict[str, float] = {}
        try:
            from rune.llm.local_embedding import get_embedding_provider
            provider = get_embedding_provider()
            embedding = await provider.embed_single(query)
            vec_results = self._vectors.search(embedding, k=len(episodes))
            for r in vec_results:
                vector_scores[r.id] = max(0.0, min(1.0, r.score))
        except Exception:
            # Fallback: keyword overlap ratio
            query_terms = set(query.lower().split()) if query.strip() else set()
            if query_terms:
                for ep in episodes:
                    summary_terms = set((ep.task_summary or "").lower().split())
                    overlap = len(query_terms & summary_terms)
                    vector_scores[ep.id] = overlap / len(query_terms)

        now = datetime.now(UTC)

        scored: list[dict[str, Any]] = []
        for ep in episodes:
            # Vector/keyword similarity (0-1)
            vec_score = vector_scores.get(ep.id, 0.0)

            # Importance (0-1)
            importance = max(0.0, min(1.0, ep.importance))

            # Recency decay: exp(-days_old / 30)
            try:
                ts = datetime.fromisoformat(ep.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                days_old = (now - ts).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days_old = 120.0  # default to ~4 months
            recency = math.exp(-days_old / 30.0)

            composite = vec_score * 0.5 + importance * 0.3 + recency * 0.2

            scored.append({
                "episode": ep,
                "score": composite,
                "breakdown": {
                    "vector": vec_score,
                    "importance": importance,
                    "recency": recency,
                },
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    async def find_similar_episodes(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find episodes similar to *query* with optional session scoping.

        When *session_id* is provided, episodes from the same session
        keep their full score while episodes from other sessions receive
        a 0.9x demotion penalty.

        Returns dicts with ``episode``, ``score``, ``breakdown``.
        """
        # Fetch a broad set first
        all_scored = await self.score_episodes(query, limit=max(limit * 5, 30))
        if not all_scored:
            return []

        if session_id is None:
            return all_scored[:limit]

        # Apply session scoping
        session_results: list[dict[str, Any]] = []
        other_results: list[dict[str, Any]] = []

        for entry in all_scored:
            ep: Episode = entry["episode"]
            if ep.conversation_id == session_id:
                session_results.append(entry)
            else:
                demoted = dict(entry)
                demoted["score"] = entry["score"] * 0.9
                other_results.append(demoted)

        # Prefer same-session if there are strong matches
        if session_results and session_results[0]["score"] >= 0.5:
            combined = session_results
        else:
            combined = session_results + other_results
            combined.sort(key=lambda x: x["score"], reverse=True)

        return combined[:limit]

    async def build_memory_context(
        self, goal: str, classification: Any = None,
    ) -> str:
        """Build a markdown context string for prompt injection.

        Merges:
        - Scored episodes relevant to the goal
        - Relevant facts from working memory
        - Tiered context (session / daily / durable)
        - Lessons from top episodes

        Respects a rough token budget of ~4000 characters.
        """
        max_chars = 4000
        parts: list[str] = []
        budget_used = 0

        def _add_section(text: str) -> bool:
            nonlocal budget_used
            if budget_used + len(text) > max_chars:
                return False
            parts.append(text)
            budget_used += len(text)
            return True

        # 0. Learned rules (from Rule Learner) - domain-scoped
        try:
            from rune.memory.rule_learner import get_rules_for_domain
            domain = getattr(classification, "goal_type", None) if classification else None
            if domain:
                rules = get_rules_for_domain(domain)
                if rules:
                    rule_lines = [f"- {r['key']}: {r['value']}" for r in rules]
                    _add_section(
                        "## Learned Rules\n" + "\n".join(rule_lines[:10]) + "\n"
                    )
        except Exception:
            pass  # Rule injection is best-effort

        # 1. Scored episodes (vector/keyword) + entity search fallback
        scored = await self.score_episodes(goal, limit=7)

        # Supplement with entity-based search: extract keywords from goal
        # and find episodes by entity match (catches cases where vector
        # search misses due to empty index or cross-language mismatch).
        scored_ids = {entry["episode"].id for entry in scored}
        goal_words = [w for w in goal.lower().split() if len(w) > 1]
        for word in goal_words[:5]:
            try:
                entity_eps = self._store.get_episodes_by_entity(word, limit=3)
                for ep in entity_eps:
                    if ep.id not in scored_ids:
                        scored_ids.add(ep.id)
                        scored.append({
                            "episode": ep,
                            "score": 0.6,
                            "breakdown": {"vector": 0, "importance": ep.importance, "recency": 0.5},
                        })
            except Exception:
                pass

        if scored:
            # Keep score_episodes relevance order; utility is for labeling only
            lines: list[str] = []
            for entry in scored:
                ep: Episode = entry["episode"]
                summary = ep.task_summary or "(no summary)"
                u = getattr(ep, "utility", 0)
                if u > 0:
                    lines.append(f"- ✅ {summary} (utility: +{u})")
                elif u < 0:
                    lines.append(f"- ⚠️ {summary} (utility: {u})")
                # utility=0 episodes are neutral, skip them
            if lines:
                _add_section("## Past Experience (auto-learned)\n" + "\n".join(lines) + "\n")

        # 2. Lessons from top episodes
        lesson_lines: list[str] = []
        for entry in scored[:3]:
            ep = entry["episode"]
            if ep.lessons:
                # lessons is stored as a string (possibly JSON list or plain text)
                try:
                    from rune.utils.fast_serde import json_decode
                    lesson_list = json_decode(ep.lessons)
                    if isinstance(lesson_list, list):
                        for lesson in lesson_list:
                            lesson_lines.append(f"- {lesson}")
                except Exception:
                    if ep.lessons.strip():
                        lesson_lines.append(f"- {ep.lessons.strip()}")
        if lesson_lines:
            section = "## Lessons Learned\n" + "\n".join(lesson_lines[:5]) + "\n"
            _add_section(section)

        # 3. Relevant facts from working memory
        if self._working.facts:
            goal_lower = goal.lower()
            matching_facts = [
                (k, v)
                for k, v in self._working.facts.items()
                if goal_lower in k.lower() or goal_lower in v.lower()
                or any(w in k.lower() or w in v.lower() for w in goal_lower.split()[:3])
            ]
            if matching_facts:
                lines = ["## Relevant Facts"]
                for k, v in matching_facts[:5]:
                    lines.append(f"- {k}: {v}")
                _add_section("\n".join(lines) + "\n")

        # 4. Tiered context (session + daily + durable)
        tiered = await self.get_tiered_context(goal)
        if tiered.strip():
            _add_section(tiered)

        if not parts:
            return ""

        return "\n".join(parts).rstrip("\n") + "\n"

    def calculate_importance(
        self,
        result: dict[str, Any],
        lessons: list[str] | None = None,
    ) -> float:
        """Calculate importance score for an episode.

        Ported from ``calculateImportance`` in manager.ts:420-441.

        Scoring:
        - Base: 0.3
        - Complexity bonus (steps): 0.1-0.3
        - Lessons bonus: min(0.3, len(lessons) * 0.1)
        - Failure bonus: +0.1
        - Outputs bonus: 0.1-0.2
        """
        score = 0.3

        # Complexity bonus based on steps
        steps = result.get("steps", 0)
        if steps >= 20:
            score += 0.3
        elif steps >= 10:
            score += 0.2
        elif steps >= 5:
            score += 0.1

        # Lessons bonus
        lesson_list = lessons or []
        score += min(0.3, len(lesson_list) * 0.1)

        # Failure bonus
        if not result.get("success", True):
            score += 0.1

        # Outputs bonus (files changed, artifacts created)
        outputs = result.get("changed_files", [])
        if len(outputs) >= 5:
            score += 0.2
        elif len(outputs) >= 1:
            score += 0.1

        return min(1.0, score)

    async def add_safety_rule(
        self,
        rule_type: str,
        pattern: str,
        reason: str = "",
        source: str = "user",
    ) -> None:
        """Add a safety rule to working memory.

        Rules are stored in-memory for the session. For persistent rules,
        edit .rune/memory/rules.md directly.
        """
        self._working.safety_rules.append({
            "type": rule_type,
            "pattern": pattern,
            "reason": reason,
        })
        log.info("safety_rule_added", type=rule_type, pattern=pattern[:50])

    async def save_episode(self, episode: Episode) -> None:
        """Save an episode to both store, vector index, and session tier."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store.save_episode, episode)

        # Track in session tier for later daily consolidation
        if episode.task_summary:
            self._tiered.add_session_memory(
                episode.id,
                {
                    "task_summary": episode.task_summary,
                    "intent": episode.intent,
                    "timestamp": episode.timestamp,
                },
            )

        # Index the episode for vector search and keyword search
        if episode.task_summary:
            meta = VectorMetadata(
                type="episode",
                id=episode.id,
                timestamp=episode.timestamp,
                summary=episode.task_summary,
            )
            # Keyword index (always available)
            self._keywords.add(episode.task_summary, meta)

            # Vector index (may fail if embedding provider unavailable)
            try:
                from rune.llm.local_embedding import get_embedding_provider
                provider = get_embedding_provider()
                embedding = await provider.embed_single(episode.task_summary)
                self._vectors.add(embedding, meta)
            except Exception as exc:
                log.warning("episode_indexing_failed", error=str(exc))

    async def save_fact(self, fact: Fact) -> None:
        """Save a fact to learned.md and update working memory."""
        from rune.memory.markdown_store import save_learned_fact
        from rune.memory.state import update_fact_meta

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            save_learned_fact,
            fact.category,
            fact.key,
            fact.value,
            fact.confidence,
        )
        await loop.run_in_executor(
            None,
            update_fact_meta,
            fact.key,
            {
                "confidence": fact.confidence,
                "source": fact.source,
                "verified": fact.last_verified or "",
                "zone": "learned_md",
            },
        )
        self._working.facts[fact.key] = fact.value

    def log_command(self, command: str, success: bool, task_id: str = "") -> None:
        """Log a command execution to history."""
        self._store.log_command(command, success, task_id)
        self._working.recent_commands.insert(0, command)
        if len(self._working.recent_commands) > 50:
            self._working.recent_commands = self._working.recent_commands[:50]

    def persist_vectors(self) -> None:
        """Save vector index to disk."""
        self._vectors.save()

    def close(self) -> None:
        """Close all resources."""
        self.persist_vectors()
        self._store.close()


# Module singleton

_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
