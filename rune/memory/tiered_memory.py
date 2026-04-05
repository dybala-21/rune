"""3-tier memory system - session, daily, durable.

Ported from src/memory/tiered-memory.ts - manages memory promotion
from ephemeral session context through daily aggregation to persistent
durable facts.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rune.memory.store import MemoryStore, get_memory_store
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Enums & data types

class MemoryTier(enum.Enum):
    SESSION = "session"
    DAILY = "daily"
    DURABLE = "durable"


@dataclass(slots=True)
class DailyMemoryEntry:
    """Aggregated summary for a single day."""

    date: str  # ISO date, e.g. "2025-12-01"
    goal_summaries: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    patterns_learned: list[str] = field(default_factory=list)
    total_tasks: int = 0
    successful_tasks: int = 0


@dataclass(slots=True)
class DurableMemoryEntry:
    """High-confidence persistent fact derived from accumulated evidence."""

    category: str = ""
    key: str = ""
    value: str = ""
    confidence: float = 0.0
    last_verified: str = ""
    source: str = ""


# TieredMemoryManager

class TieredMemoryManager:
    """Orchestrates the three memory tiers: session -> daily -> durable."""

    _DAILY_FACTS_CATEGORY = "daily_summary"
    _DURABLE_MIN_CONFIDENCE = 0.7

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store = store or get_memory_store()
        # Session tier: purely in-memory, scoped to current process lifetime
        self._session: dict[str, Any] = {}
        self._session_events: list[dict[str, Any]] = []

    def add_session_memory(self, key: str, value: Any) -> None:
        """Store a key-value pair in the current session context."""
        self._session[key] = value
        self._session_events.append({
            "key": key,
            "value": value,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        log.debug("session_memory_added", key=key)

    def get_session_context(self) -> dict[str, Any]:
        """Return a snapshot of all current session memory."""
        return dict(self._session)

    def promote_to_daily(self, entries: list[dict[str, Any]]) -> DailyMemoryEntry:
        """Aggregate session entries into a daily summary.

        Writes to daily/*.md (markdown) instead of SQLite facts table.
        *entries* is a list of dicts with optional keys: ``goal``,
        ``decision``, ``pattern``, ``success`` (bool).
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        daily = DailyMemoryEntry(date=today)
        for entry in entries:
            if goal := entry.get("goal"):
                daily.goal_summaries.append(str(goal))
            if decision := entry.get("decision"):
                daily.key_decisions.append(str(decision))
            if pattern := entry.get("pattern"):
                daily.patterns_learned.append(str(pattern))
            daily.total_tasks += 1
            if entry.get("success", False):
                daily.successful_tasks += 1

        # Write each goal as a daily log entry in markdown
        from rune.memory.markdown_store import append_daily_entry

        for goal in daily.goal_summaries:
            append_daily_entry(
                title=goal[:100],
                actions=[],
                lessons=[p for p in daily.patterns_learned],
                decisions=[d for d in daily.key_decisions],
                date=today,
            )

        log.info(
            "daily_summary_promoted",
            date=today,
            tasks=daily.total_tasks,
            successful=daily.successful_tasks,
        )
        return daily

    def get_daily_summary(self, date: str) -> DailyMemoryEntry | None:
        """Retrieve a daily summary from daily/*.md."""
        from rune.memory.markdown_store import memory_dir, parse_daily_log

        path = memory_dir() / "daily" / f"{date}.md"
        if not path.exists():
            return None

        try:
            entries = parse_daily_log(path)
            if not entries:
                return None

            daily = DailyMemoryEntry(date=date)
            for entry in entries:
                daily.goal_summaries.append(entry["title"])
                daily.total_tasks += 1
                daily.successful_tasks += 1  # assume success unless marked
                daily.key_decisions.extend(entry.get("decisions", []))
                daily.patterns_learned.extend(entry.get("lessons", []))
            return daily
        except Exception as exc:
            log.warning("daily_summary_parse_failed", date=date, error=str(exc))
            return None

    def promote_to_durable(self, entry: DurableMemoryEntry) -> bool:
        """Promote an entry to learned.md if confidence is sufficient."""
        confidence = self._calculate_confidence(entry)
        if confidence < self._DURABLE_MIN_CONFIDENCE:
            log.debug(
                "durable_promotion_rejected",
                key=entry.key,
                confidence=confidence,
                threshold=self._DURABLE_MIN_CONFIDENCE,
            )
            return False

        entry.confidence = confidence
        entry.last_verified = datetime.now(UTC).isoformat()

        from rune.memory.markdown_store import save_learned_fact
        save_learned_fact(
            category=entry.category,
            key=entry.key,
            value=entry.value,
            confidence=entry.confidence,
        )

        log.info("durable_memory_promoted", key=entry.key, confidence=confidence)
        return True

    def get_durable_facts(self, category: str) -> list[DurableMemoryEntry]:
        """Retrieve durable facts from MEMORY.md and learned.md."""
        from rune.memory.markdown_store import parse_learned_md, parse_memory_md

        results: list[DurableMemoryEntry] = []

        # From MEMORY.md (all facts are high-confidence)
        sections = parse_memory_md()
        for section, lines in sections.items():
            for line in lines:
                if ":" in line:
                    key, _, value = line.partition(":")
                    if category in ("preference", "project") or not category:
                        results.append(DurableMemoryEntry(
                            category=section.lower(),
                            key=key.strip(),
                            value=value.strip(),
                            confidence=1.0,
                            source="memory_md",
                        ))

        # From learned.md (filter by category, confidence >= 0.7)
        for fact in parse_learned_md():
            if fact["category"] == category and fact["confidence"] >= self._DURABLE_MIN_CONFIDENCE:
                results.append(DurableMemoryEntry(
                    category=fact["category"],
                    key=fact["key"],
                    value=fact["value"],
                    confidence=fact["confidence"],
                    source="learned_md",
                ))

        return results

    async def consolidate_daily(self) -> DailyMemoryEntry | None:
        """Run end-of-day aggregation: flush session events to daily summary.

        Returns the daily entry if there were events to consolidate,
        None otherwise.
        """
        if not self._session_events:
            log.debug("consolidate_daily_skipped", reason="no_session_events")
            return None

        # Build promotion entries from session events
        entries: list[dict[str, Any]] = []
        for evt in self._session_events:
            entries.append({
                "goal": evt.get("key", ""),
                "success": True,
            })

        loop = asyncio.get_running_loop()
        daily = await loop.run_in_executor(None, self.promote_to_daily, entries)

        # Clear session after consolidation
        self._session_events.clear()
        log.info("daily_consolidation_complete", date=daily.date)
        return daily

    def _calculate_confidence(self, entry: DurableMemoryEntry) -> float:
        """Compute a confidence score for a durable memory candidate.

        Uses the entry's existing confidence as a base and applies
        heuristics: penalise short values, reward entries with a source.
        """
        base = entry.confidence if entry.confidence > 0 else 0.5

        # Penalise very short values
        if len(entry.value) < 5:
            base *= 0.7

        # Reward entries that cite a source
        if entry.source:
            base = min(base * 1.1, 1.0)

        # Reward entries with recent verification
        if entry.last_verified:
            try:
                verified_dt = datetime.fromisoformat(entry.last_verified)
                age_hours = (datetime.now(UTC) - verified_dt).total_seconds() / 3600
                if age_hours < 24:
                    base = min(base * 1.05, 1.0)
            except (ValueError, TypeError):
                pass

        return round(min(max(base, 0.0), 1.0), 4)


# Module singleton

_manager: TieredMemoryManager | None = None


def get_tiered_memory_manager() -> TieredMemoryManager:
    global _manager
    if _manager is None:
        _manager = TieredMemoryManager()
    return _manager
