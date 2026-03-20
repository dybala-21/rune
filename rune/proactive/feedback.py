"""Feedback learning for RUNE proactive suggestions.

Records user acceptance/dismissal of suggestions and learns
which suggestion types are effective or should be suppressed.

Ported from src/proactive/feedback.ts - the ``record_feedback`` method
now accepts the full ``Suggestion`` object and a multi-state response
('accepted', 'dismissed', 'ignored', 'annoyed'), matching the TS API.

Persistence: feedback entries are saved to the ``proactive_feedback``
table in the memory store DB on each ``record_feedback()`` call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Suppress a type if its acceptance rate falls below this after enough samples
_SUPPRESS_THRESHOLD = 0.1
_SUPPRESS_MIN_SAMPLES = 8

FeedbackResponse = Literal["accepted", "dismissed", "ignored", "annoyed"]


def _djb2(s: str) -> str:
    """djb2 hash for collision-resistant pattern key generation."""
    h = 5381
    for ch in s:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return format(h, "x")


@dataclass(slots=True)
class FeedbackEntry:
    """A single feedback record for a suggestion."""

    suggestion_id: str
    suggestion_type: str = ""
    context_summary: str = ""
    response: FeedbackResponse = "dismissed"
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class _UserPattern:
    """Learned user preference pattern."""

    pattern: str
    kind: Literal["positive", "negative"]
    strength: float
    examples: list[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))


class FeedbackLearner:
    """Learns from suggestion feedback to improve future suggestions.

    Tracks per-type acceptance rates, identifies suggestion types that
    should be suppressed, and builds pattern-based confidence adjustments.

    The ``record_feedback`` method accepts the full ``Suggestion`` plus a
    multi-state response, matching the TS ``FeedbackLearner.recordFeedback``
    API.
    """

    __slots__ = (
        "_entries",
        "_type_stats",
        "_patterns",
        "_max_recent",
    )

    def __init__(self, *, max_recent: int = 100) -> None:
        self._entries: list[FeedbackEntry] = []
        # type -> {"accepted": int, "dismissed": int, "ignored": int, "annoyed": int, "total": int}
        self._type_stats: dict[str, Counter[str]] = defaultdict(Counter)
        self._patterns: dict[str, _UserPattern] = {}
        self._max_recent = max_recent

    # Core API - matches TS FeedbackLearner.recordFeedback()

    def record_feedback(
        self,
        suggestion: Suggestion,
        response: FeedbackResponse,
    ) -> None:
        """Record feedback for a suggestion with full context.

        This is the primary API matching the TS ``feedbackLearner.recordFeedback(
        suggestion, 'accepted'|'dismissed'|'ignored'|'annoyed')``.

        Parameters:
            suggestion: The full suggestion object.
            response: Multi-state user response.
        """
        entry = FeedbackEntry(
            suggestion_id=suggestion.id,
            suggestion_type=suggestion.type,
            context_summary=suggestion.description,
            response=response,
            confidence=suggestion.confidence,
        )

        # Keep in-memory list bounded
        self._entries.insert(0, entry)
        if len(self._entries) > self._max_recent:
            self._entries.pop()

        # Update per-type stats
        self._type_stats[entry.suggestion_type]["total"] += 1
        self._type_stats[entry.suggestion_type][response] += 1

        # Update learned patterns
        self._update_patterns(entry)

        log.debug(
            "feedback_recorded",
            suggestion_id=suggestion.id,
            response=response,
            type=entry.suggestion_type,
        )

        # Persist to DB
        self._save_entry_to_db(entry)

    # Legacy compat - record() with FeedbackEntry directly

    def record(
        self,
        entry: FeedbackEntry,
        suggestion_type: str = "",
        suggestion_source: str = "",
    ) -> None:
        """Legacy record method for backward compatibility.

        Prefer ``record_feedback(suggestion, response)`` instead.
        """
        if suggestion_type:
            entry.suggestion_type = suggestion_type
        if suggestion_source:
            entry.context_summary = suggestion_source

        self._entries.insert(0, entry)
        if len(self._entries) > self._max_recent:
            self._entries.pop()

        if entry.suggestion_type:
            self._type_stats[entry.suggestion_type]["total"] += 1
            self._type_stats[entry.suggestion_type][entry.response] += 1

        self._update_patterns(entry)

        log.debug(
            "feedback_recorded",
            suggestion_id=entry.suggestion_id,
            response=entry.response,
            type=entry.suggestion_type,
        )

        self._save_entry_to_db(entry)

    # Pattern learning (ported from TS updatePatterns)

    def _update_patterns(self, entry: FeedbackEntry) -> None:
        """Update learned patterns based on a feedback entry."""
        pattern_key = f"{entry.suggestion_type}:{_djb2(entry.context_summary)}"
        is_positive = entry.response == "accepted"
        is_negative = entry.response in ("annoyed", "dismissed")

        existing = self._patterns.get(pattern_key)

        if existing is not None:
            if is_positive:
                existing.strength = min(1.0, existing.strength + 0.1)
                if existing.kind == "negative" and existing.strength > 0.5:
                    existing.kind = "positive"
            elif is_negative:
                existing.strength = max(-1.0, existing.strength - 0.1)
                if existing.kind == "positive" and existing.strength < -0.5:
                    existing.kind = "negative"
            existing.last_seen = datetime.now(UTC)
            existing.examples.append(entry.context_summary)
            if len(existing.examples) > 10:
                existing.examples.pop(0)
        else:
            self._patterns[pattern_key] = _UserPattern(
                pattern=pattern_key,
                kind="positive" if is_positive else "negative" if is_negative else "positive",
                strength=0.3 if is_positive else -0.3 if is_negative else 0.0,
                examples=[entry.context_summary],
            )

    # Confidence / suppression queries

    def get_confidence_adjustment(
        self, suggestion_type: str, context_summary: str,
    ) -> float:
        """Return confidence adjustment for a suggestion type + context.

        Positive values mean the type performs well; negative means poorly.
        """
        pattern_key = f"{suggestion_type}:{_djb2(context_summary)}"
        pattern = self._patterns.get(pattern_key)
        if pattern is None:
            return 0.0
        return pattern.strength * 0.2

    def get_pattern_adjustments(self) -> dict[str, float]:
        """Return confidence adjustments per suggestion type.

        Positive values mean the type performs well; negative means poorly.
        Types with too few samples return 0.0.
        """
        adjustments: dict[str, float] = {}
        for stype, stats in self._type_stats.items():
            total = stats["total"]
            if total < 3:
                adjustments[stype] = 0.0
                continue
            rate = stats["accepted"] / total
            adjustments[stype] = round(rate - 0.5, 3)
        return adjustments

    def should_suppress_type(self, suggestion_type: str) -> bool:
        """Return True if the given suggestion type should be suppressed."""
        stats = self._type_stats.get(suggestion_type)
        if stats is None:
            return False

        total = stats["total"]
        if total < _SUPPRESS_MIN_SAMPLES:
            return False

        rate = stats["accepted"] / total
        if rate < _SUPPRESS_THRESHOLD:
            log.info(
                "suggestion_type_suppressed",
                type=suggestion_type,
                rate=round(rate, 3),
                samples=total,
            )
            return True
        return False

    def should_suppress(
        self, suggestion_type: str, context_summary: str,
    ) -> bool:
        """Return True if a specific type+context pattern should be suppressed."""
        pattern_key = f"{suggestion_type}:{_djb2(context_summary)}"
        pattern = self._patterns.get(pattern_key)
        if pattern is None:
            return False
        return pattern.kind == "negative" and pattern.strength < -0.5

    # Stats

    def get_stats(self) -> dict[str, float | int]:
        """Return feedback statistics."""
        total = len(self._entries)
        if total == 0:
            return {
                "total_feedback": 0,
                "accept_rate": 0.0,
                "dismiss_rate": 0.0,
                "annoyed_rate": 0.0,
                "pattern_count": 0,
                "positive_patterns": 0,
                "negative_patterns": 0,
            }

        accepted = sum(1 for e in self._entries if e.response == "accepted")
        dismissed = sum(1 for e in self._entries if e.response == "dismissed")
        annoyed = sum(1 for e in self._entries if e.response == "annoyed")

        positive_patterns = sum(1 for p in self._patterns.values() if p.kind == "positive")
        negative_patterns = sum(1 for p in self._patterns.values() if p.kind == "negative")

        return {
            "total_feedback": total,
            "accept_rate": accepted / total,
            "dismiss_rate": dismissed / total,
            "annoyed_rate": annoyed / total,
            "pattern_count": len(self._patterns),
            "positive_patterns": positive_patterns,
            "negative_patterns": negative_patterns,
        }

    # DB persistence - uses new schema (id TEXT PK, timestamp, ...)

    def _save_entry_to_db(self, entry: FeedbackEntry) -> None:
        """Persist a single feedback entry to the proactive_feedback table."""
        try:
            from rune.memory.store import get_memory_store

            store = get_memory_store()
            store.save_proactive_feedback(
                feedback_id=f"fb_{uuid4().hex[:12]}",
                suggestion_type=entry.suggestion_type,
                context_summary=entry.context_summary,
                response=entry.response,
                confidence=entry.confidence,
            )
        except Exception as exc:
            log.debug("feedback_db_save_failed", error=str(exc))

    def save_to_db(self) -> None:
        """Write all in-memory feedback entries to the proactive_feedback table."""
        try:
            from rune.memory.store import get_memory_store

            store = get_memory_store()
            for entry in self._entries:
                store.save_proactive_feedback(
                    feedback_id=f"fb_{uuid4().hex[:12]}",
                    suggestion_type=entry.suggestion_type,
                    context_summary=entry.context_summary,
                    response=entry.response,
                    confidence=entry.confidence,
                )
            log.info("feedback_saved_to_db", count=len(self._entries))
        except Exception as exc:
            log.error("feedback_save_to_db_failed", error=str(exc))

    @classmethod
    def load_from_db(cls) -> FeedbackLearner:
        """Create a FeedbackLearner populated from the proactive_feedback DB table."""
        learner = cls()
        try:
            from rune.memory.store import get_memory_store

            store = get_memory_store()
            rows = store.get_proactive_feedback(limit=100)

            for row in rows:
                response_val = row.get("response", "dismissed")
                if response_val not in ("accepted", "dismissed", "ignored", "annoyed"):
                    response_val = "dismissed"

                entry = FeedbackEntry(
                    suggestion_id=row.get("id", ""),
                    suggestion_type=row.get("suggestion_type", ""),
                    context_summary=row.get("context_summary", ""),
                    response=response_val,  # type: ignore[arg-type]
                    confidence=row.get("confidence") or 0.0,
                    timestamp=datetime.fromisoformat(row["timestamp"])
                    if row.get("timestamp")
                    else datetime.now(UTC),
                )

                learner._entries.append(entry)
                if entry.suggestion_type:
                    learner._type_stats[entry.suggestion_type]["total"] += 1
                    learner._type_stats[entry.suggestion_type][entry.response] += 1
                learner._update_patterns(entry)

            log.info("feedback_loaded_from_db", count=len(learner._entries))
        except Exception as exc:
            log.error("feedback_load_from_db_failed", error=str(exc))

        return learner
