"""Behavioral pattern learning for RUNE.

Observes user activity patterns (time-of-day, day-of-week, action sequences)
and predicts likely current/next activities.

Persistence: sequence transition counts are saved to / loaded from the
``sequence_patterns`` table in the memory store DB.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Time slot helpers

_TIME_SLOTS = {
    "morning": range(6, 12),
    "afternoon": range(12, 17),
    "evening": range(17, 22),
    "night": list(range(22, 24)) + list(range(0, 6)),
}


def get_current_time_slot() -> str:
    """Return the current time slot: morning/afternoon/evening/night."""
    hour = datetime.now().hour
    for slot, hours in _TIME_SLOTS.items():
        if hour in hours:
            return slot
    return "night"


def get_current_day_type() -> str:
    """Return 'weekday' or 'weekend'."""
    return "weekend" if datetime.now().weekday() >= 5 else "weekday"


# Activity record

@dataclass(slots=True)
class _ActivityRecord:
    activity_type: str
    metadata: dict[str, Any]
    time_slot: str
    day_type: str
    timestamp: datetime


# PatternLearner

_MAX_HISTORY = 1000


class PatternLearner:
    """Learns behavioral patterns from recorded activities."""

    __slots__ = ("_history", "_sequence_counts", "_slot_counts")

    def __init__(self) -> None:
        self._history: list[_ActivityRecord] = []
        # Counts of (prev_activity → next_activity) transitions
        self._sequence_counts: dict[str, Counter[str]] = defaultdict(Counter)
        # Counts of activity per (time_slot, day_type)
        self._slot_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def record_activity(self, activity_type: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a user activity for pattern learning."""
        time_slot = get_current_time_slot()
        day_type = get_current_day_type()

        record = _ActivityRecord(
            activity_type=activity_type,
            metadata=metadata or {},
            time_slot=time_slot,
            day_type=day_type,
            timestamp=datetime.now(),
        )

        # Update transition counts
        if self._history:
            prev = self._history[-1].activity_type
            self._sequence_counts[prev][activity_type] += 1

        # Update slot counts
        self._slot_counts[(time_slot, day_type)][activity_type] += 1

        self._history.append(record)

        # Trim history
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        log.debug(
            "pattern_recorded",
            activity=activity_type,
            slot=time_slot,
            day=day_type,
        )

    def predict_current_activity(self) -> dict[str, Any]:
        """Predict the most likely current activity based on time patterns.

        Returns a dict with keys: likely_activity, confidence, suggested_commands.
        """
        time_slot = get_current_time_slot()
        day_type = get_current_day_type()

        counts = self._slot_counts.get((time_slot, day_type))
        if not counts:
            return {
                "likely_activity": None,
                "confidence": 0.0,
                "suggested_commands": [],
            }

        total = sum(counts.values())
        most_common = counts.most_common(1)[0]
        activity, count = most_common
        confidence = count / total if total > 0 else 0.0

        # Gather associated metadata for suggested commands
        suggested: list[str] = []
        for rec in reversed(self._history):
            if rec.activity_type == activity and rec.metadata.get("command"):
                cmd = rec.metadata["command"]
                if cmd not in suggested:
                    suggested.append(cmd)
                if len(suggested) >= 3:
                    break

        return {
            "likely_activity": activity,
            "confidence": round(confidence, 3),
            "suggested_commands": suggested,
        }

    def predict_next_activity(self, current: str) -> dict[str, Any]:
        """Predict the next activity given the current one.

        Returns a dict with keys: likely_activity, confidence, suggested_commands.
        """
        transitions = self._sequence_counts.get(current)
        if not transitions:
            return {
                "likely_activity": None,
                "confidence": 0.0,
                "suggested_commands": [],
            }

        total = sum(transitions.values())
        most_common = transitions.most_common(1)[0]
        activity, count = most_common
        confidence = count / total if total > 0 else 0.0

        # Suggested commands from history
        suggested: list[str] = []
        for rec in reversed(self._history):
            if rec.activity_type == activity and rec.metadata.get("command"):
                cmd = rec.metadata["command"]
                if cmd not in suggested:
                    suggested.append(cmd)
                if len(suggested) >= 3:
                    break

        return {
            "likely_activity": activity,
            "confidence": round(confidence, 3),
            "suggested_commands": suggested,
        }

    # DB persistence

    def save_to_db(self) -> None:
        """Persist sequence transition counts to the sequence_patterns table."""
        try:
            from rune.memory.store import get_memory_store

            store = get_memory_store()

            for prev_activity, counter in self._sequence_counts.items():
                for next_activity, freq in counter.items():
                    store.save_sequence_pattern(
                        prev_activity,
                        next_activity,
                        count=freq,
                    )

            log.info("patterns_saved_to_db", transitions=len(self._sequence_counts))
        except Exception as exc:
            log.error("patterns_save_to_db_failed", error=str(exc))

    @classmethod
    def load_from_db(cls) -> PatternLearner:
        """Create a PatternLearner populated from the sequence_patterns DB table."""
        learner = cls()
        try:
            from rune.memory.store import get_memory_store

            store = get_memory_store()
            patterns = store.get_sequence_patterns()
            count = 0
            for p in patterns:
                from_act = p["from_activity"]
                to_act = p["to_activity"]
                if from_act and to_act:
                    learner._sequence_counts[from_act][to_act] = p["count"]
                    count += 1

            log.info("patterns_loaded_from_db", transitions=count)
        except Exception as exc:
            log.error("patterns_load_from_db_failed", error=str(exc))

        return learner
