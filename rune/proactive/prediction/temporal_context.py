"""Time-slot based acceptance tracking for RUNE.

Tracks suggestion acceptance rates per time slot (morning, afternoon,
evening, night) to determine the best times for proactive engagement.
"""

from __future__ import annotations

from dataclasses import dataclass

from rune.proactive.patterns import get_current_time_slot
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Minimum acceptance rate to consider a time slot "good"
_GOOD_TIME_THRESHOLD = 0.3

# Minimum samples before using a slot's rate
_MIN_SAMPLES = 3


@dataclass(slots=True)
class TimeSlotStats:
    """Acceptance statistics for a single time slot."""

    slot: str
    total: int
    accepted: int
    rate: float


class TemporalContextAnalyzer:
    """Tracks and analyses suggestion acceptance rates by time slot.

    Records outcomes per time slot and identifies the best times
    for surfacing proactive suggestions.
    """

    __slots__ = ("_slot_data",)

    def __init__(self) -> None:
        # slot_name → {"total": int, "accepted": int}
        self._slot_data: dict[str, dict[str, int]] = {}

    def record(self, slot: str, accepted: bool) -> None:
        """Record a suggestion outcome for a time slot.

        Parameters:
            slot: The time slot name (morning, afternoon, evening, night).
            accepted: Whether the suggestion was accepted.
        """
        if slot not in self._slot_data:
            self._slot_data[slot] = {"total": 0, "accepted": 0}

        self._slot_data[slot]["total"] += 1
        if accepted:
            self._slot_data[slot]["accepted"] += 1

        log.debug(
            "temporal_recorded",
            slot=slot,
            accepted=accepted,
            total=self._slot_data[slot]["total"],
        )

    def get_best_slots(self) -> list[TimeSlotStats]:
        """Return time slots sorted by acceptance rate (best first).

        Only includes slots with enough samples.

        Returns:
            List of TimeSlotStats, sorted by rate descending.
        """
        stats: list[TimeSlotStats] = []

        for slot, data in self._slot_data.items():
            total = data["total"]
            if total < _MIN_SAMPLES:
                continue

            accepted = data["accepted"]
            rate = accepted / total

            stats.append(TimeSlotStats(
                slot=slot,
                total=total,
                accepted=accepted,
                rate=round(rate, 4),
            ))

        stats.sort(key=lambda s: s.rate, reverse=True)
        return stats

    def is_good_time(self) -> bool:
        """Check if the current time slot has a favourable acceptance rate.

        Returns True if the current slot has an acceptance rate above
        the threshold, or if there is insufficient data (benefit of doubt).
        """
        current_slot = get_current_time_slot()
        data = self._slot_data.get(current_slot)

        if data is None or data["total"] < _MIN_SAMPLES:
            # Not enough data - give benefit of the doubt
            return True

        rate = data["accepted"] / data["total"]
        return rate >= _GOOD_TIME_THRESHOLD
