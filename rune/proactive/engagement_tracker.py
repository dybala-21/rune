"""User engagement tracking for proactive suggestions.

Tracks how users interact with suggestions and determines
whether to suppress them based on low acceptance rates.

Enhanced to produce ``DetailedEngagementMetrics`` with hourly
distribution, streaks, and anti-fatigue cooldown logic.

Now supports an ``EngagementStore`` protocol for persisting
metrics, channel preferences, and conversation records via
the ``initialize(store)`` method - matching the TS daemon's
``engagementStoreAdapter`` pattern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from rune.proactive.engagement_types import (
    DEFAULT_ENGAGEMENT_POLICY,
    ChannelPreference,
    ConversationRecord,
    DetailedEngagementMetrics,
    EngagementPolicy,
    create_empty_metrics,
)
from rune.proactive.types import EngagementMetrics
from rune.utils.logger import get_logger

log = get_logger(__name__)

_SUPPRESSION_THRESHOLD = 0.1
_SUPPRESSION_MIN_SHOWN = 10
_MAX_RECORDS = 10_000


# Store protocol (mirrors TS EngagementStore interface)

@runtime_checkable
class EngagementStore(Protocol):
    """Store interface used by EngagementTracker.

    Decoupled from MemoryStore - the daemon wires a thin adapter.
    """

    def get_engagement_metrics(self, user_id: str) -> DetailedEngagementMetrics | None: ...
    def store_engagement_metrics(self, user_id: str, data: DetailedEngagementMetrics) -> None: ...
    def get_channel_preferences(self, user_id: str) -> ChannelPreference | None: ...
    def store_channel_preferences(self, user_id: str, data: ChannelPreference) -> None: ...
    def store_conversation_record(self, record: ConversationRecord) -> None: ...


# Internal record

@dataclass(slots=True)
class _SuggestionRecord:
    """Internal record for a single suggestion interaction."""
    shown_at: datetime = field(default_factory=datetime.now)
    accepted: bool | None = None  # None = still pending
    resolved_at: datetime | None = None


# Serialization helpers

def _metrics_to_dict(m: DetailedEngagementMetrics) -> dict[str, Any]:
    """Convert metrics to a JSON-safe dict."""
    d = asdict(m)
    # Convert datetimes to ISO strings
    for key in ("last_response_at", "last_suggestion_at", "cooldown_until"):
        val = d.get(key)
        if isinstance(val, datetime):
            d[key] = val.isoformat()
    return d


def _metrics_from_dict(d: dict[str, Any]) -> DetailedEngagementMetrics:
    """Reconstruct metrics from a dict (reviving date fields)."""
    for key in ("last_response_at", "last_suggestion_at", "cooldown_until"):
        val = d.get(key)
        if isinstance(val, str) and val:
            try:
                d[key] = datetime.fromisoformat(val)
            except ValueError:
                d[key] = None
        elif not isinstance(val, datetime):
            d[key] = None
    # Ensure dict fields exist
    for key in (
        "active_hour_distribution", "preferred_channels",
        "channel_conversation_totals", "channel_conversation_hourly",
    ):
        if key not in d:
            d[key] = {}
    return DetailedEngagementMetrics(**{
        k: v for k, v in d.items()
        if k in DetailedEngagementMetrics.__dataclass_fields__
    })


def _channel_pref_to_dict(p: ChannelPreference) -> dict[str, Any]:
    d = asdict(p)
    if isinstance(d.get("last_updated"), datetime):
        d["last_updated"] = d["last_updated"].isoformat()
    return d


def _channel_pref_from_dict(d: dict[str, Any]) -> ChannelPreference:
    val = d.get("last_updated")
    if isinstance(val, str) and val:
        try:
            d["last_updated"] = datetime.fromisoformat(val)
        except ValueError:
            d["last_updated"] = datetime.now()
    return ChannelPreference(**{
        k: v for k, v in d.items()
        if k in ChannelPreference.__dataclass_fields__
    })


# EngagementTracker

class EngagementTracker:
    """Tracks user engagement with proactive suggestions.

    Returns both simple ``EngagementMetrics`` (backward compatible) and
    ``DetailedEngagementMetrics`` with hourly distribution, streaks,
    and anti-fatigue cooldown.

    Call ``initialize(store)`` to connect a persistence backend matching
    the ``EngagementStore`` protocol.
    """

    __slots__ = (
        "_records",
        "_detailed",
        "_policy",
        "_consecutive_ignores",
        "_cooldown_until",
        "_store",
        "_initialized",
    )

    def __init__(
        self,
        policy: EngagementPolicy | None = None,
        user_id: str = "local_user",
    ) -> None:
        self._records: dict[str, _SuggestionRecord] = {}
        self._detailed: DetailedEngagementMetrics = create_empty_metrics(user_id)
        self._policy = policy or DEFAULT_ENGAGEMENT_POLICY
        self._consecutive_ignores: int = 0
        self._cooldown_until: datetime | None = None
        self._store: EngagementStore | None = None
        self._initialized: bool = False

    # -- Store wiring (matches TS tracker.initialize()) ---------------------

    def initialize(self, store: EngagementStore) -> None:
        """Connect a persistence store and mark as initialized.

        Mirrors TS ``EngagementTracker.initialize(engagementStoreAdapter)``.
        """
        self._store = store
        self._initialized = True
        log.info("engagement_tracker_initialized")

    def is_initialized(self) -> bool:
        """Return whether the tracker has been wired to a store."""
        return self._initialized

    # -- Persistence helpers ------------------------------------------------

    def _load_metrics(self, user_id: str) -> DetailedEngagementMetrics:
        """Load metrics from the store (cache-first)."""
        if self._store is not None:
            stored = self._store.get_engagement_metrics(user_id)
            if stored is not None:
                return stored
        return create_empty_metrics(user_id)

    def _save_metrics(self, user_id: str, metrics: DetailedEngagementMetrics) -> None:
        """Persist metrics to the store if available."""
        if self._store is not None:
            self._store.store_engagement_metrics(user_id, metrics)

    def _save_conversation_record(self, record: ConversationRecord) -> None:
        """Persist a conversation record if the store is available."""
        if self._store is not None:
            self._store.store_conversation_record(record)

    # -- Recording events ---------------------------------------------------

    def record_shown(self, suggestion_id: str) -> None:
        """Record that a suggestion was shown to the user."""
        now = datetime.now()
        self._records[suggestion_id] = _SuggestionRecord(shown_at=now)

        # Update detailed metrics
        self._detailed.total_suggestions += 1
        self._detailed.last_suggestion_at = now
        hour = now.hour
        self._detailed.active_hour_distribution[hour] = (
            self._detailed.active_hour_distribution.get(hour, 0) + 1
        )

        # Daily count tracking
        today = now.strftime("%Y-%m-%d")
        if self._detailed.daily_suggestion_date != today:
            self._detailed.daily_suggestion_count = 0
            self._detailed.daily_suggestion_date = today
        self._detailed.daily_suggestion_count += 1

        # Persist
        self._save_metrics(self._detailed.user_id, self._detailed)

        # Evict oldest half when exceeding max records
        if len(self._records) > _MAX_RECORDS:
            keys = list(self._records.keys())
            evict_count = len(keys) // 2
            for k in keys[:evict_count]:
                del self._records[k]
            log.info("engagement_records_evicted", evicted=evict_count, remaining=len(self._records))

        log.debug("engagement_shown", suggestion_id=suggestion_id)

    def record_accepted(self, suggestion_id: str) -> None:
        """Record that a suggestion was accepted."""
        now = datetime.now()
        rec = self._records.get(suggestion_id)
        if rec is None:
            rec = _SuggestionRecord(shown_at=now)
            self._records[suggestion_id] = rec
        rec.accepted = True
        rec.resolved_at = now

        # Update detailed metrics
        self._detailed.total_responses += 1
        self._detailed.last_response_at = now
        self._consecutive_ignores = 0
        self._detailed.consecutive_ignores = 0

        # Update streak
        today = now.strftime("%Y-%m-%d")
        if self._detailed.last_active_date != today:
            self._detailed.streak += 1
        self._detailed.last_active_date = today

        self._recalculate_rates()
        self._save_metrics(self._detailed.user_id, self._detailed)
        log.debug("engagement_accepted", suggestion_id=suggestion_id)

    def record_dismissed(self, suggestion_id: str) -> None:
        """Record that a suggestion was dismissed."""
        now = datetime.now()
        rec = self._records.get(suggestion_id)
        if rec is None:
            rec = _SuggestionRecord(shown_at=now)
            self._records[suggestion_id] = rec
        rec.accepted = False
        rec.resolved_at = now

        # Update detailed metrics
        self._detailed.total_responses += 1
        self._detailed.last_response_at = now
        self._consecutive_ignores += 1
        self._detailed.consecutive_ignores = self._consecutive_ignores

        # Anti-fatigue: trigger cooldown after consecutive ignores
        if self._consecutive_ignores >= self._policy.consecutive_ignore_threshold:
            cooldown_minutes = self._policy.cooldown_after_consecutive_ignores
            self._cooldown_until = now + timedelta(minutes=cooldown_minutes)
            self._detailed.cooldown_until = self._cooldown_until
            log.info(
                "engagement_cooldown_triggered",
                consecutive_ignores=self._consecutive_ignores,
                cooldown_minutes=cooldown_minutes,
            )

        self._recalculate_rates()
        self._save_metrics(self._detailed.user_id, self._detailed)
        log.debug("engagement_dismissed", suggestion_id=suggestion_id)

    # -- Queries ------------------------------------------------------------

    def get_metrics(self) -> EngagementMetrics:
        """Compute current simple engagement metrics (backward compatible)."""
        shown = len(self._records)
        accepted = sum(1 for r in self._records.values() if r.accepted is True)
        dismissed = sum(1 for r in self._records.values() if r.accepted is False)
        return EngagementMetrics(
            suggestions_shown=shown,
            suggestions_accepted=accepted,
            suggestions_dismissed=dismissed,
        )

    def get_detailed_metrics(self) -> DetailedEngagementMetrics:
        """Return the full ``DetailedEngagementMetrics`` snapshot."""
        self._recalculate_rates()
        return self._detailed

    def is_in_cooldown(self) -> bool:
        """Return True if currently in an anti-fatigue cooldown period."""
        if self._cooldown_until is None:
            return False
        if datetime.now() >= self._cooldown_until:
            self._cooldown_until = None
            self._detailed.cooldown_until = None
            return False
        return True

    def should_suppress(self) -> bool:
        """Return True if suggestions should be suppressed.

        Suppresses when:
        - acceptance rate drops below 10% after at least 10 resolved, OR
        - currently in anti-fatigue cooldown
        """
        if self.is_in_cooldown():
            return True

        metrics = self.get_metrics()
        resolved = metrics.suggestions_accepted + metrics.suggestions_dismissed
        if resolved < _SUPPRESSION_MIN_SHOWN:
            return False
        return metrics.acceptance_rate < _SUPPRESSION_THRESHOLD

    def _recalculate_rates(self) -> None:
        """Recalculate response and accept rates on the detailed metrics."""
        d = self._detailed
        if d.total_suggestions > 0:
            d.response_rate = d.total_responses / d.total_suggestions
        accepted = sum(1 for r in self._records.values() if r.accepted is True)
        if d.total_responses > 0:
            d.accept_rate = accepted / d.total_responses
        # Update engagement score: weighted combination
        d.engagement_score = min(1.0, max(0.0, d.accept_rate * 0.6 + d.response_rate * 0.4))
