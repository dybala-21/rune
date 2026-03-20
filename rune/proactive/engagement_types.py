"""Proactive engagement type definitions.

Ported from src/proactive/engagement-types.ts. Conversation categories,
engagement metrics, policy, channel delivery, and configuration types.

References:
- CHI 2025 "Inner Thoughts" - 5-stage proactive pipeline
- ACM TOIS 2025 Survey - Intelligence + Adaptivity + Civility
- Engagement research - 1-3 daily targets = optimal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

# Conversation categories

class ConversationCategory(StrEnum):
    """Types of conversations the AI can initiate."""

    CASUAL_CHECKIN = "casual_checkin"
    """'How's it going? You've been at it a while.'"""

    PROGRESS_UPDATE = "progress_update"
    """'You closed 3 issues today, nice pace.'"""

    LEARNING_SHARE = "learning_share"
    """'Found something related to what you were researching.'"""

    REMINDER = "reminder"
    """'You said you'd review that PR today.'"""

    INSIGHT = "insight"
    """'I noticed you usually run tests before pushing.'"""

    QUESTION = "question"
    """'Want me to set up monitoring?'"""

    ENCOURAGEMENT = "encouragement"
    """'That commit was solid - caught a nice edge case.'"""


# Trigger types

TriggerType = Literal["time", "pattern", "milestone", "memory", "idle", "schedule"]
Priority = Literal["low", "medium", "high"]
ResponseType = Literal["accepted", "dismissed", "ignored", "annoyed"]


# Conversation opportunity

@dataclass(slots=True)
class ConversationOpportunity:
    """A proactive conversation the ConversationInitiator wants to fire."""

    id: str
    category: ConversationCategory
    trigger: TriggerType
    priority: Priority
    topic: str
    draft_message: str
    context_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    expires_at: datetime = field(default_factory=datetime.now)
    suppression_key: str = ""
    """Dedup key ``"category:topic_hash"`` - suppress within 24 h."""


# Engagement metrics (persisted to SQLite)

@dataclass(slots=True)
class DetailedEngagementMetrics:
    """Per-user engagement metrics, ported from the full TS interface."""

    user_id: str = "local_user"

    # Response tracking
    total_suggestions: int = 0
    total_responses: int = 0
    response_rate: float = 0.0
    accept_rate: float = 0.0
    last_response_at: datetime | None = None
    last_suggestion_at: datetime | None = None

    # Activity patterns
    active_hour_distribution: dict[int, int] = field(default_factory=dict)
    preferred_channels: dict[str, int] = field(default_factory=dict)
    channel_conversation_totals: dict[str, int] = field(default_factory=dict)
    channel_conversation_hourly: dict[str, dict[str, int]] = field(default_factory=dict)
    average_response_time_ms: float = 0.0

    # Engagement health
    engagement_score: float = 0.5  # neutral start
    streak: int = 0
    last_active_date: str = ""

    # Anti-fatigue
    daily_suggestion_count: int = 0
    daily_suggestion_date: str = ""
    consecutive_ignores: int = 0
    cooldown_until: datetime | None = None


def create_empty_metrics(user_id: str = "local_user") -> DetailedEngagementMetrics:
    """Create a fresh empty metrics record (mirrors TS createEmptyMetrics)."""
    today = datetime.now().strftime("%Y-%m-%d")
    return DetailedEngagementMetrics(
        user_id=user_id,
        last_active_date=today,
        daily_suggestion_date=today,
    )


# Engagement policy

@dataclass(slots=True)
class EngagementPolicy:
    """Dynamic frequency control policy."""

    base_max_per_day: int = 3
    min_max_per_day: int = 1
    max_max_per_day: int = 5
    frequency_multiplier: float = 1.0
    cooldown_after_annoyed: int = 120  # minutes
    cooldown_after_consecutive_ignores: int = 60  # minutes
    consecutive_ignore_threshold: int = 3
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    respect_system_quiet_hours: bool = True


DEFAULT_ENGAGEMENT_POLICY = EngagementPolicy()


# Channel delivery

@dataclass(slots=True)
class ChannelPreference:
    """Learned channel preferences for a user."""

    user_id: str = ""
    time_slot_preferences: dict[str, str] = field(default_factory=dict)
    explicit_preferences: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class DeliveryDecision:
    """Where and how to deliver a proactive message."""

    channel: str = ""
    recipient_id: str = ""
    formatted_message: str = ""
    priority: Priority = "medium"
    fallback_channels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DeliveryResult:
    """Outcome of a delivery attempt."""

    channel: str = ""
    delivered: bool = False
    error: str | None = None


# Conversation record

@dataclass(slots=True)
class ConversationRecord:
    """Historical record of a proactive conversation attempt."""

    id: str = ""
    user_id: str = ""
    category: ConversationCategory = ConversationCategory.CASUAL_CHECKIN
    topic: str = ""
    message: str = ""
    channel: str | None = None
    response: ResponseType | None = None
    response_time_ms: float | None = None
    created_at: datetime = field(default_factory=datetime.now)


# Engagement config

@dataclass(slots=True)
class EngagementConfig:
    """System-level engagement configuration."""

    enabled: bool = True
    max_suggestions_per_day: int = 3
    min_interval_minutes: int = 30
    personality: Literal["neutral", "warm", "concise", "enthusiastic"] = "neutral"
    language: Literal["en", "ko", "auto"] = "auto"
    categories: dict[str, bool] = field(default_factory=lambda: {
        c.value: True for c in ConversationCategory
    })
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    channel_preferences: dict[str, str] = field(default_factory=dict)
