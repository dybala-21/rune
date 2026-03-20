"""Proactive conversation initiation for RUNE.

Determines when and how the agent should proactively start a
conversation (greetings, follow-ups, reminders, insights).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from rune.proactive.context import AwarenessContext
from rune.proactive.types import EngagementMetrics, Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)

InitiativeType = Literal["greeting", "followup", "reminder", "insight"]

# Greeting is appropriate between these hours
_GREETING_HOURS = range(6, 22)

# Minimum acceptance rate before the agent proactively initiates
_MIN_RECEPTIVITY = 0.2

# Minimum shown before using acceptance rate for decisions
_MIN_SHOWN_FOR_RATE = 5


class ConversationInitiator:
    """Decides when and how the agent should proactively start conversations."""

    __slots__ = ("_last_greeting_date", "_last_followup_at")

    def __init__(self) -> None:
        self._last_greeting_date: str = ""
        self._last_followup_at: datetime | None = None

    def should_initiate(
        self,
        context: AwarenessContext,
        engagement: EngagementMetrics,
    ) -> bool:
        """Determine whether the agent should proactively initiate a conversation.

        Parameters:
            context: Current awareness context.
            engagement: User engagement metrics.

        Returns:
            True if conditions are favourable for initiation.
        """
        # Respect user receptivity
        if engagement.suggestions_shown >= _MIN_SHOWN_FOR_RATE:
            if engagement.acceptance_rate < _MIN_RECEPTIVITY:
                return False

        # Check if there is a reason to initiate
        if self._check_greeting_time(context):
            return True

        return bool(self._check_followup_needed(context))

    def create_initiative(
        self,
        initiative_type: InitiativeType,
        context: AwarenessContext,
    ) -> Suggestion:
        """Create a proactive suggestion for the given initiative type.

        Parameters:
            initiative_type: The type of initiative to create.
            context: Current awareness context.

        Returns:
            A Suggestion representing the proactive initiative.
        """
        now = datetime.now()

        if initiative_type == "greeting":
            self._last_greeting_date = now.strftime("%Y-%m-%d")
            hour = now.hour
            if hour < 12:
                greeting = "Good morning"
            elif hour < 17:
                greeting = "Good afternoon"
            else:
                greeting = "Good evening"

            return Suggestion(
                type="insight",
                title=greeting,
                description=f"{greeting}! Ready to help you with today's work.",
                confidence=0.6,
                source="conversation_initiator",
            )

        if initiative_type == "followup":
            self._last_followup_at = now
            return Suggestion(
                type="followup",
                title="Follow-up check",
                description="How did the previous task go? Need any adjustments?",
                confidence=0.5,
                source="conversation_initiator",
            )

        if initiative_type == "reminder":
            return Suggestion(
                type="reminder",
                title="Pending items",
                description="You have uncommitted changes. Would you like to review them?",
                confidence=0.6,
                source="conversation_initiator",
            )

        # insight
        return Suggestion(
            type="insight",
            title="Workspace observation",
            description="I noticed some patterns in your workflow that might be worth discussing.",
            confidence=0.4,
            source="conversation_initiator",
        )

    def _check_greeting_time(self, context: AwarenessContext) -> bool:
        """Check if it is an appropriate time for a greeting.

        Returns True if:
        - Current hour is within greeting hours
        - No greeting has been sent today
        """
        hour = context.time_context.get("hour", 12)
        if hour not in _GREETING_HOURS:
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        return self._last_greeting_date != today

    def _check_followup_needed(self, context: AwarenessContext) -> bool:
        """Check if a follow-up conversation is warranted.

        Returns True if there are uncommitted git changes and no
        recent follow-up has been sent.
        """
        if not context.git_status:
            return False

        if self._last_followup_at is not None:
            elapsed = (datetime.now() - self._last_followup_at).total_seconds()
            if elapsed < 1800:  # 30 minutes cooldown
                return False

        return True
