"""Proactive suggestion type definitions.

Types used by the proactive engine, engagement tracker, and pattern learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

SuggestionType = Literal["reminder", "optimization", "warning", "insight", "followup"]
SuggestionStatus = Literal["pending", "accepted", "dismissed", "expired"]


@dataclass(slots=True)
class Suggestion:
    """A proactive suggestion surfaced to the user."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    type: SuggestionType = "insight"
    title: str = ""
    description: str = ""
    confidence: float = 0.5
    source: str = ""
    status: SuggestionStatus = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None


@dataclass(slots=True)
class EngagementMetrics:
    """Aggregated metrics on how the user engages with suggestions."""

    suggestions_shown: int = 0
    suggestions_accepted: int = 0
    suggestions_dismissed: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of shown suggestions that were accepted."""
        total = self.suggestions_accepted + self.suggestions_dismissed
        if total == 0:
            return 0.0
        return self.suggestions_accepted / total
