"""Define proactive suggestions and engagement metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
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
    response_source: Literal["user"] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    # A passing check confirms execution, not user approval.
    verification: list[str] = field(default_factory=list)
    execution_status: str | None = None
    execution_result: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=UTC)
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            self.expires_at = self.expires_at.replace(tzinfo=UTC)


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
