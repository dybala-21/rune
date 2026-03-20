"""Natural proactive message generation for RUNE.

Composes human-friendly messages for proactive suggestions
with support for Korean and English languages.
"""

from __future__ import annotations

from rune.proactive.types import Suggestion, SuggestionType
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Korean templates per suggestion type
_KO_TEMPLATES: dict[SuggestionType, str] = {
    "reminder": "[리마인더] {title}\n{description}",
    "optimization": "[최적화 제안] {title}\n{description}",
    "warning": "[주의] {title}\n{description}",
    "insight": "[인사이트] {title}\n{description}",
    "followup": "[후속 작업] {title}\n{description}",
}

# English templates per suggestion type
_EN_TEMPLATES: dict[SuggestionType, str] = {
    "reminder": "[Reminder] {title}\n{description}",
    "optimization": "[Optimization] {title}\n{description}",
    "warning": "[Warning] {title}\n{description}",
    "insight": "[Insight] {title}\n{description}",
    "followup": "[Follow-up] {title}\n{description}",
}

# Prefix icons per type
_TYPE_ICONS: dict[SuggestionType, str] = {
    "reminder": ">>",
    "optimization": ">>",
    "warning": "!!",
    "insight": "**",
    "followup": "->",
}


def compose_message(suggestion: Suggestion, user_language: str = "ko") -> str:
    """Compose a natural-language message for a suggestion.

    Parameters:
        suggestion: The suggestion to compose a message for.
        user_language: Target language ("ko" for Korean, "en" or anything else for English).

    Returns:
        A formatted string message.
    """
    if user_language == "ko":
        return _compose_korean(suggestion)
    return _compose_english(suggestion)


def _compose_korean(suggestion: Suggestion) -> str:
    """Compose a Korean-language message for a suggestion."""
    template = _KO_TEMPLATES.get(suggestion.type, _KO_TEMPLATES["insight"])
    message = template.format(
        title=suggestion.title or "알림",
        description=suggestion.description or "",
    )

    # Add confidence indicator for high-confidence suggestions
    if suggestion.confidence >= 0.8:
        message += "\n(높은 확신도)"
    elif suggestion.confidence <= 0.3:
        message += "\n(참고 사항)"

    return message.strip()


def _compose_english(suggestion: Suggestion) -> str:
    """Compose an English-language message for a suggestion."""
    template = _EN_TEMPLATES.get(suggestion.type, _EN_TEMPLATES["insight"])
    message = template.format(
        title=suggestion.title or "Notice",
        description=suggestion.description or "",
    )

    # Add confidence indicator for high-confidence suggestions
    if suggestion.confidence >= 0.8:
        message += "\n(high confidence)"
    elif suggestion.confidence <= 0.3:
        message += "\n(for your reference)"

    return message.strip()
