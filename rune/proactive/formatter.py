"""Proactive suggestion formatter for RUNE.

Converts raw Suggestion objects into conversational text.
No buttons, no key hints - RUNE talks, user responds naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

from rune.proactive.types import Suggestion

# Formatted output

@dataclass(slots=True)
class FormattedSuggestion:
    """Channel-ready suggestion as conversational text."""

    suggestion_id: str = ""
    body: str = ""           # Full conversational text (question form)
    raw_description: str = ""  # Original description for context injection
    confidence: float = 0.0
    source: str = ""
    intensity: str = "nudge"  # nudge | suggest | intervene


# Intensity classification

def _classify_intensity(confidence: float) -> str:
    if confidence >= 0.8:
        return "intervene"
    if confidence >= 0.5:
        return "suggest"
    return "nudge"


# Conversational text generation

# Maps suggestion type to a conversational question.
# The body already contains the detail - we just need the right ending.
_QUESTION_SUFFIX = {
    "warning": "도와드릴까요?",
    "optimization": "해볼까요?",
    "reminder": "",  # reminders are informational, no question needed
    "followup": "이어서 할까요?",
    "insight": "",  # insights are informational
}


def format_suggestion(suggestion: Suggestion) -> FormattedSuggestion:
    """Convert a raw Suggestion into conversational text."""
    intensity = _classify_intensity(suggestion.confidence)
    description = suggestion.description or suggestion.title

    # Build conversational body
    suffix = _QUESTION_SUFFIX.get(suggestion.type, "")
    if suffix and not description.rstrip().endswith("?") and not description.rstrip().endswith("까요?"):
        body = f"{description} {suffix}"
    else:
        body = description

    return FormattedSuggestion(
        suggestion_id=suggestion.id,
        body=body,
        raw_description=description,
        confidence=suggestion.confidence,
        source=suggestion.source,
        intensity=intensity,
    )


def format_for_channel(suggestion: Suggestion, channel: str) -> str:
    """Format a suggestion as plain text for external channels."""
    fmt = format_suggestion(suggestion)
    return f"🤖 RUNE: {fmt.body}"
