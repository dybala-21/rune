"""Intent schema for RUNE LLM outputs.

Ported from src/llm/schemas/intent.ts - Pydantic model for parsed intents
and regex-based quick-match patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentSchema(BaseModel):
    """Structured intent parsed from a natural language command."""

    domain: Literal["file", "browser", "process", "network", "git", "conversation", "memory"] = Field(
        description="Intent domain",
    )
    action: str = Field(description="The action to perform (e.g., organize, delete, search)")
    target_type: str = Field(
        alias="targetType",
        description="Type of target (e.g., file, folder, process). Empty string if N/A.",
    )
    target_pattern: str = Field(
        alias="targetPattern",
        description="File pattern or regex (e.g., *.pdf). Empty string if N/A.",
    )
    target_location: str = Field(
        alias="targetLocation",
        description="Path or URL. Empty string if N/A.",
    )
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    ambiguous_fields: list[str] = Field(
        alias="ambiguousFields",
        description="Fields that need clarification. Empty list if none.",
    )
    clarification_needed: bool = Field(
        alias="clarificationNeeded",
        description="Whether user input is needed",
    )
    suggested_question: str = Field(
        alias="suggestedQuestion",
        description="Question to ask if clarification needed. Empty string if not.",
    )

    model_config = ConfigDict(populate_by_name=True)


# Quick-match patterns (rule-based, no LLM needed)

@dataclass(slots=True)
class QuickPattern:
    """Regex pattern mapped to a pre-defined intent stub."""
    pattern: re.Pattern[str]
    domain: str
    action: str
    target: dict[str, Any] = field(default_factory=dict)


QUICK_PATTERNS: list[QuickPattern] = [
    # --- Conversation ---
    QuickPattern(
        pattern=re.compile(r"^(hello|hi|hey|greetings)[\s!?]*$", re.IGNORECASE),
        domain="conversation", action="greet",
    ),
    QuickPattern(
        pattern=re.compile(r"^(help|what can you)[\s?]*$", re.IGNORECASE),
        domain="conversation", action="help",
    ),
    QuickPattern(
        pattern=re.compile(r"^(thanks|thank you)[\s!]*$", re.IGNORECASE),
        domain="conversation", action="thanks",
    ),
    # --- File ---
    QuickPattern(
        pattern=re.compile(r"organize.*download", re.IGNORECASE),
        domain="file", action="organize", target={"location": "~/Downloads"},
    ),
    QuickPattern(
        pattern=re.compile(r"delete.*file", re.IGNORECASE),
        domain="file", action="delete",
    ),
    QuickPattern(
        pattern=re.compile(r"find.*file|search.*file", re.IGNORECASE),
        domain="file", action="search",
    ),
    QuickPattern(
        pattern=re.compile(r"move.*file", re.IGNORECASE),
        domain="file", action="move",
    ),
    QuickPattern(
        pattern=re.compile(r"copy.*file", re.IGNORECASE),
        domain="file", action="copy",
    ),
    QuickPattern(
        pattern=re.compile(r"receipt.*organize|organize.*receipt", re.IGNORECASE),
        domain="file", action="organize",
        target={"type": "receipt", "pattern": "*.{jpg,png,pdf}"},
    ),
    # --- Git ---
    QuickPattern(
        pattern=re.compile(r"git.*status", re.IGNORECASE),
        domain="git", action="status",
    ),
    QuickPattern(
        pattern=re.compile(r"git.*commit", re.IGNORECASE),
        domain="git", action="commit",
    ),
    # --- Process ---
    QuickPattern(
        pattern=re.compile(r"kill.*process|stop.*process", re.IGNORECASE),
        domain="process", action="kill",
    ),
    QuickPattern(
        pattern=re.compile(r"list.*process", re.IGNORECASE),
        domain="process", action="list",
    ),
    # --- Memory / Repeat ---
    QuickPattern(
        pattern=re.compile(r"last time|like before|repeat|again", re.IGNORECASE),
        domain="memory", action="repeat",
    ),
]


def match_quick_pattern(user_input: str) -> QuickPattern | None:
    """Try to match *user_input* against the quick-pattern list.

    Returns the first matching ``QuickPattern`` or ``None``.
    """
    for qp in QUICK_PATTERNS:
        if qp.pattern.search(user_input):
            return qp
    return None
