"""Conversation type definitions for RUNE.

Data structures representing conversation turns, full conversations,
and their metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    channel: str = ""
    episode_id: str = ""
    execution_context: str = ""
    goal_type: str = ""  # GoalType of this turn's goal (for domain change detection)
    archived: bool = False


@dataclass(slots=True)
class Conversation:
    """A full conversation with metadata."""

    id: str = field(default_factory=lambda: uuid4().hex[:16])
    user_id: str = ""
    title: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    digest: str = ""
    status: str = "active"  # "active" | "archived"
    execution_context: str = ""
