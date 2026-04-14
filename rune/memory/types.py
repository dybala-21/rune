"""Shared data types for the RUNE memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Episode:
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    timestamp: str = ""
    task_summary: str = ""
    intent: str = ""
    plan: str = ""
    result: str = ""
    lessons: str = ""
    embedding: bytes | None = None
    conversation_id: str = ""
    importance: float = 0.5
    entities: str = ""
    files_touched: str = ""
    commitments: str = ""
    duration_ms: float = 0.0
    utility: int = 0
    # Tier 2 advisor counters (Phase 2a). All default to 0 so existing
    # Episode construction paths remain backward compatible.
    advisor_calls: int = 0
    advisor_followed_count: int = 0
    advisor_output_tokens: int = 0


@dataclass(slots=True)
class Fact:
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    category: str = ""
    key: str = ""
    value: str = ""
    source: str = ""
    confidence: float = 1.0
    last_verified: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class SafetyRule:
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    type: str = ""
    pattern: str = ""
    reason: str = ""
    source: str = ""
    created_at: str = ""


@dataclass(slots=True)
class VectorMetadata:
    type: str = ""
    id: str = ""
    timestamp: str = ""
    summary: str = ""
    category: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    id: str
    score: float
    metadata: VectorMetadata
    text: str = ""


@dataclass(slots=True)
class WorkingMemory:
    """In-memory fast-access store for current session context."""
    facts: dict[str, str] = field(default_factory=dict)
    safety_rules: list[dict[str, str]] = field(default_factory=list)
    recent_commands: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
