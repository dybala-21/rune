"""Skill type definitions for RUNE.

Data structures for skill registration, matching, and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Skill:
    """A registered skill (loaded from SKILL.md or programmatically defined)."""

    name: str
    description: str
    body: str = ""
    scope: str = "user"  # "user" | "project"
    author: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""


@dataclass(slots=True)
class SkillMatch:
    """A skill matched against a query with a relevance score."""

    skill: Skill
    score: float = 0.0
    reason: str = ""
