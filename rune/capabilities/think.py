"""Think capability - internal reasoning tool.

Ported from src/capabilities/think.ts - allows the agent to reason
without taking external action.  Includes session-scoped thought
history tracking (max 100 entries, FIFO eviction).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel

_MAX_HISTORY = 100


class ThinkParams(BaseModel):
    thought: str = Field(description="The agent's internal reasoning")
    category: str = Field(default="general", description="Category of thought: general, planning, debugging, analysis")
    scratchpad: str = Field(default="", description="Scratchpad space for working notes")


# Thought history (module-level, session-scoped)

_thought_history: list[dict] = []


async def think(params: ThinkParams) -> CapabilityResult:
    """Record an internal reasoning step (no side effects)."""
    entry = {
        "thought": params.thought,
        "category": params.category,
        "timestamp": datetime.now(UTC).isoformat(),
        "scratchpad": params.scratchpad,
    }

    _thought_history.append(entry)

    # FIFO eviction: keep only the most recent entries
    if len(_thought_history) > _MAX_HISTORY:
        _thought_history.pop(0)

    return CapabilityResult(
        success=True,
        output=f"Thought recorded: {params.thought[:200]}",
        metadata={
            "thought_length": len(params.thought),
            "category": params.category,
            "has_scratchpad": bool(params.scratchpad),
            "history_length": len(_thought_history),
        },
    )


# History accessors


def get_thought_history() -> list[dict]:
    """Return a copy of all recorded thoughts."""
    return list(_thought_history)


def clear_thought_history() -> None:
    """Reset the thought history."""
    _thought_history.clear()


def summarize_thoughts() -> str:
    """Create a concise summary of the thinking process, grouped by category."""
    if not _thought_history:
        return "No thoughts recorded."

    by_category: dict[str, list[str]] = defaultdict(list)
    for entry in _thought_history:
        by_category[entry["category"]].append(entry["thought"])

    sections: list[str] = []
    for category, thoughts in by_category.items():
        header = f"[{category.upper()}] ({len(thoughts)} thoughts)"
        # List key points (first line / truncated)
        points = "\n".join(f"  - {t[:120]}" for t in thoughts)
        sections.append(f"{header}\n{points}")

    return "\n\n".join(sections)


def get_thought_count() -> int:
    """Return the number of recorded thoughts."""
    return len(_thought_history)


# Registration


def register_think_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="think",
        description="Internal reasoning — think through a problem",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=ThinkParams,
        execute=think,
    ))
