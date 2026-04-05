"""Shared types and protocols for the rehydration subsystem.

All modules in ``rune/agent/rehydration/`` import from here.
External callers should use the ``__init__.py`` re-exports instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


# Loop state view — read-only facade presented to signals
@runtime_checkable
class LoopStateView(Protocol):
    """Read-only view of the agent loop state that signals can inspect.

    Signals receive this instead of the full loop to prevent coupling.
    """

    @property
    def step(self) -> int: ...

    @property
    def activity_phase(self) -> str: ...

    @property
    def phase_just_changed(self) -> bool: ...

    @property
    def stall_consecutive(self) -> int: ...

    @property
    def gate_blocked_count(self) -> int: ...

    @property
    def token_budget_fraction(self) -> float: ...

    def recent_tool_names(self, n: int) -> list[str]: ...

    def goal(self) -> str: ...


# Signal reading — returned by each signal on evaluation
@dataclass(frozen=True, slots=True)
class SignalReading:
    """A signal's decision for the current step."""

    name: str
    fired: bool
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Rehydration decision — returned by the trigger
@dataclass(frozen=True, slots=True)
class RehydrationDecision:
    fired: bool
    reading: SignalReading | None = None
    reason: str = ""

    @classmethod
    def fire(cls, reading: SignalReading) -> RehydrationDecision:
        return cls(fired=True, reading=reading, reason=reading.name)

    @classmethod
    def skip(cls, reason: str) -> RehydrationDecision:
        return cls(fired=False, reason=reason)


# Compacted record — one per message saved before deletion
_MAX_PENDING = 500  # soft cap on recorder flush queue


@dataclass(slots=True)
class CompactedRecord:
    step: int
    activity_phase: str
    role: str
    tool_name: str
    original_content: str
    compaction_event: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_markdown(self) -> str:
        """Render as an append-only markdown entry."""
        header = (
            f"## {self.timestamp} | step={self.step} | "
            f"phase={self.activity_phase} | role={self.role}"
        )
        if self.tool_name:
            header += f" | tool={self.tool_name}"
        tokens_est = len(self.original_content) // 4
        body = (
            f"\n**Original content** ({tokens_est} tokens est.):\n"
            f"{self.original_content}\n\n"
            f"**Compaction event**: {self.compaction_event}"
        )
        return header + body

    def to_vector_metadata(self, session_id: str) -> dict[str, Any]:
        """Build metadata dict for FAISS indexing."""
        return {
            "type": "compacted_turn",
            "id": f"cpt-{session_id}-{self.step}",
            "timestamp": self.timestamp,
            "summary": self.original_content[:400],
            "category": self.activity_phase,
            "extra": {
                "session_id": session_id,
                "step": self.step,
                "activity_phase": self.activity_phase,
                "tool_name": self.tool_name,
                "role": self.role,
            },
        }
