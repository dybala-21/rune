"""Rehydration signals — detect when the agent needs past context.

Each signal is a small, stateless (or self-contained) class that reads
the ``LoopStateView`` and returns a ``SignalReading``.

All signals in this file are hot-path safe (O(1), no I/O).
Slow signals (e.g. SemanticDrift) belong in a future phase.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rune.agent.rehydration.protocols import LoopStateView, SignalReading


# Signal protocol
@runtime_checkable
class Signal(Protocol):
    """Plug-in rehydration signal.

    Implementations must not modify loop state.
    """

    name: str

    def evaluate(self, state: LoopStateView) -> SignalReading: ...


# Concrete signals
class PhaseTransitionSignal:
    """Fires on activity phase change."""

    name = "phase_transition"

    def evaluate(self, state: LoopStateView) -> SignalReading:
        if state.phase_just_changed:
            return SignalReading(self.name, True, confidence=0.8)
        return SignalReading(self.name, False)


class StallRecoverySignal:
    """Fires when the agent has made no progress for N consecutive steps."""

    name = "stall_recovery"

    def __init__(self, threshold: int = 2) -> None:
        self._threshold = threshold

    def evaluate(self, state: LoopStateView) -> SignalReading:
        if state.stall_consecutive >= self._threshold:
            return SignalReading(
                self.name,
                True,
                confidence=0.9,
                metadata={"stall_count": state.stall_consecutive},
            )
        return SignalReading(self.name, False)


class GateBlockedSignal:
    """Fires when the completion gate has blocked this run."""

    name = "gate_blocked"

    def evaluate(self, state: LoopStateView) -> SignalReading:
        if state.gate_blocked_count >= 1:
            return SignalReading(self.name, True, confidence=0.85)
        return SignalReading(self.name, False)


# Default signal set
def default_signals() -> list[Signal]:
    """Return the default set of hot-path signals."""
    return [
        PhaseTransitionSignal(),
        StallRecoverySignal(threshold=2),
        GateBlockedSignal(),
    ]
