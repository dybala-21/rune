"""Rehydration trigger — aggregates signals, enforces cooldown and budget.

Called every step on the hot path.  Must return in microseconds when
nothing fires.  All signals evaluated here are synchronous and O(1).
"""

from __future__ import annotations

from rune.agent.rehydration.protocols import LoopStateView, RehydrationDecision
from rune.agent.rehydration.signals import Signal, default_signals
from rune.utils.logger import get_logger

log = get_logger(__name__)


class RehydrationTrigger:
    """Evaluate signals each step, with cooldown and budget guard."""

    def __init__(
        self,
        *,
        signals: list[Signal] | None = None,
        cooldown_steps: int = 3,
        budget_cap: float = 0.85,
        min_step: int = 5,
    ) -> None:
        self._signals = signals or default_signals()
        self._cooldown = cooldown_steps
        self._budget_cap = budget_cap
        self._min_step = min_step
        self._last_fire_step: int | None = None

        # Counters for self-improving integration
        self.total_evaluations: int = 0
        self.total_fires: int = 0
        self.fire_signals: list[str] = []

    def evaluate(self, state: LoopStateView) -> RehydrationDecision:
        """Evaluate all signals. Returns decision in O(1)."""
        self.total_evaluations += 1

        # Guard: too early in session
        if state.step < self._min_step:
            return RehydrationDecision.skip("too_early")

        # Guard: budget tight — don't inject more tokens
        if state.token_budget_fraction > self._budget_cap:
            return RehydrationDecision.skip("budget_tight")

        # Guard: cooldown — prevent back-to-back fires
        if self._last_fire_step is not None:
            if state.step - self._last_fire_step < self._cooldown:
                return RehydrationDecision.skip("cooldown")

        # Evaluate signals in order, fire on first match
        for sig in self._signals:
            reading = sig.evaluate(state)
            if reading.fired:
                self._last_fire_step = state.step
                self.total_fires += 1
                self.fire_signals.append(reading.name)
                log.debug(
                    "rehydration_triggered",
                    signal=reading.name,
                    step=state.step,
                    confidence=reading.confidence,
                )
                return RehydrationDecision.fire(reading)

        return RehydrationDecision.skip("no_signal")
