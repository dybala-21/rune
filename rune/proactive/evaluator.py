"""Intelligent proactive evaluator for RUNE.

Scores suggestions across four dimensions (urgency, relevance,
receptivity, value) and decides whether to act on them. Integrates
with ReflexionLearner for adaptive threshold management.

Ported from src/proactive/evaluator.ts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rune.proactive.context import AwarenessContext
from rune.proactive.intelligence import (
    OpportuneScore,
    compute_opportune_score,
    detect_intent_signals,
    should_intervene,
)
from rune.proactive.types import EngagementMetrics, Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Result of evaluating a proactive suggestion."""

    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    should_act: bool = False
    event_type: str = ""
    opportune_score: OpportuneScore | None = None


class ProactiveEvaluator:
    """Evaluates suggestions with reflexion-adaptive thresholds.

    Integrates with ReflexionLearner to:
    - Skip disabled event types
    - Enforce minimum interval between interventions
    - Apply adjusted score thresholds
    - Record rejections for future learning
    """

    __slots__ = ("_last_intervention_time", "_min_interval_seconds")

    def __init__(self, min_interval_seconds: float = 120.0) -> None:
        self._last_intervention_time: float = 0.0
        self._min_interval_seconds = min_interval_seconds

    def evaluate(
        self,
        suggestion: Suggestion,
        context: AwarenessContext,
        engagement: EngagementMetrics,
        *,
        event_type: str = "",
        recent_commands: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a suggestion with full intelligence pipeline.

        Parameters:
            suggestion: The candidate suggestion.
            context: Current environment/awareness context.
            engagement: User engagement metrics.
            event_type: The triggering event type (e.g. "task_failed").
            recent_commands: Recent user commands for intent detection.

        Returns:
            An EvaluationResult with the combined score and decision.
        """
        reasons: list[str] = []

        # 1. Check reflexion: is this event type disabled?
        try:
            from rune.proactive.reflexion import get_reflexion_learner

            learner = get_reflexion_learner()
            if event_type and learner.is_event_disabled(event_type):
                reasons.append(f"event_type '{event_type}' disabled by reflexion")
                return EvaluationResult(
                    score=0.0, reasons=reasons, should_act=False, event_type=event_type,
                )
        except Exception:
            learner = None

        # 2. Check minimum interval
        elapsed = time.monotonic() - self._last_intervention_time
        min_interval = self._min_interval_seconds
        if learner:
            min_interval = max(min_interval, learner.get_min_interval())
        if elapsed < min_interval:
            reasons.append(
                f"too soon: {elapsed:.0f}s < {min_interval:.0f}s min interval"
            )
            return EvaluationResult(
                score=0.0, reasons=reasons, should_act=False, event_type=event_type,
            )

        # 3. Compute opportune score
        score = compute_opportune_score(
            suggestion, context, engagement,
            event_type=event_type,
            recent_commands=recent_commands,
        )
        reasons.extend(score.reasoning)

        # 4. Detect intent signals
        signals = detect_intent_signals(context, recent_commands)
        if signals:
            for sig in signals:
                reasons.append(f"intent: {sig.intent} (conf={sig.confidence:.2f})")

        # 5. Apply reflexion threshold override
        threshold_override: float | None = None
        if learner:
            threshold_override = learner.get_score_threshold()

        # 6. Decide
        act = should_intervene(
            score, event_type, signals=signals, min_score_override=threshold_override,
        )

        if act:
            self._last_intervention_time = time.monotonic()

        log.debug(
            "proactive_evaluation",
            suggestion_type=suggestion.type,
            event_type=event_type,
            score=score.total,
            should_act=act,
        )

        return EvaluationResult(
            score=score.total,
            reasons=reasons,
            should_act=act,
            event_type=event_type,
            opportune_score=score,
        )

    def reset_intervention_time(self) -> None:
        """Reset the last intervention timer (e.g. after user-initiated action)."""
        self._last_intervention_time = 0.0
