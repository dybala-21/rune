"""User frustration detection for RUNE.

Analyses recent user actions to detect frustration signals such as
repeated failures, rapid cancellations, and error loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

FrustrationLevel = Literal["none", "mild", "moderate", "high"]

# Thresholds
_REPEATED_FAILURE_THRESHOLD = 3
_RAPID_CANCELLATION_THRESHOLD = 3
_ERROR_HIGH_THRESHOLD = 5
_ERROR_MODERATE_THRESHOLD = 3


@dataclass(slots=True)
class FrustrationSignal:
    """Result of frustration analysis."""

    level: FrustrationLevel = "none"
    indicators: list[str] = field(default_factory=list)
    suggested_action: str = ""


class FrustrationDetector:
    """Detects user frustration from behavioural signals.

    Analyses recent action patterns, error counts, and repeated
    commands to produce a frustration signal with suggested action.
    """

    __slots__ = ()

    def analyze(
        self,
        recent_actions: list[dict[str, Any]],
        error_count: int,
        repeated_commands: int,
    ) -> FrustrationSignal:
        """Analyse recent context for frustration signals.

        Parameters:
            recent_actions: List of recent action dicts (each should have "type"
                and optionally "success", "tool" keys).
            error_count: Number of errors in the recent window.
            repeated_commands: Count of consecutively repeated commands.

        Returns:
            A FrustrationSignal with the detected level and suggested action.
        """
        indicators: list[str] = []
        score = 0

        # Check repeated failures
        if self._check_repeated_failures(recent_actions):
            indicators.append("repeated_failures")
            score += 2

        # Check rapid cancellations
        if self._check_rapid_cancellations(recent_actions):
            indicators.append("rapid_cancellations")
            score += 2

        # Error count contribution
        if error_count >= _ERROR_HIGH_THRESHOLD:
            indicators.append(f"high_error_count ({error_count})")
            score += 3
        elif error_count >= _ERROR_MODERATE_THRESHOLD:
            indicators.append(f"elevated_error_count ({error_count})")
            score += 1

        # Repeated commands
        if repeated_commands >= _REPEATED_FAILURE_THRESHOLD:
            indicators.append(f"repeated_commands ({repeated_commands})")
            score += 1

        # Determine level
        if score >= 4:
            level: FrustrationLevel = "high"
            suggested_action = "Offer to take a different approach or explain the issue."
        elif score >= 2:
            level = "moderate"
            suggested_action = "Suggest an alternative strategy or provide more context."
        elif score >= 1:
            level = "mild"
            suggested_action = "Acknowledge difficulty and offer help."
        else:
            level = "none"
            suggested_action = ""

        signal = FrustrationSignal(
            level=level,
            indicators=indicators,
            suggested_action=suggested_action,
        )

        if level != "none":
            log.debug(
                "frustration_detected",
                level=level,
                indicators=indicators,
                score=score,
            )

        return signal

    def _check_repeated_failures(self, actions: list[dict[str, Any]]) -> bool:
        """Check if recent actions show a pattern of repeated failures.

        Returns True if the last N actions all failed.
        """
        if len(actions) < _REPEATED_FAILURE_THRESHOLD:
            return False

        recent = actions[-_REPEATED_FAILURE_THRESHOLD:]
        return all(not a.get("success", True) for a in recent)

    def _check_rapid_cancellations(self, actions: list[dict[str, Any]]) -> bool:
        """Check if recent actions show a pattern of rapid cancellations.

        Returns True if multiple recent actions were cancellations.
        """
        if len(actions) < _RAPID_CANCELLATION_THRESHOLD:
            return False

        recent = actions[-_RAPID_CANCELLATION_THRESHOLD * 2:]
        cancellation_count = sum(
            1 for a in recent if a.get("type") == "cancellation"
        )
        return cancellation_count >= _RAPID_CANCELLATION_THRESHOLD
