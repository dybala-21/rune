"""Prediction type definitions for RUNE.

Shared types used across the prediction subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rune.proactive.prediction.frustration_detector import FrustrationSignal
from rune.proactive.prediction.need_inferer import InferredNeed


@dataclass(slots=True)
class PredictionResult:
    """Combined result from all prediction subsystems."""

    tool_predictions: list[tuple[str, float]] = field(default_factory=list)
    frustration: FrustrationSignal | None = None
    needs: list[InferredNeed] = field(default_factory=list)
    temporal_quality: float = 0.5
