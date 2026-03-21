"""Prediction engine orchestrator for RUNE.

Combines behaviour prediction, frustration detection, need inference,
and temporal context analysis into a single prediction result.
"""

from __future__ import annotations

from typing import Any

from rune.proactive.context import AwarenessContext
from rune.proactive.prediction.behavior_predictor import BehaviorPredictor
from rune.proactive.prediction.frustration_detector import FrustrationDetector
from rune.proactive.prediction.need_inferer import NeedInferer
from rune.proactive.prediction.temporal_context import TemporalContextAnalyzer
from rune.proactive.prediction.types import PredictionResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

_engine: PredictionEngine | None = None


class PredictionEngine:
    """Orchestrates all prediction subsystems into a combined prediction.

    Combines:
        - BehaviorPredictor: N-gram tool sequence prediction
        - FrustrationDetector: User frustration analysis
        - NeedInferer: Implicit need detection
        - TemporalContextAnalyzer: Time-slot quality assessment
    """

    __slots__ = (
        "_behavior_predictor",
        "_frustration_detector",
        "_need_inferer",
        "_temporal_context",
        "_recent_actions",
    )

    def __init__(
        self,
        behavior_predictor: BehaviorPredictor | None = None,
        frustration_detector: FrustrationDetector | None = None,
        need_inferer: NeedInferer | None = None,
        temporal_context: TemporalContextAnalyzer | None = None,
    ) -> None:
        self._behavior_predictor = behavior_predictor or BehaviorPredictor()
        self._frustration_detector = frustration_detector or FrustrationDetector()
        self._need_inferer = need_inferer or NeedInferer()
        self._temporal_context = temporal_context or TemporalContextAnalyzer()
        self._recent_actions: list[dict[str, object]] = []

    @property
    def behavior_predictor(self) -> BehaviorPredictor:
        return self._behavior_predictor

    @property
    def frustration_detector(self) -> FrustrationDetector:
        return self._frustration_detector

    @property
    def need_inferer(self) -> NeedInferer:
        return self._need_inferer

    @property
    def temporal_context(self) -> TemporalContextAnalyzer:
        return self._temporal_context

    def predict(self, context: dict[str, Any]) -> PredictionResult:
        """Run all prediction subsystems and return a combined result.

        Parameters:
            context: A dict containing:
                - recent_actions (list[dict]): Recent user actions.
                - error_count (int): Number of recent errors.
                - repeated_commands (int): Count of repeated commands.
                - awareness_context (AwarenessContext, optional): Current context.

        Returns:
            A PredictionResult combining all subsystem outputs.
        """
        recent_actions: list[dict[str, Any]] = context.get("recent_actions", [])
        error_count: int = context.get("error_count", 0)
        repeated_commands: int = context.get("repeated_commands", 0)

        # Tool predictions
        tool_predictions = self._behavior_predictor.predict_next()

        # Frustration analysis
        frustration = self._frustration_detector.analyze(
            recent_actions=recent_actions,
            error_count=error_count,
            repeated_commands=repeated_commands,
        )

        # Need inference
        awareness_ctx = context.get("awareness_context")
        if awareness_ctx is None:
            awareness_ctx = AwarenessContext()
        needs = self._need_inferer.infer(awareness_ctx, recent_actions)

        # Temporal quality
        temporal_quality = 1.0 if self._temporal_context.is_good_time() else 0.3

        result = PredictionResult(
            tool_predictions=tool_predictions,
            frustration=frustration,
            needs=needs,
            temporal_quality=temporal_quality,
        )

        log.debug(
            "prediction_complete",
            tool_predictions=len(tool_predictions),
            frustration_level=frustration.level,
            needs=len(needs),
            temporal_quality=temporal_quality,
        )

        return result


def get_prediction_engine() -> PredictionEngine:
    """Get or create the singleton PredictionEngine.

    On first creation, seeds BehaviorPredictor from tool_call_log
    so that cross-session tool patterns are available immediately.
    """
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
        # Seed behavior predictor from persistent tool_call_log
        try:
            from rune.memory.store import get_memory_store
            store = get_memory_store()
            calls = store.get_recent_tool_calls(limit=200)
            _skip = {"uv", "python", "python3", "npx", "run", "exec", "sudo", "-m", "-c"}
            for c in reversed(calls):  # oldest first
                if not c.get("result_success", True):
                    continue  # Skip failed calls — don't learn failure patterns
                name = c["tool_name"]
                if name == "bash_execute":
                    cmd = (c.get("params") or {}).get("command", "")
                    for part in cmd.split():
                        if part not in _skip and not part.startswith("-"):
                            name = f"bash:{part}" if part else name
                            break
                _engine.behavior_predictor.record_tool_call(name)
            if calls:
                log.info("behavior_predictor_seeded", history=len(calls))
        except Exception:
            pass  # Seeding failure must never block the engine
    return _engine
