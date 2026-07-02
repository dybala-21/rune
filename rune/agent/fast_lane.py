"""Simple-query fast lane (docs/design/simple-query-fast-path.md).

High-confidence chat/web goals run on the provider's fast tier with a
tight tool-round cap, and a fresh web search counts as grounding. Other
goal types never enter. Failover only reacts to exceptions and only
walks down the fallback chain, so the loop itself upshifts back to the
primary model when the completion gate blocks repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FAST_LANE_GOAL_TYPES: frozenset[str] = frozenset({"chat", "web"})

# Per-step tool rounds while the lane is active: one search + answer,
# small slack. Bypasses the 12-round floor in _compute_tool_rounds.
FAST_LANE_TOOL_ROUNDS = 3

# Completion-gate blocks tolerated before upshifting to the primary model.
FAST_LANE_UPSHIFT_BLOCKS = 2


@dataclass(frozen=True)
class FastLaneDecision:
    active: bool
    reason: str
    # Loop-format model string ("provider/model"; bare model for openai).
    model: str = ""


def decide_fast_lane(classification: Any) -> FastLaneDecision:
    """Decide whether this goal takes the fast lane and resolve its model.

    Entry is gated on goal_type and confidence only (is_multi_task is
    never populated by the classifier). Never raises: any failure means
    no lane and the run stays on the primary model.
    """
    try:
        return _decide_fast_lane(classification)
    except Exception as exc:
        from rune.utils.logger import get_logger

        get_logger(__name__).warning("fast_lane_decision_failed", error=str(exc)[:200])
        return FastLaneDecision(active=False, reason="error")


def _decide_fast_lane(classification: Any) -> FastLaneDecision:
    from rune.config.loader import get_config

    llm = get_config().llm
    if not getattr(llm, "route_simple_queries", True):
        return FastLaneDecision(active=False, reason="disabled")

    goal_type = getattr(classification, "goal_type", "")
    if goal_type not in FAST_LANE_GOAL_TYPES:
        return FastLaneDecision(active=False, reason=f"goal_type:{goal_type or 'unknown'}")

    confidence = float(getattr(classification, "confidence", 0.0))
    threshold = float(getattr(llm, "simple_query_confidence", 0.8))
    if confidence < threshold:
        return FastLaneDecision(active=False, reason=f"confidence:{confidence:.2f}<{threshold:.2f}")

    from rune.llm.client import get_llm_client, loop_model_string
    from rune.types import ModelTier

    try:
        tier = ModelTier(getattr(llm, "simple_query_tier", "fast"))
    except ValueError:
        from rune.utils.logger import get_logger

        get_logger(__name__).warning(
            "fast_lane_invalid_tier",
            configured=str(getattr(llm, "simple_query_tier", "")),
            fallback="fast",
        )
        tier = ModelTier.FAST

    client = get_llm_client()
    provider = client._effective_provider(None)
    resolved = client.resolve_model(tier, provider)
    if not resolved:
        return FastLaneDecision(active=False, reason="no_tier_model")

    # On ollama every tier resolves to the session model, so the downshift
    # is a no-op there; the round cap and grounding rule still apply.
    return FastLaneDecision(
        active=True,
        reason="simple_goal",
        model=loop_model_string(provider.value, resolved),
    )
