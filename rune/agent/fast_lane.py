"""Use the fast tier for eligible chat/web tasks in automatic model mode.

The loop limits tool rounds and returns to the primary model after repeated
completion blocks. A fresh web search supplies grounding for web answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FAST_LANE_GOAL_TYPES: frozenset[str] = frozenset({"chat", "web"})

# Overrides the usual 12-round minimum for simple tasks.
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
    """Select a fast-tier model when settings and classification permit it.

    Keep the primary model if selection fails.
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
    if (getattr(llm, "active_model", None) or "").strip():
        return FastLaneDecision(active=False, reason="explicit_model")
    if getattr(classification, "decision_backend", "connected") != "connected":
        return FastLaneDecision(active=False, reason="backend_confidence_not_comparable")

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
