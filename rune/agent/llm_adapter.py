"""LLM adapter - model selection for NativeAgentLoop.

Ported from src/agent/llm-adapter.ts.  Provides ``create_agent_model``,
``analyze_goal_complexity``, and ``create_model_from_classification``
which map goal metadata to the appropriate LLM model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

Complexity = Literal["simple", "moderate", "complex"]


@dataclass(slots=True)
class GoalAnalysis:
    """Result of analysing goal complexity."""

    complexity: Complexity
    requires_code: bool


@dataclass(slots=True)
class ModelSelection:
    """Result of model selection: provider + model id."""

    provider: str
    model_id: str
    reason: str = ""


# Goal complexity analysis (heuristic, no LLM call)

async def analyze_goal_complexity(goal: str) -> GoalAnalysis:
    """Analyse *goal* text and return complexity + code requirement.

    Uses the goal classifier (``rune.agent.goal_classifier``) when available,
    otherwise falls back to simple heuristics.
    """
    try:
        from rune.agent.goal_classifier import classify_goal

        classification = await classify_goal(goal)
        return GoalAnalysis(
            complexity=classification.complexity,  # type: ignore[arg-type]
            requires_code=classification.requires_code,
        )
    except Exception:
        log.debug("goal_classifier_unavailable_fallback")
        return _heuristic_complexity(goal)


def _heuristic_complexity(goal: str) -> GoalAnalysis:
    """Simple keyword-based complexity estimation."""
    goal_lower = goal.lower()

    complex_signals = [
        "refactor", "migrate", "architecture", "redesign",
        "performance", "optimize", "security audit", "deploy",
        "ci/cd", "pipeline", "infrastructure",
    ]
    code_signals = [
        "code", "function", "class", "implement", "write",
        "test", "debug", "fix bug", "create file", "script",
        "api", "endpoint", "component",
    ]

    requires_code = any(kw in goal_lower for kw in code_signals)

    complex_count = sum(1 for kw in complex_signals if kw in goal_lower)
    if complex_count >= 2 or len(goal) > 500:
        complexity: Complexity = "complex"
    elif complex_count >= 1 or requires_code or len(goal) > 200:
        complexity = "moderate"
    else:
        complexity = "simple"

    return GoalAnalysis(complexity=complexity, requires_code=requires_code)


# Model creation helpers

async def create_agent_model(
    provider_name: str | None = None,
    model_id: str | None = None,
) -> ModelSelection:
    """Create (select) an LLM model for agent usage.

    If *provider_name* and *model_id* are given explicitly, they are
    validated and returned.  Otherwise the LLM router picks the default.
    """
    try:
        from rune.llm.client import get_llm_client

        client = get_llm_client()
        await client.initialize()

        if provider_name and model_id:
            return ModelSelection(
                provider=provider_name,
                model_id=model_id,
                reason="explicit_selection",
            )

        # Default: route for simple task
        from rune.llm.router import get_llm_router

        router = get_llm_router()
        result = await router.route(complexity="simple")
        return ModelSelection(
            provider=result.provider,
            model_id=result.model_id,
            reason=result.reason,
        )
    except ImportError:
        log.debug("llm_client_not_available_returning_default")
        return ModelSelection(
            provider="openai",
            model_id="gpt-4o",
            reason="fallback_default",
        )


async def create_agent_model_for_goal(goal: str) -> ModelSelection:
    """Analyse *goal* complexity, then route to the best model.

    Complex coding tasks -> stronger model; simple tasks -> faster model.
    """
    analysis = await analyze_goal_complexity(goal)

    try:
        from rune.llm.router import get_llm_router

        router = get_llm_router()
        result = await router.route(
            complexity=analysis.complexity,
            requires_code=analysis.requires_code,
        )
        log.info(
            "goal_based_model_selection",
            complexity=analysis.complexity,
            requires_code=analysis.requires_code,
            selected_model=result.model_id,
            reason=result.reason,
        )
        return ModelSelection(
            provider=result.provider,
            model_id=result.model_id,
            reason=result.reason,
        )
    except ImportError:
        log.debug("llm_router_not_available_returning_default")
        return ModelSelection(
            provider="openai",
            model_id="gpt-4o",
            reason="fallback_default",
        )


async def create_model_from_classification(
    classification: Any,
) -> ModelSelection:
    """Select a model based on a pre-computed GoalClassification.

    Used as a *model_factory* by ``NativeAgentLoop.run()``.
    This function does NOT call the LLM - only evaluates router rules (<1ms).
    """
    complexity: Complexity = getattr(classification, "complexity", "simple")
    requires_code: bool = getattr(classification, "requires_code", False)

    try:
        from rune.llm.router import get_llm_router

        router = get_llm_router()
        result = await router.route(
            complexity=complexity,
            requires_code=requires_code,
        )
        return ModelSelection(
            provider=result.provider,
            model_id=result.model_id,
            reason=result.reason,
        )
    except ImportError:
        return ModelSelection(
            provider="openai",
            model_id="gpt-4o",
            reason="fallback_default",
        )
