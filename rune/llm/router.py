"""LLM routing logic for RUNE.

Ported from src/llm/router.ts - 3-tier routing (best/coding/fast),
cloud-first vs local-first modes, task-based tier selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rune.config import get_config
from rune.types import Domain, ModelTier, Provider, RiskLevel

# Task context for routing decisions

@dataclass(slots=True)
class TaskContext:
    domain: Domain = Domain.GENERAL
    risk: RiskLevel = RiskLevel.LOW
    requires_code: bool = False
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    action: str = ""


# Routing Rules

def _select_tier(task: TaskContext) -> tuple[ModelTier, str]:
    """Select model tier based on task context. Returns (tier, reason)."""
    # High-risk -> best
    if task.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return ModelTier.BEST, "High-risk task requires best model"

    # Code-related -> coding
    if task.domain == Domain.GIT or task.requires_code:
        return ModelTier.CODING, "Code-related task"

    # Complex -> best
    if task.complexity == "complex":
        return ModelTier.BEST, "Complex task"

    # Simple summarize/classify -> fast
    if task.complexity == "simple" and task.action in ("summarize", "classify", "translate"):
        return ModelTier.FAST, "Simple classification/summarization task"

    # Default -> best
    return ModelTier.BEST, "Default routing"


def _get_provider_order(mode: str) -> list[Provider]:
    """Get provider evaluation order based on routing mode."""
    config = get_config()
    default = Provider(config.llm.default_provider)

    if mode == "local-first":
        others = [p for p in [Provider.OPENAI, Provider.ANTHROPIC] if p != default]
        return [Provider.OLLAMA, default, *others]

    # cloud-first (default)
    others = [p for p in [Provider.OPENAI, Provider.ANTHROPIC] if p != default]
    return [default, *others, Provider.OLLAMA]


# Public API

@dataclass(slots=True)
class RoutingResult:
    model: str
    provider: Provider
    tier: ModelTier
    reason: str


def route(task: TaskContext | None = None) -> RoutingResult:
    """Route a task to the best model/provider combination."""
    config = get_config()
    task = task or TaskContext()

    tier, reason = _select_tier(task)
    providers = _get_provider_order(config.llm.routing_mode)

    # Use the first provider (actual health-check-based selection
    # happens at call time in the LLM client)
    provider = providers[0]

    from rune.llm.client import get_llm_client
    model = get_llm_client().resolve_model(tier, provider)

    return RoutingResult(
        model=model,
        provider=provider,
        tier=tier,
        reason=reason,
    )
