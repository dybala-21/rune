"""Context Window Budget Management for RUNE.

Ported from src/conversation/context-budget.ts -- goal-aware context budget
policy that applies dynamic defaults while preserving explicit overrides.

When a ``GoalClassification`` hint is provided the budget is derived from
the LLM classification result; otherwise a regex-based fallback is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ============================================================================
# Types
# ============================================================================


@dataclass
class GoalClassification:
    """Minimal representation of a goal classifier result."""

    category: str = ""  # e.g. "chat", "code", "web"
    complexity: str = ""  # "simple" | "complex"
    is_continuation: bool = False
    is_complex_coding: bool = False
    requires_code: bool = False
    requires_execution: bool = False


@dataclass
class ResolveContextBudgetOptions:
    conversation_budget_tokens: int | None = None
    memory_budget_tokens: int | None = None
    conversation_budget_chars: int | None = None
    memory_budget_chars: int | None = None
    model_context_window_tokens: int | None = None
    reserved_context_tokens: int | None = None
    classification_hint: GoalClassification | None = None


@dataclass
class ResolvedContextBudget:
    conversation_budget_tokens: int = 0
    memory_budget_tokens: int = 0


# ============================================================================
# Regex Fallback Patterns
# ============================================================================

_CASUAL_CHAT_PATTERN = re.compile(
    r"^(안녕|하이|hello|hi|hey|ㅎㅎ|ㅋㅋ|lol|sup|yo|고마워|thanks|ok|네|응)$",
    re.IGNORECASE,
)
_SIMPLE_WEB_PATTERN = re.compile(
    r"검색|조회|찾아|뉴스|날씨|weather|search|look\s*up|latest|기사",
    re.IGNORECASE,
)
_COMPLEX_CODING_PATTERN = re.compile(
    r"아키텍처|마이크로서비스|microservice|웹소켓|websocket|디스코드|discord"
    r"|구현|개발|리팩|refactor|ci/cd|배포|deployment|인프라|테스트\s*전략|full-?stack",
    re.IGNORECASE,
)
_CONTINUATION_PATTERN = re.compile(
    r"다음\s*단계|이어서|계속|continue|next\s*step|마저",
    re.IGNORECASE,
)


# ============================================================================
# Helpers
# ============================================================================


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _split_budget(total_tokens: int) -> ResolvedContextBudget:
    total = _clamp(round(total_tokens), 12_000, 64_000)
    conversation = round(total * (2 / 3))
    memory = total - conversation
    return ResolvedContextBudget(
        conversation_budget_tokens=_clamp(conversation, 6_000, 48_000),
        memory_budget_tokens=_clamp(memory, 2_000, 16_000),
    )


# ============================================================================
# Budget Inference
# ============================================================================


def _infer_budget_from_classification(c: GoalClassification) -> int:
    """Derive a token budget from a GoalClassification object."""
    if c.category == "chat" and c.complexity == "simple" and not c.is_continuation:
        return 12_000
    if c.is_complex_coding:
        return 36_000
    if c.complexity == "complex" or (c.requires_code and c.requires_execution):
        return 36_000
    if c.category == "web" and c.complexity == "simple":
        return 20_000
    if c.is_continuation:
        return 30_000
    if c.requires_code:
        return 30_000
    return 24_000


def _infer_dynamic_total_budget(goal: str) -> int:
    """Regex fallback when no classification hint is available."""
    lower = goal.strip().lower()

    if _CASUAL_CHAT_PATTERN.search(lower):
        return 12_000
    if _COMPLEX_CODING_PATTERN.search(lower) or len(lower) >= 120:
        return 36_000
    if _SIMPLE_WEB_PATTERN.search(lower):
        return 20_000
    if _CONTINUATION_PATTERN.search(lower):
        return 30_000
    return 24_000


def _constrain_by_model_window(
    total_budget: int,
    model_context_window_tokens: int | None,
    reserved_context_tokens: int | None,
) -> int:
    if not model_context_window_tokens or model_context_window_tokens <= 0:
        return total_budget

    reserved = reserved_context_tokens or max(
        8_000, round(model_context_window_tokens * 0.15)
    )
    available = max(12_000, model_context_window_tokens - reserved)
    model_max_budget = min(64_000, round(available * 0.5))
    return _clamp(total_budget, 12_000, model_max_budget)


# ============================================================================
# Public API
# ============================================================================


def resolve_context_budget(
    goal: str,
    options: ResolveContextBudgetOptions | None = None,
) -> ResolvedContextBudget:
    """Resolve the context budget for a given *goal*.

    Priority:
    1. Explicit token overrides (``conversation_budget_tokens`` /
       ``memory_budget_tokens``).
    2. Character-based overrides converted to tokens (/ 4).
    3. Classification hint (LLM-derived) budget.
    4. Regex-based dynamic budget fallback.

    The result is further constrained by the model context window when
    ``model_context_window_tokens`` is provided.
    """
    if options is None:
        options = ResolveContextBudgetOptions()

    # 1) Explicit token overrides
    if options.conversation_budget_tokens or options.memory_budget_tokens:
        return ResolvedContextBudget(
            conversation_budget_tokens=_clamp(
                options.conversation_budget_tokens or 16_000, 1_000, 64_000
            ),
            memory_budget_tokens=_clamp(
                options.memory_budget_tokens or 8_000, 1_000, 32_000
            ),
        )

    # 2) Character-based overrides
    if options.conversation_budget_chars or options.memory_budget_chars:
        return ResolvedContextBudget(
            conversation_budget_tokens=_clamp(
                round(options.conversation_budget_chars / 4)
                if options.conversation_budget_chars
                else 16_000,
                1_000,
                64_000,
            ),
            memory_budget_tokens=_clamp(
                round(options.memory_budget_chars / 4)
                if options.memory_budget_chars
                else 8_000,
                1_000,
                32_000,
            ),
        )

    # 3/4) Classification hint or regex fallback
    if options.classification_hint:
        dynamic_budget = _infer_budget_from_classification(options.classification_hint)
    else:
        dynamic_budget = _infer_dynamic_total_budget(goal)

    bounded_budget = _constrain_by_model_window(
        dynamic_budget,
        options.model_context_window_tokens,
        options.reserved_context_tokens,
    )
    return _split_budget(bounded_budget)
