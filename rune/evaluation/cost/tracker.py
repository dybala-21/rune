"""Token cost tracking for RUNE.

Tracks token usage and estimated costs across models, providing
budget management and cost breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Approximate costs per 1M tokens (USD) - major models
_COST_TABLE: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
}

# Fallback cost per 1M tokens for unknown models
_DEFAULT_INPUT_COST = 3.0
_DEFAULT_OUTPUT_COST = 15.0


@dataclass(slots=True)
class _UsageRecord:
    """Internal record of token usage for a single model."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0


class CostTracker:
    """Tracks token usage and estimated costs across models.

    Provides cost breakdowns per model and budget management.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        # model → usage record
        self._records: dict[str, _UsageRecord] = {}

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage for a model.

        Parameters:
            model: The model identifier.
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (completion) tokens.
        """
        if model not in self._records:
            self._records[model] = _UsageRecord(model=model)

        rec = self._records[model]
        rec.input_tokens += input_tokens
        rec.output_tokens += output_tokens
        rec.call_count += 1

        log.debug(
            "cost_recorded",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def get_total_cost(self) -> float:
        """Calculate the estimated total cost across all models.

        Returns:
            Estimated total cost in USD.
        """
        total = 0.0
        for rec in self._records.values():
            costs = _COST_TABLE.get(rec.model, {})
            input_cost = costs.get("input", _DEFAULT_INPUT_COST)
            output_cost = costs.get("output", _DEFAULT_OUTPUT_COST)

            total += (rec.input_tokens / 1_000_000) * input_cost
            total += (rec.output_tokens / 1_000_000) * output_cost

        return round(total, 6)

    def get_breakdown(self) -> dict[str, Any]:
        """Get a detailed cost breakdown per model.

        Returns:
            Dict mapping model names to their usage stats and costs.
        """
        breakdown: dict[str, Any] = {}

        for model, rec in self._records.items():
            costs = _COST_TABLE.get(model, {})
            input_rate = costs.get("input", _DEFAULT_INPUT_COST)
            output_rate = costs.get("output", _DEFAULT_OUTPUT_COST)

            input_cost = (rec.input_tokens / 1_000_000) * input_rate
            output_cost = (rec.output_tokens / 1_000_000) * output_rate

            breakdown[model] = {
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "call_count": rec.call_count,
                "input_cost_usd": round(input_cost, 6),
                "output_cost_usd": round(output_cost, 6),
                "total_cost_usd": round(input_cost + output_cost, 6),
            }

        return breakdown

    def get_budget_remaining(self, budget: float) -> float:
        """Calculate how much of the budget remains.

        Parameters:
            budget: The total budget in USD.

        Returns:
            Remaining budget in USD (may be negative if over budget).
        """
        return round(budget - self.get_total_cost(), 6)
