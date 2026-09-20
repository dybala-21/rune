"""Compatibility helpers using the shared pricing catalog."""

from rune.llm.pricing import rates_for
from rune.ui.cost import estimate_cost
from rune.ui.cost import format_cost as format_cost


def get_model_pricing(model: str) -> dict[str, float] | None:
    rates = rates_for(model)
    return {"input": rates.input, "output": rates.output} if rates is not None else None


def estimate_session_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    # Aggregate totals cannot establish per-request long-context pricing.
    rates = rates_for(model)
    if rates is None or rates.long_above is not None:
        return None
    return estimate_cost(model, input_tokens, output_tokens)
