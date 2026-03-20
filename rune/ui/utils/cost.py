"""Cost tracking utilities for LLM sessions.

Mirrors the TS cost.ts module - provides model pricing data,
per-session cost estimation, and formatting helpers.
"""

from __future__ import annotations

# Model pricing (USD per 1 M tokens)

MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-5-mini": {"input": 0.50, "output": 2.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Fallback (used when model is unrecognised)
    "_default": {"input": 2.00, "output": 8.00},
}


def get_model_pricing(model: str) -> dict[str, float]:
    """Return ``{"input": ..., "output": ...}`` pricing for *model*.

    If the exact model name is not found the function tries a prefix match
    (e.g. ``"gpt-5.2-preview"`` matches ``"gpt-5.2"``).  Falls back to
    ``_default`` pricing when nothing matches.
    """
    if model in MODEL_PRICING and model != "_default":
        return MODEL_PRICING[model]

    # Prefix / partial match
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if key == "_default":
            continue
        if model.startswith(key) or key.startswith(model):
            return MODEL_PRICING[key]

    return MODEL_PRICING["_default"]


def estimate_session_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate the dollar cost for a session given token counts.

    Returns the cost in USD.
    """
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def format_cost(amount: float) -> str:
    """Format a dollar amount for display.

    Examples: ``"$0.00"``, ``"$0.02"``, ``"$1.23"``.
    """
    if amount < 0.01:
        return f"${amount:.4f}" if amount > 0 else "$0.00"
    return f"${amount:.2f}"
