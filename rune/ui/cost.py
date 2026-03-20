"""API cost estimation for RUNE TUI.

Provides per-model pricing data and helpers to estimate and
format the dollar cost of API calls.
"""

from __future__ import annotations

# Cost table: model → (input_cost_per_1k, output_cost_per_1k) in USD

COST_PER_1K: dict[str, tuple[float, float]] = {
    # OpenAI - latest
    "gpt-5.4": (0.0025, 0.0100),
    "gpt-5.4-pro": (0.0100, 0.0400),
    "gpt-5-mini": (0.0003, 0.0012),
    "gpt-5-nano": (0.0001, 0.0004),
    "gpt-5.3-codex": (0.0025, 0.0100),
    "gpt-5.2": (0.0025, 0.0100),
    # OpenAI - older
    "gpt-4.1": (0.0020, 0.0080),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.000150, 0.000600),
    "o4-mini": (0.0011, 0.0044),
    "o3-mini": (0.0011, 0.0044),
    # Anthropic
    "claude-opus-4-6": (0.0150, 0.0750),
    "claude-opus-4-5-20251101": (0.0150, 0.0750),
    "claude-sonnet-4-5-20250929": (0.0030, 0.0150),
    "claude-sonnet-4-20250514": (0.0030, 0.0150),
    "claude-haiku-4-5-20251001": (0.0008, 0.0040),
    # Gemini
    "gemini-2.5-flash": (0.000150, 0.000600),
    # Ollama (local - zero cost)
    "llama3:70b": (0.0, 0.0),
    "llama3:8b": (0.0, 0.0),
    "codellama:34b": (0.0, 0.0),
    "mistral:7b": (0.0, 0.0),
    "mixtral:8x7b": (0.0, 0.0),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate the dollar cost of an API call.

    Parameters
    ----------
    model:
        Model identifier (must be a key in COST_PER_1K).
    input_tokens:
        Number of input (prompt) tokens.
    output_tokens:
        Number of output (completion) tokens.

    Returns
    -------
    Estimated cost in USD. Returns 0.0 for unknown models.
    """
    rates = COST_PER_1K.get(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate


def format_cost(cost: float) -> str:
    """Format a dollar cost as a string.

    Examples: "$0.0000", "$0.0234", "$1.50"
    """
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1.0:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
