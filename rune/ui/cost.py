"""Display helpers for the shared model-token cost estimate."""

from rune.llm.pricing import estimate_request_cost


def estimate_cost(model: str, input_tokens: int, output_tokens: int, *,
                  cached_input_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    return estimate_request_cost(model, {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens, "cache_write_tokens": cache_write_tokens,
        "cache_write_reported": True,
    })


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "Unavailable"
    return f"${cost:.4f}" if cost < 1 else f"${cost:.2f}"
