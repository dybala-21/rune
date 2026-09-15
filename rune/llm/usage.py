"""Normalize provider usage without discarding cache billing fields."""

from __future__ import annotations

from functools import wraps
from typing import Any


def _get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _count(value: Any, *paths: str) -> int | None:
    for path in paths:
        item = value
        for key in path.split("."):
            item = _get(item, key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            return item
    return None


def token_counts(usage: Any) -> dict[str, Any] | None:
    input_tokens = _count(usage, "prompt_tokens", "input_tokens")
    output_tokens = _count(usage, "completion_tokens", "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    cached = _count(usage, "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens",
                    "prompt_tokens_details.cache_read_input_tokens", "input_token_details.cache_read",
                    "input_token_details.cached_tokens", "cache_read_input_tokens", "cached_tokens", "cache_read_tokens")
    written = _count(usage, "prompt_tokens_details.cache_creation_tokens", "prompt_tokens_details.cache_write_tokens",
                     "input_tokens_details.cache_write_tokens", "input_token_details.cache_creation", "input_token_details.cache_write",
                     "cache_creation_input_tokens", "cache_write_tokens", "cache_write_input_tokens")
    # Native Anthropic input_tokens excludes cache reads and writes; LiteLLM prompt_tokens includes them.
    if _get(usage, "prompt_tokens") is None and _get(usage, "cache_read_input_tokens") is not None:
        input_tokens += (cached or 0) + (written or 0)
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached or 0, "cache_write_tokens": written or 0,
        "cache_write_reported": written is not None,
        "reasoning_tokens": _count(usage, "completion_tokens_details.reasoning_tokens",
                                   "output_tokens_details.reasoning_tokens", "output_token_details.reasoning_tokens", "reasoning_tokens") or 0,
    }


def merge_usage(previous: dict | None, counts: dict) -> dict:
    """Merge cumulative snapshots from one request, including partial final events."""
    if previous is None:
        return counts
    merged = {key: max(value, previous[key]) for key, value in counts.items()}
    merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged


def preserve_responses_cache_usage() -> None:
    """Bridge the cache-write field omitted by older LiteLLM normalizers."""
    from litellm.responses.utils import ResponseAPILoggingUtils

    original = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage
    if getattr(original, "_rune_cache_usage", False):
        return

    @wraps(original)
    def normalize(usage_input):
        result = original(usage_input)
        written = _count(usage_input, "input_tokens_details.cache_write_tokens")
        if written is not None and result.prompt_tokens_details is not None:
            result.prompt_tokens_details.cache_creation_tokens = written
        return result

    normalize._rune_cache_usage = True
    ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage = staticmethod(normalize)
