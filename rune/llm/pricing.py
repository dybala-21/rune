"""Token-cost estimates from published rates, calculated per request."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rates:
    input: float
    output: float
    cached: float
    write: float | None = None
    long_above: int | None = None
    long_output_factor: float = 2.0


# USD per million tokens; reviewed 2026-09-20. Unknown models stay unpriced.
# https://developers.openai.com/api/docs/models/gpt-6-astra
# https://developers.openai.com/api/docs/models/gpt-5.4-mini
# https://developers.openai.com/api/docs/models/gpt-4o
# https://docs.x.ai/developers/pricing
# https://platform.claude.com/docs/en/about-claude/pricing
# https://ai.google.dev/gemini-api/docs/pricing
# https://cloud.google.com/vertex-ai/generative-ai/pricing
_RATES = {
    "openai/gpt-6-astra": Rates(10, 50, 1, 12.5, 272000, 1.5),
    "openai/gpt-5.4-mini": Rates(.75, 4.5, .075),
    "openai/gpt-4o": Rates(2.5, 10, 1.25),
    "xai/grok-4.6": Rates(2, 6, .5, long_above=199999),
    "xai/grok-4.5": Rates(2, 6, .3, long_above=199999),
    "xai/grok-4.3": Rates(1.25, 2.5, .2, long_above=199999),
    "xai/grok-build-0.1": Rates(1, 2, .2, long_above=199999),
    "xai/grok-4.20-0309-reasoning": Rates(1.25, 2.5, .2, long_above=199999),
    "xai/grok-4.20-0309-non-reasoning": Rates(1.25, 2.5, .2, long_above=199999),
    **{f"anthropic/{name}": Rates(5, 25, .5, 6.25) for name in (
        "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5")},
    "anthropic/claude-sonnet-5": Rates(2, 10, .2, 2.5),
    "anthropic/claude-sonnet-4-6": Rates(3, 15, .3, 3.75),
    "anthropic/claude-haiku-4-5": Rates(1, 5, .1, 1.25),
    "gemini/gemini-2.5-flash": Rates(.3, 2.5, .03),
    "gemini/gemini-2.5-pro": Rates(1.25, 10, .125, long_above=200000, long_output_factor=1.5),
    "gemini/gemini-3.1-pro-preview": Rates(2, 12, .2, long_above=200000, long_output_factor=1.5),
}
_LOCAL = {"ollama", "ollama_chat", "lmstudio", "llamacpp", "local", "vllm"}


def model_key(model: str) -> str:
    model = model.replace(":", "/", 1) if "/" not in model else model
    provider, separator, name = model.partition("/")
    if not separator:
        name = provider
        provider = next((p for prefix, p in (("claude-", "anthropic"), ("grok-", "xai"), ("gemini-", "gemini"))
                         if name.startswith(prefix)), "openai")
    if provider == "vertex_ai" and name.startswith("gemini-"):
        provider = "gemini"
    if provider == "openai":
        name = name.removeprefix("responses/")
    if provider == "xai":
        from rune.llm.xai import model_name
        name = model_name(f"xai/{name}") or name
    if provider == "anthropic":
        name = re.sub(r"-\d{8}$", "", name)
    return f"{provider}/{name}"


def rates_for(model: str) -> Rates | None:
    key = model_key(model)
    if key.partition("/")[0] in _LOCAL:
        return Rates(0, 0, 0, 0)
    return _RATES.get(key)


def estimate_request_cost(model: str, usage: dict, request: dict | None = None) -> float | None:
    """Estimate token charges only; unknown rates or usage return None."""
    request = {**(request or {}), **((request or {}).get("extra_body") or {})}
    rates = rates_for(model)
    if rates is None or request.get("api_base") or request.get("base_url"):
        return None
    if request.get("service_tier") not in (None, "auto", "default", "standard"):
        return None
    input_tokens, output = usage.get("input_tokens"), usage.get("output_tokens")
    cached, written = usage.get("cached_input_tokens", 0), usage.get("cache_write_tokens", 0)
    hour = usage.get("cache_write_1h_tokens", 0)
    if any(type(n) is not int or n < 0 for n in (input_tokens, output, cached, written, hour)):
        return None
    if cached + written > input_tokens or hour > written or written and rates.write is None:
        return None
    key = model_key(model)
    if model.startswith("vertex_ai/") and key.startswith("gemini/gemini-3") and request.get("vertex_location") != "global":
        return None
    if key == "openai/gpt-6-astra" and not usage.get("cache_write_reported"):
        return None
    factor = 1.0
    if request.get("speed") == "fast":
        if key not in {"anthropic/claude-opus-5", "anthropic/claude-opus-4-8"}:
            return None
        factor = 2.0
    if request.get("inference_geo") not in (None, "global"):
        return None
    # Explicit cache storage and audio have separate rates and are not included.
    if request.get("cached_content") or request.get("modalities") or any(
        part.get("type") in {"input_audio", "audio_url", "video_url"}
        for message in request.get("messages", []) if isinstance(message.get("content"), list)
        for part in message["content"] if isinstance(part, dict)
    ):
        return None
    long = rates.long_above is not None and input_tokens > rates.long_above
    input_factor, output_factor = (2.0, rates.long_output_factor) if long else (1.0, 1.0)
    total = ((input_tokens - cached - written) * rates.input + cached * rates.cached
             + (written - hour) * (rates.write or 0) + hour * rates.input * 2) * input_factor
    total += output * rates.output * output_factor
    return total * factor / 1_000_000


def usage_payload(trace: Any) -> dict | None:
    usage = getattr(trace, "timings", {}).get("usage")
    if not usage:
        from rune.agent.timing import current_usage
        usage = current_usage()
    if not usage:
        return None
    missing = usage["calls"] - usage["reported_calls"] + usage.get("unpriced_calls", usage["reported_calls"])
    return {"total": usage["total_tokens"], "input": usage["input_tokens"], "output": usage["output_tokens"],
            "cacheRead": usage["cached_input_tokens"], "cacheCreation": usage["cache_write_tokens"],
            "cost": {"usd": None if missing else usage.get("cost_usd", 0),
                     "knownUsd": usage.get("cost_usd", 0), "unpricedCalls": missing,
                     "scope": "model_tokens"}}


def combine_usage_payloads(*parts: dict | None, pending: bool = False, incomplete: bool = False) -> dict | None:
    parts = tuple(part for part in parts if part is not None)
    if not parts:
        return None
    missing = sum((part.get("cost") or {}).get("unpricedCalls", 0) for part in parts)
    known = sum((part.get("cost") or {}).get("knownUsd", 0) for part in parts)
    complete = not (pending or incomplete) and all((part.get("cost") or {}).get("usd") is not None for part in parts)
    return {**{key: sum(part.get(key, 0) for part in parts)
               for key in ("total", "input", "output", "cacheRead", "cacheCreation")},
            "cost": {"usd": known if complete else None, "knownUsd": known,
                     "unpricedCalls": missing, **({"pending": True} if pending else {}),
                     **({"incomplete": True} if incomplete else {}), "scope": "model_tokens"}}
