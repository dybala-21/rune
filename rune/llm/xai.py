"""Grok model discovery and request capabilities."""

from __future__ import annotations

import hashlib
import os
import time

import httpx

from rune.utils.logger import get_logger

log = get_logger(__name__)

# https://docs.x.ai/developers/models
MODELS = (
    "grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1",
    "grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning",
)
REASONING = {
    "grok-4.6": ("low", "medium", "high", "xhigh"),
    "grok-4.5": ("low", "medium", "high", "xhigh"),
    "grok-4.3": ("none", "low", "medium", "high", "xhigh"),
}
_ALIASES = {
    "grok-4.5-latest": "grok-4.5", "grok-build-latest": "grok-4.5",
    "grok-4.3-latest": "grok-4.3",
    "grok-code-fast-1": "grok-build-0.1", "grok-code-fast": "grok-build-0.1",
    "grok-4.20-reasoning": "grok-4.20-0309-reasoning",
    "grok-4.20-reasoning-latest": "grok-4.20-0309-reasoning",
    "grok-4.20-non-reasoning": "grok-4.20-0309-non-reasoning",
    "grok-4.20-non-reasoning-latest": "grok-4.20-0309-non-reasoning",
}
_cached: tuple[str, float, list[str] | None] | None = None


def model_name(model: str) -> str | None:
    provider, separator, name = model.partition("/")
    if separator and provider != "xai":
        return None
    name = name if separator else model
    name = _ALIASES.get(name, name)
    return name if name in MODELS else None


def invalidate_cache() -> None:
    global _cached
    _cached = None


async def available_model_ids() -> list[str] | None:
    """List the account's language models, or return None to use fallbacks."""
    global _cached
    from rune.config import get_config

    get_config()  # Load the user's .env before checking credentials.
    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        return None
    credential = hashlib.sha256(key.encode()).hexdigest()
    now = time.monotonic()
    if _cached is not None and _cached[0] == credential and now - _cached[1] < 300:
        return _cached[2]
    models = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://api.x.ai/v1/language-models",
                                        headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
            rows = response.json()["models"]
        if not isinstance(rows, list):
            raise ValueError("Invalid language model catalog")
        candidates = []
        for row in rows:
            name = row.get("id")
            # Server-managed multi-agent models use a different tool protocol.
            if (isinstance(name, str) and name.startswith("grok-") and "multi-agent" not in name
                    and "text" in row.get("output_modalities", [])):
                candidates.append((int(row.get("created", 0)), name))
        models = list(dict.fromkeys(name for _, name in sorted(candidates, reverse=True)))
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        log.debug("xai_model_fetch_failed", error=type(exc).__name__)
    _cached = (credential, now, models)
    return models
