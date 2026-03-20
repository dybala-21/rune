"""Model registry for RUNE.

Multi-provider model listing with dynamic OpenAI fetch, hardcoded
fallbacks for all providers, and 5-minute caching.

Supported providers:
  OpenAI, Anthropic, Google Gemini, xAI Grok, Azure OpenAI,
  Mistral, DeepSeek, Cohere

All providers use API Key authentication. Azure additionally supports
Entra ID (OAuth 2.0). OAuth is NOT supported for Anthropic or Google
due to account suspension risk (OpenClaw ban wave, 2025-2026).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

from rune.config import get_config
from rune.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ModelInfo:
    """Information about a single LLM model."""
    id: str
    provider: str
    label: str


@dataclass(slots=True, frozen=True)
class ProviderInfo:
    """Provider with its available models."""
    id: str
    label: str
    available: bool
    models: list[ModelInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider configurations - env var name for API key detection
# ---------------------------------------------------------------------------

PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openai":    ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini":    ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "xai":       ["XAI_API_KEY"],
    "azure":     ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
    "mistral":   ["MISTRAL_API_KEY"],
    "deepseek":  ["DEEPSEEK_API_KEY"],
    "cohere":    ["COHERE_API_KEY", "CO_API_KEY"],
}


def _has_provider_key(provider: str) -> bool:
    """Check if any env var for the provider is set."""
    config = get_config()
    for env_key in PROVIDER_ENV_KEYS.get(provider, []):
        if os.environ.get(env_key) or getattr(config, env_key.lower(), None):
            return True
    return False


# ---------------------------------------------------------------------------
# Hardcoded model lists (fallbacks / static providers)
# ---------------------------------------------------------------------------

# -- OpenAI ------------------------------------------------------------------

FALLBACK_OPENAI_MODELS: list[ModelInfo] = [
    # GPT-5.4 series
    ModelInfo(id="gpt-5.4", provider="openai", label="GPT-5.4"),
    ModelInfo(id="gpt-5.4-pro", provider="openai", label="GPT-5.4 Pro"),
    # GPT-5.2 series
    ModelInfo(id="gpt-5.2", provider="openai", label="GPT-5.2"),
    ModelInfo(id="gpt-5.2-pro", provider="openai", label="GPT-5.2 Pro"),
    # GPT-5.1 series
    ModelInfo(id="gpt-5.1", provider="openai", label="GPT-5.1"),
    # GPT-5 series
    ModelInfo(id="gpt-5", provider="openai", label="GPT-5"),
    ModelInfo(id="gpt-5-pro", provider="openai", label="GPT-5 Pro"),
    ModelInfo(id="gpt-5-mini", provider="openai", label="GPT-5 Mini"),
    ModelInfo(id="gpt-5-nano", provider="openai", label="GPT-5 Nano"),
    # Codex (coding-optimized)
    ModelInfo(id="gpt-5-codex", provider="openai", label="GPT-5 Codex"),
    ModelInfo(id="gpt-5.3-codex", provider="openai", label="GPT-5.3 Codex"),
    ModelInfo(id="gpt-5.2-codex", provider="openai", label="GPT-5.2 Codex"),
    ModelInfo(id="gpt-5.1-codex", provider="openai", label="GPT-5.1 Codex"),
    ModelInfo(id="gpt-5.1-codex-max", provider="openai", label="GPT-5.1 Codex Max"),
    ModelInfo(id="gpt-5.1-codex-mini", provider="openai", label="GPT-5.1 Codex Mini"),
    # GPT-4.1 series
    ModelInfo(id="gpt-4.1", provider="openai", label="GPT-4.1"),
    ModelInfo(id="gpt-4.1-mini", provider="openai", label="GPT-4.1 Mini"),
    ModelInfo(id="gpt-4.1-nano", provider="openai", label="GPT-4.1 Nano"),
    # GPT-4o series
    ModelInfo(id="gpt-4o", provider="openai", label="GPT-4o"),
    ModelInfo(id="gpt-4o-mini", provider="openai", label="GPT-4o Mini"),
    # Reasoning models
    ModelInfo(id="o3-pro", provider="openai", label="o3 Pro"),
    ModelInfo(id="o3", provider="openai", label="o3"),
    ModelInfo(id="o3-mini", provider="openai", label="o3 Mini"),
    ModelInfo(id="o4-mini", provider="openai", label="o4 Mini"),
    ModelInfo(id="o1-pro", provider="openai", label="o1 Pro"),
    ModelInfo(id="o1", provider="openai", label="o1"),
    # Deep Research
    ModelInfo(id="o3-deep-research", provider="openai", label="o3 Deep Research"),
    ModelInfo(id="o4-mini-deep-research", provider="openai", label="o4 Mini Deep Research"),
    # Open-weight
    ModelInfo(id="gpt-oss-120b", provider="openai", label="GPT-OSS 120B"),
    ModelInfo(id="gpt-oss-20b", provider="openai", label="GPT-OSS 20B"),
    # Search
    ModelInfo(id="gpt-4o-search-preview", provider="openai", label="GPT-4o Search"),
    ModelInfo(id="gpt-4o-mini-search-preview", provider="openai", label="GPT-4o Mini Search"),
    # Legacy
    ModelInfo(id="gpt-4.5-preview", provider="openai", label="GPT-4.5 Preview"),
    ModelInfo(id="gpt-4-turbo", provider="openai", label="GPT-4 Turbo"),
    ModelInfo(id="gpt-4", provider="openai", label="GPT-4"),
    ModelInfo(id="gpt-3.5-turbo", provider="openai", label="GPT-3.5 Turbo"),
]

# -- Anthropic ---------------------------------------------------------------

ANTHROPIC_MODELS: list[ModelInfo] = [
    # Current generation
    ModelInfo(id="claude-opus-4-6", provider="anthropic", label="Claude Opus 4.6"),
    ModelInfo(id="claude-sonnet-4-6", provider="anthropic", label="Claude Sonnet 4.6"),
    ModelInfo(id="claude-haiku-4-5-20251001", provider="anthropic", label="Claude Haiku 4.5"),
    # Legacy (still available)
    ModelInfo(id="claude-sonnet-4-5-20250929", provider="anthropic", label="Claude Sonnet 4.5"),
    ModelInfo(id="claude-opus-4-5-20251101", provider="anthropic", label="Claude Opus 4.5"),
    ModelInfo(id="claude-opus-4-1-20250805", provider="anthropic", label="Claude Opus 4.1"),
    ModelInfo(id="claude-sonnet-4-20250514", provider="anthropic", label="Claude Sonnet 4"),
    ModelInfo(id="claude-opus-4-20250514", provider="anthropic", label="Claude Opus 4"),
    # Deprecated (retiring 2026-04-19, still usable)
    ModelInfo(id="claude-3-haiku-20240307", provider="anthropic", label="Claude 3 Haiku (deprecated)"),
]

# -- Google Gemini -----------------------------------------------------------

GEMINI_MODELS: list[ModelInfo] = [
    ModelInfo(id="gemini-3-pro", provider="gemini", label="Gemini 3 Pro"),
    ModelInfo(id="gemini-2.5-pro", provider="gemini", label="Gemini 2.5 Pro"),
    ModelInfo(id="gemini-2.5-flash", provider="gemini", label="Gemini 2.5 Flash"),
    ModelInfo(id="gemini-2.0-flash", provider="gemini", label="Gemini 2.0 Flash"),
    ModelInfo(id="gemini-2.0-flash-lite", provider="gemini", label="Gemini 2.0 Flash Lite"),
]

# -- xAI Grok ----------------------------------------------------------------

XAI_MODELS: list[ModelInfo] = [
    ModelInfo(id="grok-3", provider="xai", label="Grok 3"),
    ModelInfo(id="grok-3-fast", provider="xai", label="Grok 3 Fast"),
    ModelInfo(id="grok-3-mini", provider="xai", label="Grok 3 Mini"),
    ModelInfo(id="grok-3-mini-fast", provider="xai", label="Grok 3 Mini Fast"),
    ModelInfo(id="grok-2", provider="xai", label="Grok 2"),
]

# -- Azure OpenAI (user deploys their own model names) -----------------------

AZURE_MODELS: list[ModelInfo] = [
    # Azure uses deployment names, but these are common defaults
    ModelInfo(id="gpt-4o", provider="azure", label="Azure GPT-4o"),
    ModelInfo(id="gpt-4", provider="azure", label="Azure GPT-4"),
    ModelInfo(id="gpt-4-turbo", provider="azure", label="Azure GPT-4 Turbo"),
    ModelInfo(id="gpt-35-turbo", provider="azure", label="Azure GPT-3.5 Turbo"),
]

# -- Mistral -----------------------------------------------------------------

MISTRAL_MODELS: list[ModelInfo] = [
    ModelInfo(id="mistral-large-latest", provider="mistral", label="Mistral Large"),
    ModelInfo(id="mistral-medium-latest", provider="mistral", label="Mistral Medium"),
    ModelInfo(id="mistral-small-latest", provider="mistral", label="Mistral Small"),
    ModelInfo(id="codestral-latest", provider="mistral", label="Codestral"),
    ModelInfo(id="open-mistral-nemo", provider="mistral", label="Mistral Nemo"),
    ModelInfo(id="open-mixtral-8x22b", provider="mistral", label="Mixtral 8x22B"),
]

# -- DeepSeek ----------------------------------------------------------------

DEEPSEEK_MODELS: list[ModelInfo] = [
    ModelInfo(id="deepseek-chat", provider="deepseek", label="DeepSeek V3"),
    ModelInfo(id="deepseek-reasoner", provider="deepseek", label="DeepSeek R1"),
]

# -- Cohere ------------------------------------------------------------------

COHERE_MODELS: list[ModelInfo] = [
    ModelInfo(id="command-a-03-2025", provider="cohere", label="Command A"),
    ModelInfo(id="command-r-plus-08-2024", provider="cohere", label="Command R+"),
    ModelInfo(id="command-r-08-2024", provider="cohere", label="Command R"),
    ModelInfo(id="command-light", provider="cohere", label="Command Light"),
]

# -- All static providers mapped ---------------------------------------------

_STATIC_PROVIDER_MODELS: dict[str, list[ModelInfo]] = {
    "anthropic": ANTHROPIC_MODELS,
    "gemini":    GEMINI_MODELS,
    "xai":       XAI_MODELS,
    "azure":     AZURE_MODELS,
    "mistral":   MISTRAL_MODELS,
    "deepseek":  DEEPSEEK_MODELS,
    "cohere":    COHERE_MODELS,
}

_PROVIDER_LABELS: dict[str, str] = {
    "openai":    "OpenAI",
    "anthropic": "Anthropic",
    "gemini":    "Google Gemini",
    "xai":       "xAI Grok",
    "azure":     "Azure OpenAI",
    "mistral":   "Mistral AI",
    "deepseek":  "DeepSeek",
    "cohere":    "Cohere",
}


# ---------------------------------------------------------------------------
# OpenAI dynamic fetch
# ---------------------------------------------------------------------------

CHAT_MODEL_PREFIXES = ("gpt-5", "gpt-4", "gpt-4o", "gpt-4.1", "gpt-3.5", "gpt-oss", "o1", "o3", "o4", "chatgpt")
EXCLUDED_KEYWORDS = (
    "instruct", "realtime", "audio", "search", "transcription",
    "tts", "whisper", "dall-e", "embedding",
)


def _is_chat_model(model_id: str) -> bool:
    """Check if a model ID corresponds to a chat model."""
    lower = model_id.lower()
    if any(kw in lower for kw in EXCLUDED_KEYWORDS):
        return False
    return any(lower.startswith(prefix) for prefix in CHAT_MODEL_PREFIXES)


def _model_sort_key(model_id: str) -> int:
    """Sort key for OpenAI models (lower = higher priority)."""
    if model_id.startswith("gpt-5.4"):
        return 0
    if model_id.startswith("gpt-5.2"):
        return 1
    if model_id.startswith("gpt-5.1"):
        return 2
    if model_id.startswith("gpt-5"):
        return 3
    if model_id.startswith("o4"):
        return 6
    if model_id.startswith("o3-pro"):
        return 7
    if model_id.startswith("o3"):
        return 8
    if model_id.startswith("o1"):
        return 9
    if model_id.startswith("gpt-4.1"):
        return 10
    if model_id.startswith("gpt-4o"):
        return 11
    if model_id.startswith("gpt-4"):
        return 12
    if model_id.startswith("gpt-3"):
        return 15
    return 20


async def _fetch_openai_models(api_key: str) -> list[ModelInfo]:
    """Fetch available chat models from the OpenAI API."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    model_ids = [m["id"] for m in data.get("data", []) if _is_chat_model(m["id"])]
    model_ids.sort(key=_model_sort_key)
    return [ModelInfo(id=mid, provider="openai", label=mid) for mid in model_ids]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cached_models: list[ModelInfo] | None = None
_cache_timestamp: float = 0.0
_CACHE_TTL = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_available_models() -> list[ModelInfo]:
    """Return all available models across providers, with caching."""
    global _cached_models, _cache_timestamp

    now = time.monotonic()
    if _cached_models is not None and (now - _cache_timestamp) < _CACHE_TTL:
        return _cached_models

    models: list[ModelInfo] = []

    # OpenAI (dynamic fetch with fallback)
    if _has_provider_key("openai"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        try:
            fetched = await _fetch_openai_models(openai_key)
            models.extend(fetched if fetched else FALLBACK_OPENAI_MODELS)
        except Exception as exc:
            log.debug("openai_model_fetch_failed", error=str(exc))
            models.extend(FALLBACK_OPENAI_MODELS)

    # All static providers - add if API key is present
    for provider_id, provider_models in _STATIC_PROVIDER_MODELS.items():
        if _has_provider_key(provider_id):
            models.extend(provider_models)

    _cached_models = models
    _cache_timestamp = now
    return models


async def get_available_providers() -> list[ProviderInfo]:
    """Return providers grouped with their models."""
    models = await get_available_models()

    # Group by provider
    by_provider: dict[str, list[ModelInfo]] = {}
    for m in models:
        by_provider.setdefault(m.provider, []).append(m)

    return [
        ProviderInfo(
            id=pid,
            label=_PROVIDER_LABELS.get(pid, pid),
            available=True,
            models=pmodels,
        )
        for pid, pmodels in by_provider.items()
    ]


def invalidate_cache() -> None:
    """Clear the model cache so the next call re-fetches."""
    global _cached_models, _cache_timestamp
    _cached_models = None
    _cache_timestamp = 0.0
