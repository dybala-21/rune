"""Provider-native search support map for RUNE.

Ported from src/capabilities/search/provider-search-map.ts - records
which LLM providers offer server-side web search and their cost info.
"""

from __future__ import annotations

from typing import TypedDict


class ProviderSearchConfig(TypedDict, total=False):
    """Configuration entry for a provider's native search capability."""
    has_native_search: bool
    tool_factory: str
    package_name: str
    base_cost_per_1k: float       # $/1K searches (base)
    effective_cost_per_1k: float   # $/1K searches (with token overhead)


#: Map of provider name -> native search configuration.
PROVIDER_SEARCH_MAP: dict[str, ProviderSearchConfig] = {
    # Providers with server-side search (sorted by cost)
    "xai": {
        "has_native_search": True,
        "tool_factory": "xai.tools.webSearch",
        "package_name": "@ai-sdk/xai",
        "base_cost_per_1k": 5,
        "effective_cost_per_1k": 6,
    },
    "groq": {
        "has_native_search": True,
        "tool_factory": "groq.tools.browserSearch",
        "package_name": "@ai-sdk/groq",
        "base_cost_per_1k": 7,
        "effective_cost_per_1k": 8,
    },
    "anthropic": {
        "has_native_search": True,
        "tool_factory": "anthropic.tools.webSearch_20250305",
        "package_name": "@ai-sdk/anthropic",
        "base_cost_per_1k": 10,
        "effective_cost_per_1k": 20,
    },
    "google": {
        "has_native_search": True,
        "tool_factory": "google.tools.googleSearch",
        "package_name": "@ai-sdk/google",
        "base_cost_per_1k": 14,
        "effective_cost_per_1k": 14,
    },
    "openai": {
        "has_native_search": True,
        "tool_factory": "openai.tools.webSearch",
        "package_name": "@ai-sdk/openai",
        "base_cost_per_1k": 10,
        "effective_cost_per_1k": 30,
    },
    "azure": {
        "has_native_search": True,
        "tool_factory": "azure.tools.webSearchPreview",
        "package_name": "@ai-sdk/azure",
        "base_cost_per_1k": 35,
        "effective_cost_per_1k": 40,
    },
    # Providers without server-side search (client-side fallback)
    "ollama": {"has_native_search": False},
    "deepseek": {"has_native_search": False},
    "together": {"has_native_search": False},
    "fireworks": {"has_native_search": False},
    "lmstudio": {"has_native_search": False},
    "mistral": {"has_native_search": False},
    "cohere": {"has_native_search": False},
}


def get_native_search_providers() -> list[str]:
    """Return provider names that support native search, sorted by cost."""
    providers = [
        (name, cfg.get("effective_cost_per_1k", float("inf")))
        for name, cfg in PROVIDER_SEARCH_MAP.items()
        if cfg.get("has_native_search")
    ]
    providers.sort(key=lambda x: x[1])
    return [name for name, _ in providers]
