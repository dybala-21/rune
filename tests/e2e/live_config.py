"""Resolve model credentials after Rune has loaded its configuration."""

import os
from pathlib import Path

from rune.llm.models import PROVIDER_ENV_KEYS


def select_model(cfg, provider=None, model=None):
    provider = provider or cfg.llm.active_provider or cfg.llm.default_provider
    tiers = getattr(cfg.llm.models, provider, None)
    if tiers is None:
        raise ValueError(f"Unsupported E2E provider: {provider}")
    available = provider == "ollama" or any(
        os.environ.get(key) or getattr(cfg, key.lower(), None)
        for key in PROVIDER_ENV_KEYS.get(provider, [])
    )
    if provider == "gemini":
        credentials = cfg.google_credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        available |= bool(credentials and Path(credentials).is_file())
    if not available:
        raise ValueError(f"No credentials configured for the requested E2E provider: {provider}")
    active = cfg.llm.active_model if cfg.llm.active_provider == provider else None
    return provider, model or active or tiers.best


def configure_model(cfg, provider, model):
    cfg.llm.active_provider = cfg.llm.default_provider = provider
    cfg.llm.active_model = cfg.llm.default_model = model
    tiers = getattr(cfg.llm.models, provider)
    tiers.best = tiers.fast = tiers.coding = model
