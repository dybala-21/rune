"""Model selection for RUNE.

Ported from src/llm/model-selection.ts - active model selection,
persistence, and effective model resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from rune.config import get_config
from rune.types import Provider
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Types

@dataclass(slots=True)
class ActiveModelSelection:
    """Currently selected provider + model pair."""
    provider: Provider
    model: str


# Selection helpers

def get_active_model_selection() -> ActiveModelSelection | None:
    """Return the user's explicit model override, or ``None``."""
    config = get_config()
    llm_cfg = config.llm

    provider = getattr(llm_cfg, "active_provider", None)
    model = (getattr(llm_cfg, "active_model", None) or "").strip()

    if not provider or not model:
        return None

    return ActiveModelSelection(provider=Provider(provider), model=model)


def get_effective_model_selection() -> ActiveModelSelection:
    """Return the active selection or fall back to the default provider/model."""
    active = get_active_model_selection()
    if active is not None:
        return active

    config = get_config()
    llm_cfg = config.llm
    provider = Provider(llm_cfg.default_provider)

    # Resolve default model for the provider
    provider_models = getattr(llm_cfg.models, provider.value, None)
    default_model = getattr(provider_models, "best", "unknown") if provider_models else "unknown"

    return ActiveModelSelection(provider=provider, model=default_model)


async def persist_active_model_selection(selection: ActiveModelSelection) -> ActiveModelSelection:
    """Persist the active model selection to config and reset the LLM client."""
    from rune.config import get_config_loader

    loader = get_config_loader()
    config = await loader.load()

    config.llm.active_provider = selection.provider.value  # type: ignore[attr-defined]
    config.llm.active_model = selection.model  # type: ignore[attr-defined]
    await loader.save(config)

    # Reset the singleton client so it picks up the new selection
    from rune.llm.client import get_llm_client
    client = get_llm_client()
    client._initialized = False  # noqa: SLF001

    log.info("model_selection_persisted", provider=selection.provider, model=selection.model)
    return selection


async def clear_active_model_selection() -> ActiveModelSelection:
    """Clear the active override and return the effective (default) selection."""
    from rune.config import get_config_loader

    loader = get_config_loader()
    config = await loader.load()

    config.llm.active_provider = None  # type: ignore[attr-defined]
    config.llm.active_model = None  # type: ignore[attr-defined]
    await loader.save(config)

    # Reset client
    from rune.llm.client import get_llm_client
    client = get_llm_client()
    client._initialized = False  # noqa: SLF001

    return get_effective_model_selection()
