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


def persist_active_model_selection(selection: ActiveModelSelection) -> ActiveModelSelection:
    """Set the active model and write it to config.yaml."""
    llm_cfg = get_config().llm
    llm_cfg.active_provider = selection.provider.value
    llm_cfg.active_model = selection.model

    from rune.config import save_config_values
    save_config_values({
        "llm.activeProvider": selection.provider.value,
        "llm.activeModel": selection.model,
    })

    _reset_llm_client()
    log.info(
        "model_selection_persisted",
        provider=selection.provider.value,
        model=selection.model,
    )
    return selection


def clear_active_model_selection() -> ActiveModelSelection:
    """Drop the override and fall back to the configured default."""
    llm_cfg = get_config().llm
    llm_cfg.active_provider = None
    llm_cfg.active_model = None

    from rune.config import save_config_values
    save_config_values({"llm.activeProvider": None, "llm.activeModel": None})

    _reset_llm_client()
    log.info("model_selection_cleared")
    return get_effective_model_selection()


def _reset_llm_client() -> None:
    """Force the shared client to re-resolve its provider on next use."""
    from rune.llm.client import get_llm_client

    get_llm_client()._initialized = False  # noqa: SLF001
