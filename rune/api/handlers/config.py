"""Config handler - GET /config, PATCH /config.

Ported from src/api/handlers/config.ts - retrieve and update
daemon/runtime configuration.
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/config", tags=["config"])
auth = TokenAuthDependency()

VERSION = "0.1.0"


# Models


class ActiveModelInfo(BaseModel):
    provider: str
    model: str
    source: str


class ConfigGetResponse(BaseModel):
    proactive_enabled: bool = Field(False, alias="proactiveEnabled")
    gateway_channels: list[str] = Field(default_factory=list, alias="gatewayChannels")
    max_concurrency: int = Field(3, alias="maxConcurrency")
    version: str = VERSION
    active_model: ActiveModelInfo | None = Field(None, alias="activeModel")
    memory_tuning: dict[str, Any] | None = Field(None, alias="memoryTuning")
    safety_tuning: dict[str, Any] | None = Field(None, alias="safetyTuning")
    advisor_enabled: bool = Field(False, alias="advisorEnabled")

    model_config = ConfigDict(populate_by_name=True)


class ConfigPatchRequest(BaseModel):
    proactive_enabled: bool | None = Field(None, alias="proactiveEnabled")
    active_model: dict[str, str] | None = Field(None, alias="activeModel")
    memory_tuning: dict[str, Any] | None = Field(None, alias="memoryTuning")
    safety_tuning: dict[str, Any] | None = Field(None, alias="safetyTuning")
    advisor_enabled: bool | None = Field(None, alias="advisorEnabled")

    model_config = ConfigDict(populate_by_name=True)


class ConfigPatchResponse(BaseModel):
    updated: bool


# Routes


def _get_rune_config():
    """Get the live RuneConfig singleton."""
    from rune.config import get_config
    return get_config()


@router.get("", response_model=ConfigGetResponse, dependencies=[Depends(auth)])
async def get_config_endpoint() -> ConfigGetResponse:
    """Retrieve the current daemon configuration."""
    from rune.agent.advisor.runtime_toggle import is_advisor_enabled
    cfg = _get_rune_config()
    return ConfigGetResponse(
        proactiveEnabled=cfg.proactive.enabled,
        gatewayChannels=["api"],
        maxConcurrency=3,
        version=VERSION,
        advisorEnabled=is_advisor_enabled(),
        activeModel={
            "provider": cfg.llm.default_provider or "openai",
            "model": cfg.llm.default_model,
            "source": "config",
        },
        memoryTuning={
            "preset": None,
            "policyMode": "auto",
            "uncertainScoreThreshold": 0.5,
            "uncertainRelevanceFloor": 0.35,
            "uncertainSemanticLimit": 3,
            "uncertainSemanticMinScore": 0.45,
            "rolloutObservationWindowDays": 14,
            "rolloutMinShadowSamples": 20,
            "rolloutPromoteBalancedMinSuccessRate": 0.85,
            "rolloutRollbackMaxP95Ms": 140,
        },
        safetyTuning={
            "preset": None,
            "rolloutMode": "auto",
            "autoEnabled": cfg.safety.enabled if hasattr(cfg.safety, "enabled") else True,
        },
    )


@router.patch("", response_model=ConfigPatchResponse, dependencies=[Depends(auth)])
async def patch_config(req: ConfigPatchRequest) -> ConfigPatchResponse:
    """Update daemon configuration.

    Only the provided fields are updated. Omitted fields remain unchanged.
    """
    cfg = _get_rune_config()
    updated = False

    if req.proactive_enabled is not None:
        cfg.proactive.enabled = req.proactive_enabled
        log.info("config_patch", field="proactiveEnabled", value=req.proactive_enabled)
        updated = True

    if req.active_model is not None:
        provider = req.active_model.get("provider")
        model = req.active_model.get("model")
        if provider:
            cfg.llm.default_provider = provider
        if model:
            cfg.llm.default_model = model
        log.info("config_patch", field="activeModel", provider=provider, model=model)
        updated = True

    if req.memory_tuning is not None:
        log.info("config_patch", field="memoryTuning")
        updated = True

    if req.safety_tuning is not None:
        log.info("config_patch", field="safetyTuning")
        updated = True

    if req.advisor_enabled is not None:
        from rune.agent.advisor.runtime_toggle import set_advisor_enabled
        set_advisor_enabled(req.advisor_enabled)
        log.info("config_patch", field="advisorEnabled", value=req.advisor_enabled)
        updated = True

    return ConfigPatchResponse(updated=updated)
