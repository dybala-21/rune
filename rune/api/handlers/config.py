"""Read and update the daemon's active configuration."""

import asyncio
import os
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.config.schema import DecisionRoutingConfig
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
    # Let the UI show when approvals are disabled.
    approval_mode: str = Field("standard", alias="approvalMode")
    # The active model's capabilities determine which reasoning controls are shown.
    reasoning_effort: str | None = Field(None, alias="reasoningEffort")
    reasoning_supported: bool = Field(False, alias="reasoningSupported")
    reasoning_options: list[str] = Field(default_factory=list, alias="reasoningOptions")
    reasoning_budgets: dict[str, int] = Field(default_factory=dict, alias="reasoningBudgets")
    decision_routing: dict[str, Any] = Field(default_factory=dict, alias="decisionRouting")

    model_config = ConfigDict(populate_by_name=True)


class ConfigPatchRequest(BaseModel):
    proactive_enabled: bool | None = Field(None, alias="proactiveEnabled")
    active_model: dict[str, str] | None = Field(None, alias="activeModel")
    memory_tuning: dict[str, Any] | None = Field(None, alias="memoryTuning")
    safety_tuning: dict[str, Any] | None = Field(None, alias="safetyTuning")
    advisor_enabled: bool | None = Field(None, alias="advisorEnabled")
    decision_routing: DecisionRoutingConfig | None = Field(None, alias="decisionRouting")

    model_config = ConfigDict(populate_by_name=True)


class ConfigPatchResponse(BaseModel):
    updated: bool


# Routes


def _get_rune_config():
    """Get the live RuneConfig singleton."""
    from rune.config import get_config
    return get_config()


# Only expose memory settings that the runtime reads.
_MEMORY_ENV_KEYS = {
    "uncertainSemanticLimit": "RUNE_MEMORY_UNCERTAIN_SEMANTIC_LIMIT",
    "uncertainSemanticMinScore": "RUNE_MEMORY_UNCERTAIN_SEMANTIC_MIN_SCORE",
    "semanticLimit": "RUNE_MEMORY_SEMANTIC_LIMIT",
    "semanticMinScore": "RUNE_MEMORY_SEMANTIC_MIN_SCORE",
    "maxEpisodes": "RUNE_MEMORY_MAX_EPISODES",
    "contextMaxChars": "RUNE_MEMORY_CONTEXT_MAX_CHARS",
}


def _plan_memory_tuning(tuning: dict[str, Any]) -> list[tuple[str, Any]]:
    """Validate supported tuning fields and return edits without applying them."""
    plan: list[tuple[str, Any]] = []

    for field in _MEMORY_ENV_KEYS:
        value = tuning.get(field)
        if value is not None:
            plan.append((field, value))

    mode = tuning.get("policyMode")
    if mode:
        from rune.memory.rollout_manager import _VALID_MODES

        if mode not in _VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Unknown memory policy mode "{mode}". '
                    f"Valid modes: {', '.join(sorted(_VALID_MODES))}"
                ),
            )
        plan.append(("policyMode", mode))

    return plan


def _apply_memory_tuning(
    plan: list[tuple[str, Any]], tuning: dict[str, Any]
) -> None:
    """Apply a plan produced by :func:`_plan_memory_tuning`."""
    from rune.utils.env import set_env

    scope = tuning.get("scope")
    scope = scope if scope in ("user", "project") else "user"

    for field, value in plan:
        if field == "policyMode":
            from rune.memory.rollout_manager import get_rollout_manager

            get_rollout_manager().set_mode(value)
        else:
            set_env(_MEMORY_ENV_KEYS[field], str(value), scope=scope)


def _memory_tuning_state() -> dict[str, Any]:
    """Read the tuning values currently used by the memory pipeline."""
    from rune.memory.rollout_manager import get_rollout_manager
    from rune.memory.tuning import get_tuning_config

    tuning = get_tuning_config()
    state: dict[str, Any] = {
        "preset": None,
        "policyMode": get_rollout_manager().get_mode(),
    }
    for field, _env in _MEMORY_ENV_KEYS.items():
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", field).lower()
        if snake in tuning:
            state[field] = tuning[snake]
    return state


async def set_reasoning_effort(effort: str, *, provider: str, model: str) -> dict[str, str | None]:
    """Validate against the selected model before persisting the preference."""
    from rune.agent.model_traits import reasoning_efforts
    from rune.config import save_config_values
    from rune.llm.client import loop_model_string
    from rune.llm.model_selection import get_effective_model_selection
    from rune.llm.reasoning import reasoning_model_key

    effective = get_effective_model_selection()
    if provider != effective.provider.value or model != effective.model:
        raise ValueError("The model changed. Select reasoning depth again.")
    resolved = loop_model_string(provider, model)
    options = await asyncio.to_thread(reasoning_efforts, resolved)
    if get_effective_model_selection() != effective:
        raise ValueError("The model changed. Select reasoning depth again.")
    if not isinstance(effort, str) or (effort and effort not in options):
        raise ValueError(f"{effective.model} supports: {', '.join(options) or 'provider default only'}")
    cfg = _get_rune_config()
    preferences = {**cfg.llm.reasoning_efforts, reasoning_model_key(resolved): effort or None}
    if save_config_values({
        "llm.reasoningEfforts": preferences, "llm.reasoningEffort": None, "llm.reasoning_effort": None,
    }) is None:
        raise OSError("Could not save reasoning settings. Please try again.")
    cfg.llm.reasoning_efforts = preferences
    cfg.llm.reasoning_effort = None
    return {"reasoningEffort": effort or None}


@router.get("", response_model=ConfigGetResponse, dependencies=[Depends(auth)])
async def get_config_endpoint() -> ConfigGetResponse:
    """Retrieve the current daemon configuration."""
    from rune.agent.advisor.runtime_toggle import is_advisor_enabled
    from rune.agent.tool_adapter import approval_mode
    from rune.llm.client import loop_model_string
    from rune.llm.model_selection import get_effective_model_selection
    from rune.llm.reasoning import reasoning_control, reasoning_model_key

    cfg = _get_rune_config()
    effective = get_effective_model_selection()
    from rune.agent.decision_router import accelerator_status
    from rune.utils.env import effective_env_scope

    decision_status = accelerator_status(effective) if cfg.llm.decision_routing.backend == "jev" else "disabled"
    model = loop_model_string(effective.provider.value, effective.model)
    control = await asyncio.to_thread(reasoning_control, model)
    options = control.efforts
    effort = cfg.llm.reasoning_efforts.get(reasoning_model_key(model))
    return ConfigGetResponse(
        proactiveEnabled=cfg.proactive.enabled,
        gatewayChannels=["api"],
        maxConcurrency=3,
        version=VERSION,
        advisorEnabled=is_advisor_enabled(),
        approvalMode=approval_mode(),
        # The active selection takes precedence over provider defaults.
        activeModel={
            "provider": effective.provider.value,
            "model": effective.model,
            "source": "active" if cfg.llm.active_model else "default",
        },
        reasoningEffort=effort if effort in options else None,
        reasoningSupported=bool(options),
        reasoningOptions=list(options),
        reasoningBudgets=dict(control.budgets),
        decisionRouting={
            **cfg.llm.decision_routing.model_dump(by_alias=True),
            "status": decision_status,
            "effectiveBackend": "jev" if decision_status in {"ready", "unverified"} else "connected",
            "hasKey": bool(os.environ.get("TYPESAFE_API_KEY", "").strip()),
            "keyScope": effective_env_scope("TYPESAFE_API_KEY"),
        },
        memoryTuning=_memory_tuning_state(),
        safetyTuning={
            "preset": None,
            "rolloutMode": cfg.safety.rollout_mode,
            # The shell gate builds its own policy and never reads this, so
            # reporting it as active would overstate what the setting does.
            "autoEnabled": False,
        },
    )


@router.patch("", response_model=ConfigPatchResponse, dependencies=[Depends(auth)])
async def patch_config(req: ConfigPatchRequest) -> ConfigPatchResponse:
    """Update daemon configuration.

    Only the provided fields are updated. Omitted fields remain unchanged.
    """
    cfg = _get_rune_config()

    # Validate all fields before applying changes from this request.
    if req.safety_tuning is not None:
        log.info("config_patch_rejected", field="safetyTuning", value=req.safety_tuning)
        raise HTTPException(
            status_code=501,
            detail=(
                "Safety presets are not wired to the execution policy yet — "
                "the shell gate reads a fixed policy, so a preset here would "
                "have no effect."
            ),
        )

    provider = model = None
    if req.active_model is not None:
        provider = req.active_model.get("provider")
        model = req.active_model.get("model")
        if provider:
            from rune.types import Provider

            try:
                Provider(provider)
            except ValueError as exc:
                known = ", ".join(sorted(p.value for p in Provider))
                raise HTTPException(
                    status_code=400,
                    detail=f'Unknown provider "{provider}". Known providers: {known}',
                ) from exc

    memory_plan: list[tuple[str, Any]] = []
    if req.memory_tuning is not None:
        memory_plan = _plan_memory_tuning(req.memory_tuning)
        if not memory_plan:
            raise HTTPException(
                status_code=400,
                detail="No memory tuning field in this request is connected to anything",
            )

    routing = None
    routing_updates = req.decision_routing.model_dump(exclude_unset=True) if req.decision_routing is not None else {}
    if routing_updates:
        routing = cfg.llm.decision_routing.model_copy(update=routing_updates)
        if routing_updates.get("backend") == "jev" and not os.environ.get("TYPESAFE_API_KEY", "").strip():
            raise HTTPException(status_code=400, detail="Add a TypeSafe API key before enabling Jev.")

    to_persist: dict[str, Any] = {}
    if routing is not None:
        to_persist.update({f"llm.decisionRouting.{key}": value for key, value in
                           req.decision_routing.model_dump(by_alias=True, exclude_unset=True).items()})
    if req.proactive_enabled is not None:
        to_persist["proactive.enabled"] = req.proactive_enabled
    if provider:
        to_persist.update({"llm.defaultProvider": provider, "llm.activeProvider": provider})
    if model:
        to_persist.update({"llm.defaultModel": model, "llm.activeModel": model})
    if provider or model:
        to_persist.update({"llm.reasoningEfforts": cfg.llm.reasoning_efforts,
                           "llm.reasoningEffort": None, "llm.reasoning_effort": None})
    if to_persist:
        from rune.config import save_config_values

        if save_config_values(to_persist) is None:
            raise HTTPException(status_code=500, detail="Could not save settings. No configuration changes were applied.")

    updated = routing is not None
    if routing is not None:
        cfg.llm.decision_routing = routing

    if req.proactive_enabled is not None:
        cfg.proactive.enabled = req.proactive_enabled
        log.info("config_patch", field="proactiveEnabled", value=req.proactive_enabled)
        updated = True

    if provider:
        cfg.llm.default_provider = provider
        cfg.llm.active_provider = provider
        updated = True
    if model:
        cfg.llm.default_model = model
        cfg.llm.active_model = model
        updated = True
    if req.active_model is not None:
        log.info("config_patch", field="activeModel", provider=provider, model=model)

    if memory_plan:
        _apply_memory_tuning(memory_plan, req.memory_tuning or {})
        log.info("config_patch", field="memoryTuning", applied=[k for k, _ in memory_plan])
        updated = True

    if req.advisor_enabled is not None:
        from rune.agent.advisor.runtime_toggle import set_advisor_enabled
        set_advisor_enabled(req.advisor_enabled)
        log.info("config_patch", field="advisorEnabled", value=req.advisor_enabled)
        updated = True

    return ConfigPatchResponse(updated=updated)
