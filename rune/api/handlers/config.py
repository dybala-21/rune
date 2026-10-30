"""Read and update the daemon's active configuration."""

import asyncio
import os
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.config.schema import (
    AzureDeployments,
    AzurePricingConfig,
    CompatibleEndpoints,
    DecisionRoutingConfig,
)
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


class RunLimits(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", strict=True)

    model_requests: int = Field(100, ge=0, alias="modelRequests")
    cost_usd: float | None = Field(None, ge=0, allow_inf_nan=False, alias="costUsd")


class ConfigGetResponse(BaseModel):
    azure: dict[str, Any] = Field(default_factory=dict)
    local_models: dict[str, Any] = Field(default_factory=dict, alias="localModels")
    vertex_location: str | None = Field(None, alias="vertexLocation")
    run_limits: RunLimits = Field(default_factory=RunLimits, alias="runLimits")
    execution_environment: dict[str, Any] = Field(default_factory=dict, alias="executionEnvironment")
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


class ExecutionEnvironmentPatch(BaseModel):
    backend: Literal["local", "container"] | None = None
    image: str | None = Field(None, min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/:@-]*$")


class ConfigPatchRequest(BaseModel):
    azure_deployments: AzureDeployments | None = Field(None, alias="azureDeployments")
    endpoints: CompatibleEndpoints | None = None
    azure_pricing: AzurePricingConfig | None = Field(None, alias="azurePricing")
    vertex_location: str | None = Field(None, alias="vertexLocation", min_length=1, max_length=64,
                                       pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    run_limits: RunLimits | None = Field(None, alias="runLimits")
    execution_environment: ExecutionEnvironmentPatch | None = Field(None, alias="executionEnvironment")
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
    from rune.cloud.boundary import hosted
    from rune.llm.azure import (
        ENDPOINT_NAMES,
        GPT6_MODELS,
        KEY_NAMES,
        endpoint,
        environment_name,
    )
    from rune.llm.context import model_environment
    from rune.llm.decision_backends import BACKENDS
    from rune.llm.target import vertex_active
    from rune.safety.execution_environment import execution_config
    from rune.utils.env import effective_env_scope

    environment = execution_config()
    decision_settings = cfg.llm.decision_routing
    decision_status = accelerator_status(effective, settings=decision_settings)
    decision_connections = {
        name: {"hasKey": bool(model_environment(target.key_name).strip()),
               "keyScope": effective_env_scope(target.key_name),
               "status": accelerator_status(effective, settings=decision_settings, backend=name)}
        for name, target in BACKENDS.items()
    }
    selected_connection = decision_connections.get(decision_settings.backend, {})
    model = loop_model_string(effective.provider.value, effective.model)
    control = await asyncio.to_thread(reasoning_control, model)
    options = control.efforts
    effort = cfg.llm.reasoning_efforts.get(reasoning_model_key(model))
    return ConfigGetResponse(
        azure={
            "endpoint": endpoint(), "endpointKey": environment_name(ENDPOINT_NAMES),
            "endpointScope": effective_env_scope(environment_name(ENDPOINT_NAMES)),
            "hasKey": bool(model_environment(environment_name(KEY_NAMES)).strip()),
            "keyName": environment_name(KEY_NAMES),
            "keyScope": effective_env_scope(environment_name(KEY_NAMES)),
            "deployments": cfg.llm.azure_deployments, "suggestedModels": GPT6_MODELS,
            "pricing": cfg.llm.azure_pricing.model_dump() if cfg.llm.azure_pricing else None,
        },
        localModels=_local_models_state(cfg),
        vertexLocation=(os.environ.get("VERTEX_LOCATION") or os.environ.get("VERTEXAI_LOCATION")
                        or cfg.vertex_location) if vertex_active() else None,
        runLimits=RunLimits(modelRequests=cfg.llm.model_request_limit, costUsd=cfg.llm.cost_budget_usd),
        executionEnvironment={"backend": environment.backend, "image": environment.image,
                              "allowNetwork": environment.allow_network, "managed": hosted()},
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
            "effectiveBackend": decision_settings.backend if decision_status in {"ready", "unverified"} else "connected",
            "hasKey": selected_connection.get("hasKey", False),
            "keyScope": selected_connection.get("keyScope"),
            "providers": decision_connections,
        },
        memoryTuning=_memory_tuning_state(),
        safetyTuning={
            "preset": None,
            "rolloutMode": cfg.safety.rollout_mode,
            # The shell gate does not use this setting, so do not report it as active.
            "autoEnabled": False,
        },
    )


@router.patch("", response_model=ConfigPatchResponse, dependencies=[Depends(auth)])
async def patch_config(req: ConfigPatchRequest) -> ConfigPatchResponse:
    """Update supplied configuration fields, leaving omitted fields unchanged."""
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

    if provider or model or req.vertex_location is not None:
        from rune.llm.model_selection import get_effective_model_selection
        from rune.llm.target import ModelCompatibilityError, resolve_target
        selected = get_effective_model_selection()
        try:
            resolve_target(model or selected.model, provider or selected.provider,
                           options={"vertex_location": req.vertex_location} if req.vertex_location else None)
        except ModelCompatibilityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        from rune.llm.context import model_environment
        from rune.llm.decision_backends import decision_target

        merged = {**cfg.llm.decision_routing.model_dump(), **routing_updates}
        if "microsoft" in routing_updates:
            merged["microsoft"] = {**cfg.llm.decision_routing.microsoft.model_dump(), **routing_updates["microsoft"]}
        routing = DecisionRoutingConfig.model_validate(merged)
        if routing.backend != "connected":
            target = decision_target(routing)
            if not model_environment(target.key_name).strip():
                raise HTTPException(status_code=400, detail=f"Add an API key before enabling {target.label}.")
            if not target.endpoint or not target.deployment:
                raise HTTPException(status_code=400, detail="Add the Azure resource endpoint and deployment name.")

    to_persist: dict[str, Any] = {}
    if req.azure_deployments is not None:
        to_persist.update({"llm.azureDeployments": req.azure_deployments, "llm.azure_deployments": None})
    if "azure_pricing" in req.model_fields_set:
        to_persist.update({"llm.azurePricing": req.azure_pricing.model_dump() if req.azure_pricing else None,
                           "llm.azure_pricing": None})
    if req.endpoints is not None:
        to_persist["llm.endpoints"] = {name: entry.model_dump(by_alias=True, exclude_none=True)
                                       for name, entry in req.endpoints.items()}
    if req.vertex_location is not None:
        to_persist["vertex_location"] = req.vertex_location
    limits = req.run_limits.model_dump(exclude_unset=True) if req.run_limits is not None else {}
    limit_fields = {"model_requests": ("model_request_limit", "modelRequestLimit"),
                    "cost_usd": ("cost_budget_usd", "costBudgetUsd")}
    for name, value in limits.items():
        field, alias = limit_fields[name]
        # Clear the alternate spelling so resetting a limit survives reloads.
        to_persist.update({f"llm.{alias}": value, f"llm.{field}": None})
    environment = req.execution_environment.model_dump(exclude_none=True) if req.execution_environment else {}
    from rune.cloud.boundary import hosted, sandbox
    if hosted() and any(value != getattr(sandbox(), key) for key, value in environment.items()):
        raise HTTPException(403, "Hosted execution settings are managed by the operator")
    to_persist.update({f"safety.sandbox.{key}": value for key, value in environment.items()})
    if routing is not None:
        for key, value in req.decision_routing.model_dump(by_alias=True, exclude_unset=True).items():
            if key == "microsoft":
                to_persist.update({f"llm.decisionRouting.microsoft.{name}": field for name, field in value.items()})
            else:
                to_persist[f"llm.decisionRouting.{key}"] = value
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

    updated = routing is not None or bool(environment) or bool(limits)
    if "azure_pricing" in req.model_fields_set:
        cfg.llm.azure_pricing = req.azure_pricing
        updated = True
    if req.azure_deployments is not None:
        cfg.llm.azure_deployments = req.azure_deployments
        from rune.llm.models import invalidate_cache
        invalidate_cache()
        updated = True
    if req.endpoints is not None:
        cfg.llm.endpoints = req.endpoints
        from rune.llm.models import invalidate_cache
        invalidate_cache()
        updated = True
    if req.vertex_location is not None:
        cfg.vertex_location = req.vertex_location
        os.environ["VERTEX_LOCATION"] = req.vertex_location
        updated = True
    for name, value in limits.items():
        setattr(cfg.llm, limit_fields[name][0], value)
    if environment:
        cfg.safety.sandbox = cfg.safety.sandbox.model_copy(update=environment)
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


def _local_models_state(cfg) -> dict[str, Any]:
    from rune.llm.context import model_environment
    from rune.llm.endpoints import OLLAMA_NAMES, is_free, key_name, ollama_root
    from rune.utils.env import effective_env_scope

    ollama_key = next((name for name in OLLAMA_NAMES if model_environment(name).strip()), OLLAMA_NAMES[0])
    return {
        "ollama": {"baseUrl": ollama_root(), "envKey": ollama_key, "scope": effective_env_scope(ollama_key)},
        "endpoints": {
            name: {**entry.model_dump(by_alias=True), "keyName": key_name(name),
                   "hasKey": bool(model_environment(key_name(name)).strip()),
                   "keyScope": effective_env_scope(key_name(name)), "selfHosted": is_free(name)}
            for name, entry in cfg.llm.endpoints.items()
        },
    }


class EndpointTestRequest(BaseModel):
    model: str | None = Field(None, min_length=1, max_length=200)


@router.post("/endpoints/{name}/test", dependencies=[Depends(auth)])
async def test_endpoint(name: str, req: EndpointTestRequest) -> dict[str, Any]:
    """List the server's models and send one tool request, which may be billed."""
    from rune.llm.endpoints import configured, diagnose

    if name not in configured():
        raise HTTPException(status_code=404, detail=f'No endpoint named "{name}". Save it first.')
    result = await diagnose(name, req.model)
    log.info("endpoint_tested", endpoint=name, reachable=result["reachable"], tools=result["tools"])
    return result
