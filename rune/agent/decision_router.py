"""Task routing with an optional decision model."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rune.agent.provenance import ArtifactRoleHints
from rune.llm.context import llm_settings, model_environment
from rune.llm.decision_backends import DecisionTarget, decision_target
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.config.schema import DecisionRoutingConfig
    from rune.llm.model_selection import ActiveModelSelection

log = get_logger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    values: dict[str, Any]
    backend: str
    model: str
    fallback_reason: str = ""
    artifact_roles: ArtifactRoleHints | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass
class DecisionHealth:
    verified: bool = False
    rejected: bool = False
    cooldown_until: float = 0


_health: dict[str, DecisionHealth] = {}


def _key_id(target: DecisionTarget) -> str:
    identity = (target.backend, target.endpoint, target.deployment, target.model, model_environment(target.key_name))
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def accelerator_status(selected: ActiveModelSelection | None = None, *,
                       settings: DecisionRoutingConfig | None = None, backend: str | None = None) -> str:
    from rune.llm.endpoints import self_hosted
    from rune.llm.model_selection import get_effective_model_selection

    settings = settings or llm_settings().decision_routing
    backend = backend or settings.backend
    if backend == "connected":
        return "disabled"
    target = decision_target(settings, backend)
    if not model_environment(target.key_name).strip():
        return "missing_key"
    if not target.endpoint or not target.deployment:
        return "missing_config"
    selected = selected or get_effective_model_selection()
    if self_hosted(selected.provider, selected.model):
        return "local"
    llm = llm_settings()
    # Automatic model selection already runs its own classifier.
    if llm.route_simple_queries and not (llm.active_model or "").strip():
        return "automatic_model"
    state = _health.get(_key_id(target), DecisionHealth())
    if state.rejected:
        return "auth_error"
    if time.monotonic() < state.cooldown_until:
        return "cooldown"
    return "ready" if state.verified else "unverified"


def _remember(key: str, state: DecisionHealth) -> None:
    _health[key] = state
    while len(_health) > 32:
        del _health[next(iter(_health))]


async def classify_request(system: str, content: str) -> RoutingDecision:
    from rune.agent import classification_response as contract
    from rune.llm.client import get_llm_client
    from rune.llm.model_selection import get_effective_model_selection

    selected = get_effective_model_selection()
    settings = llm_settings().decision_routing.model_copy()
    # Routing and fallback share the same time budget.
    deadline = asyncio.get_running_loop().time() + contract.ROUTING_TIMEOUT
    reason = ""
    artifact_roles = None
    diagnostics = {}
    target = None
    if settings.backend != "connected":
        target = decision_target(settings)
        reason = accelerator_status(selected, settings=settings)
        if reason in {"ready", "unverified"}:
            from rune.llm.system_one import DecisionAbstained, DecisionUnavailable, classify

            api_key = model_environment(target.key_name)
            key_id = _key_id(target)
            try:
                timeout = min(settings.timeout_ms / 1000, max(0.001, deadline - asyncio.get_running_loop().time()))
                async with asyncio.timeout(timeout):
                    batch = await classify(system, content, timeout=timeout, api_key=api_key, target=target)
                _remember(key_id, DecisionHealth(verified=True))
                artifact_roles = batch.artifact_roles
                diagnostics = batch.diagnostics
                if batch.values is not None:
                    return RoutingDecision(contract.validate_decision(batch.values), target.backend, target.model,
                                           artifact_roles=artifact_roles, diagnostics=diagnostics)
                reason = batch.fallback_reason
            except DecisionAbstained as exc:
                reason = exc.reason
                if reason != "large_request":
                    _remember(key_id, DecisionHealth(verified=True))
            except (DecisionUnavailable, TimeoutError) as exc:
                reason = exc.reason if isinstance(exc, DecisionUnavailable) else "timeout"
                _remember(key_id, DecisionHealth(rejected=reason in {"http_401", "http_403"},
                                                 cooldown_until=time.monotonic() + 30))
            log.info("decision_fallback", backend=target.backend, reason=reason, **diagnostics)

    request = json.loads(content).get("request_to_classify", "")
    roles = artifact_roles.matching_roles(request) if artifact_roles is not None else {}
    try:
        values = await contract.request_classification(
            get_llm_client(), system, content, selected=selected, deadline=deadline,
            file_roles=roles,
        )
    except Exception as exc:
        if not reason:
            raise
        from rune.agent.request_budget import BudgetExceeded
        from rune.llm.failures import request_failure
        from rune.llm.target import ModelCompatibilityError

        detail = str(exc) if isinstance(exc, (BudgetExceeded, contract.InvalidClassification, ModelCompatibilityError)) else request_failure(exc).kind
        raise contract.RoutingUnavailable(
            f"{target.label if target else 'Decision'} routing unavailable ({reason}); connected fallback stopped: {detail}",
            fallback_reason=reason, diagnostics=diagnostics,
        ) from exc
    if roles:
        artifact_roles = ArtifactRoleHints.for_request(request, roles)
    return RoutingDecision(values, "connected", selected.model, reason, artifact_roles, diagnostics)
