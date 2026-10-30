"""Select a decision backend without changing the model that does the work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from rune.agent.provenance import ArtifactRoleHints
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    values: dict[str, Any]
    backend: str
    model: str
    fallback_reason: str = ""
    artifact_roles: ArtifactRoleHints | None = None
    diagnostics: dict = field(default_factory=dict)


# Pause on service failures, not on uncertainty about an individual request.
_cooldown: tuple[str, float] | None = None
_rejected_key: str | None = None
_verified_key: str | None = None


def _key_id() -> str:
    return hashlib.sha256(os.environ.get("TYPESAFE_API_KEY", "").encode()).hexdigest()


def accelerator_status(selected=None) -> str:
    from rune.llm.context import llm_settings
    from rune.llm.endpoints import self_hosted
    from rune.llm.model_selection import get_effective_model_selection

    selected = selected or get_effective_model_selection()
    if self_hosted(selected.provider, selected.model):
        return "local"
    llm = llm_settings()
    # Automatic model selection keeps its existing classifier and cost policy.
    if llm.route_simple_queries and not (llm.active_model or "").strip():
        return "automatic_model"
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return "missing_key"
    key_id = _key_id()
    if _rejected_key == key_id:
        return "auth_error"
    if _cooldown and _cooldown[0] == key_id and time.monotonic() < _cooldown[1]:
        return "cooldown"
    return "ready" if _verified_key == key_id else "unverified"


async def classify_request(system: str, content: str) -> RoutingDecision:
    from rune.agent import classification_response as contract
    from rune.llm.client import get_llm_client
    from rune.llm.model_selection import get_effective_model_selection

    selected = get_effective_model_selection()
    from rune.llm.context import llm_settings
    settings = llm_settings().decision_routing.model_copy()
    # Jev and the fallback share one deadline.
    deadline = asyncio.get_running_loop().time() + contract.ROUTING_TIMEOUT
    reason = ""
    artifact_roles = None
    diagnostics = {}
    if settings.backend == "jev":
        reason = accelerator_status(selected)
        if reason in {"ready", "unverified"}:
            from rune.llm.jev import MODEL, DecisionAbstained, JevUnavailable, classify

            global _cooldown, _rejected_key, _verified_key
            # A request using an old key must not change the replacement key's status.
            api_key = os.environ["TYPESAFE_API_KEY"]
            key_id = hashlib.sha256(api_key.encode()).hexdigest()
            try:
                timeout = min(settings.timeout_ms / 1000, max(0.001, deadline - asyncio.get_running_loop().time()))
                async with asyncio.timeout(timeout):
                    batch = await classify(system, content, timeout=timeout, api_key=api_key)
                if _key_id() == key_id:
                    _verified_key = key_id
                artifact_roles = batch.artifact_roles
                diagnostics = batch.diagnostics
                if batch.values is not None:
                    return RoutingDecision(contract.validate_decision(batch.values), "jev", MODEL,
                                           artifact_roles=artifact_roles, diagnostics=diagnostics)
                reason = batch.fallback_reason
            except DecisionAbstained as exc:
                reason = exc.reason
                if reason != "large_request" and _key_id() == key_id:
                    _verified_key = key_id
            except (JevUnavailable, TimeoutError) as exc:
                reason = exc.reason if isinstance(exc, JevUnavailable) else "timeout"
                if _key_id() == key_id:
                    if reason in {"http_401", "http_403"}:
                        _rejected_key = key_id
                    else:
                        _cooldown = (key_id, time.monotonic() + 30)
            log.info("decision_fallback", backend="jev", reason=reason, **diagnostics)

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
            f"Jev routing unavailable ({reason}); connected fallback stopped: {detail}",
            fallback_reason=reason, diagnostics=diagnostics,
        ) from exc
    if roles:
        artifact_roles = ArtifactRoleHints.for_request(request, roles)
    return RoutingDecision(values, "connected", selected.model, reason, artifact_roles, diagnostics)
