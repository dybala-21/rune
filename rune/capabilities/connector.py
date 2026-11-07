"""Use configured connectors without putting credentials in tool arguments."""

import asyncio
import json
from typing import Any

from pydantic import BaseModel, TypeAdapter

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.connectors.client import BrokerUnavailable, call_broker, hosted_approval
from rune.connectors.models import ConnectorRequest
from rune.safety.approval_context import approval_channel, was_approved
from rune.types import CapabilityResult, Domain, RiskLevel


class ConnectorListParams(BaseModel):
    pass


async def connector_list(_params: ConnectorListParams) -> CapabilityResult:
    try:
        return CapabilityResult(success=True, output=json.dumps(await call_broker("list", {})))
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Connector broker unavailable ({type(exc).__name__})")


# How often a waiting request is re-sent to the broker to learn whether the owner decided.
APPROVAL_POLL_SECONDS = 2.0


async def connector_request(params: ConnectorRequest) -> CapabilityResult:
    hosted = hosted_approval()
    if not hosted and not was_approved():
        return CapabilityResult(success=False, error="Connector request requires approval",
                                metadata={"action_status": "not_executed", "requires_approval": True})
    payload = params.model_dump()
    try:
        result = await call_broker("request", payload)
        if hosted and (result.get("metadata") or {}).get("approval_status") == "pending":
            return await _wait_for_owner(params, payload, result)
        return TypeAdapter(CapabilityResult).validate_python(result)
    except BrokerUnavailable as exc:
        return CapabilityResult(success=False, error=str(exc), metadata={"action_status": "not_executed"})
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Connector result unavailable ({type(exc).__name__}); do not repeat a write without checking its outcome",
                                metadata={"action_status": "unknown"})


async def _wait_for_owner(params: ConnectorRequest, payload: dict[str, Any], pending: dict[str, Any]) -> CapabilityResult:
    """Hold a hosted call while the owner decides on the broker's approval page.

    The broker runs the request the first time it is sent after approval, so waiting is sending
    the same request again until it is no longer pending. Declining in the app stops the wait.
    """
    url = str((pending.get("metadata") or {}).get("approval_url") or "")
    external = getattr(approval_channel(), "external", None)
    if external is None or not url.startswith("https://"):
        # This run can't show the page (no approval channel, or no https URL). The pending result
        # names the page, and an identical call after approval runs.
        return TypeAdapter(CapabilityResult).validate_python(pending)

    stopped = asyncio.Event()

    async def decided() -> dict[str, Any] | None:
        while True:
            try:
                await asyncio.wait_for(stopped.wait(), APPROVAL_POLL_SECONDS)
                return None
            except TimeoutError:
                pass
            result = await call_broker("request", payload)
            if (result.get("metadata") or {}).get("approval_status") != "pending":
                return result
            if stopped.is_set():
                return None

    display = f"{params.method} {params.origin}{params.path}"[:200]
    outcome, result = await external(display, f"Connected API: {display}. Approve it on the approval page.", url,
                                     decided, stopped.set)
    if outcome == "done":
        return TypeAdapter(CapabilityResult).validate_python(result)
    error = ("The user declined this request." if outcome == "declined"
             else "Approval timed out; the request did not run.")
    return CapabilityResult(success=False, error=error,
                            metadata={"action_status": "not_executed",
                                      **({"approval_decision": "denied"} if outcome == "declined" else {})})


def register_connector_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="connector_list", domain=Domain.GENERAL, group="web", risk_level=RiskLevel.LOW,
        description="List connected HTTP APIs and their permitted origins, paths and methods. No credentials are returned.",
        parameters_model=ConnectorListParams, execute=connector_list,
    ))
    registry.register(CapabilityDefinition(
        name="connector_request", domain=Domain.GENERAL, group="web", risk_level=RiskLevel.HIGH,
        description="Call a configured API using its connector name. List connectors first. Requires approval; credentials stay in the broker. Never repeat an uncertain write.",
        parameters_model=ConnectorRequest, execute=connector_request,
    ))
