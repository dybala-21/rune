"""Use configured connectors without putting credentials in tool arguments."""

import json

from pydantic import BaseModel, TypeAdapter

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.connectors.client import BrokerUnavailable, call_broker
from rune.connectors.models import ConnectorRequest
from rune.safety.approval_context import was_approved
from rune.types import CapabilityResult, Domain, RiskLevel


class ConnectorListParams(BaseModel):
    pass


async def connector_list(_params: ConnectorListParams) -> CapabilityResult:
    try:
        return CapabilityResult(success=True, output=json.dumps(await call_broker("list", {})))
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Connector broker unavailable ({type(exc).__name__})")


async def connector_request(params: ConnectorRequest) -> CapabilityResult:
    if not was_approved():
        return CapabilityResult(success=False, error="Connector request requires approval",
                                metadata={"action_status": "not_executed", "requires_approval": True})
    try:
        return TypeAdapter(CapabilityResult).validate_python(await call_broker("request", params.model_dump()))
    except BrokerUnavailable as exc:
        return CapabilityResult(success=False, error=str(exc), metadata={"action_status": "not_executed"})
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Connector result unavailable ({type(exc).__name__}); do not repeat a write without checking its outcome",
                                metadata={"action_status": "unknown"})


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
