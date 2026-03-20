"""Service integration capabilities for RUNE.

Ported from src/capabilities/service.ts - connect, disconnect, reconnect,
status, and list external services (Google Calendar, Notion, etc.).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schemas

class ServiceConnectParams(BaseModel):
    """Parameters for service.connect."""
    query: str = Field(description="Service name or keyword (e.g. 'google calendar', 'notion')")
    channel: str | None = Field(
        default=None,
        description="Current channel (tui, cli, telegram, discord). Affects OAuth browser flow.",
    )


class ServiceStatusParams(BaseModel):
    """Parameters for service.status."""
    service_id: str | None = Field(
        default=None, alias="serviceId",
        description="Specific service ID. Omit for full status.",
    )

    model_config = ConfigDict(populate_by_name=True)


class ServiceListParams(BaseModel):
    """Parameters for service.list."""
    category: str | None = Field(
        default=None,
        description="Category filter (productivity, development, communication, etc.)",
    )


class ServiceDisconnectParams(BaseModel):
    """Parameters for service.disconnect."""
    query: str = Field(description="Service name or keyword")


class ServiceReconnectParams(BaseModel):
    """Parameters for service.reconnect."""
    query: str = Field(description="Service name or keyword")
    channel: str | None = Field(default=None, description="Current channel")


# ServiceConnection & ServiceRegistry

@dataclass(slots=True)
class ServiceConnection:
    """Tracks a live connection to an external service."""

    service_id: str
    service_name: str
    connected_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    config_path: str = ""
    credentials: dict[str, str] = field(default_factory=dict)
    tool_count: int = 0
    healthy: bool = True


class ServiceRegistry:
    """Singleton registry that tracks connected services."""

    _instance: ServiceRegistry | None = None
    _connections: dict[str, ServiceConnection]

    def __init__(self) -> None:
        self._connections = {}

    @classmethod
    def get_instance(cls) -> ServiceRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def connect(
        self, name: str, service_name: str, config_path: str = "",
        credentials: dict[str, str] | None = None, tool_count: int = 0,
    ) -> ServiceConnection:
        """Register a connected service."""
        conn = ServiceConnection(
            service_id=name,
            service_name=service_name,
            config_path=config_path,
            credentials=credentials or {},
            tool_count=tool_count,
            healthy=True,
        )
        self._connections[name] = conn
        log.info("service_registry_connect", service_id=name)
        return conn

    def disconnect(self, name: str) -> None:
        """Remove a service from the registry."""
        if name in self._connections:
            del self._connections[name]
            log.info("service_registry_disconnect", service_id=name)

    def get_connection(self, name: str) -> ServiceConnection | None:
        return self._connections.get(name)

    def get_status(self, name: str) -> dict[str, Any]:
        """Return status info for a service."""
        conn = self._connections.get(name)
        if conn is None:
            return {
                "connected": False,
                "service_id": name,
            }
        return {
            "connected": True,
            "service_id": conn.service_id,
            "service_name": conn.service_name,
            "last_activity": conn.last_activity,
            "healthy": conn.healthy,
            "tool_count": conn.tool_count,
        }

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered connections."""
        return [self.get_status(name) for name in self._connections]

    def is_connected(self, name: str) -> bool:
        return name in self._connections


# Capability implementations

async def service_connect(params: ServiceConnectParams) -> CapabilityResult:
    """Connect an external service (handles OAuth auto-flow when supported)."""
    log.info("service_connect", query=params.query, channel=params.channel)

    from rune.services.catalog import find_service, get_service_catalog
    from rune.services.connector import ServiceConnector

    service = find_service(params.query)

    if not service:
        catalog = get_service_catalog()
        available = "\n".join(f"  - {s.name}: {s.description}" for s in catalog)
        return CapabilityResult(
            success=False,
            output=f'Service "{params.query}" not found.\n\nAvailable services:\n{available}',
            error="Service not found",
            suggestions=[s.name for s in catalog],
        )

    # Check current status
    status = ServiceConnector.check_status(service)
    registry = ServiceRegistry.get_instance()

    # Already connected
    if status.connected or registry.is_connected(service.id):
        return CapabilityResult(
            success=True,
            output=f"{service.name} is already connected.",
            metadata={"serviceId": service.id, "status": "connected"},
        )

    # No auth needed
    if service.auth.type.value == "none":
        # Auto-connect services that need no authentication
        success, config_path = ServiceConnector.connect(service, {})
        if success:
            registry.connect(service.id, service.name, config_path=config_path)
        return CapabilityResult(
            success=True,
            output=f"{service.name} requires no authentication. MCP config has been added.",
            metadata={"serviceId": service.id, "status": "no_auth_needed"},
        )

    # Check which credentials are missing
    missing: list[str] = []
    available_creds: dict[str, str] = {}
    for cred in service.auth.credentials:
        val = os.environ.get(cred.key, "")
        if val:
            available_creds[cred.key] = val
        else:
            missing.append(cred.key)

    # All credentials present - connect directly
    if not missing:
        success, config_path = ServiceConnector.connect(service, available_creds)
        if success:
            registry.connect(
                service.id, service.name,
                config_path=config_path, credentials=available_creds,
            )
            return CapabilityResult(
                success=True,
                output=f"{service.name} connected successfully!",
                metadata={"serviceId": service.id, "status": "connected"},
            )
        return CapabilityResult(
            success=False,
            output=f"Failed to configure {service.name} MCP server.",
            error="MCP configuration failed",
            metadata={"serviceId": service.id},
        )

    # Credentials missing - show setup guide
    steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(service.auth.setup_steps))
    cred_info = "\n".join(
        f"- {c.label} (env: {c.key})"
        + (f"\n  Help: {c.help_url}" if c.help_url else "")
        for c in service.auth.credentials
    )
    auth_label = "OAuth" if service.auth.type.value == "oauth" else "API key"
    output = "\n".join([
        f"Setting up {service.name} connection.",
        "",
        f"Required: {auth_label}",
        cred_info,
        "",
        "Setup steps:",
        steps,
        "",
        f"Missing environment variables: {', '.join(missing)}",
        "Set these and try again.",
    ])

    return CapabilityResult(
        success=True,
        output=output,
        metadata={
            "serviceId": service.id,
            "status": "setup_guide_shown",
            "authType": service.auth.type.value,
            "missingCredentials": missing,
        },
    )


async def service_status(params: ServiceStatusParams) -> CapabilityResult:
    """Check connection status of external services."""
    log.info("service_status", service_id=params.service_id)

    from rune.services.catalog import find_service_by_id, get_service_catalog
    from rune.services.connector import ServiceConnector

    registry = ServiceRegistry.get_instance()

    if params.service_id:
        service = find_service_by_id(params.service_id)
        if not service:
            return CapabilityResult(
                success=False,
                output="",
                error=f'Service "{params.service_id}" not found in catalog',
            )
        status = ServiceConnector.check_status(service)
        reg_status = registry.get_status(service.id)
        connected = status.connected or reg_status.get("connected", False)
        icon = "[connected]" if connected else "[disconnected]"
        detail = (
            "Connected and available"
            if connected
            else ("Configured but not connected" if status.configured else "Not configured")
        )
        return CapabilityResult(
            success=True,
            output=f"{icon} {service.name}: {detail}",
            metadata={"status": {
                "service_id": service.id,
                "connected": connected,
                "configured": status.configured,
                "healthy": reg_status.get("healthy", False),
            }},
        )

    # Full status
    catalog = get_service_catalog()
    lines: list[str] = []
    for svc in catalog:
        status = ServiceConnector.check_status(svc)
        reg_status = registry.get_status(svc.id)
        connected = status.connected or reg_status.get("connected", False)
        icon = "[connected]" if connected else "[disconnected]"
        detail = (
            "available"
            if connected
            else ("configured" if status.configured else "not set up")
        )
        lines.append(f"{icon} {svc.name} ({svc.id}): {detail}")

    return CapabilityResult(
        success=True,
        output="Service status:\n" + "\n".join(lines),
        metadata={"serviceCount": len(catalog)},
    )


async def service_list(params: ServiceListParams) -> CapabilityResult:
    """List available external services."""
    log.info("service_list", category=params.category)

    from rune.services.catalog import get_service_catalog

    catalog = get_service_catalog()
    filtered = (
        [s for s in catalog if s.category.value == params.category]
        if params.category
        else catalog
    )

    if not filtered:
        msg = (
            f'No services found in category "{params.category}".'
            if params.category
            else "No services registered."
        )
        return CapabilityResult(success=True, output=msg)

    # Group by category
    groups: dict[str, list[Any]] = {}
    for svc in filtered:
        cat = svc.category.value
        groups.setdefault(cat, []).append(svc)

    category_labels = {
        "productivity": "Productivity",
        "communication": "Communication",
        "development": "Development",
        "data": "Data",
        "media": "Media",
        "lifestyle": "Lifestyle",
        "system": "System",
        "other": "Other",
    }

    def difficulty_label(auth_type: str) -> str:
        if auth_type == "none":
            return "[easy]"
        if auth_type == "api-key":
            return "[medium]"
        if auth_type == "oauth":
            return "[complex]"
        return "[?]"

    lines: list[str] = ["Available services:"]
    for cat, services in groups.items():
        label = category_labels.get(cat, cat)
        lines.append(f"\n[{label}]")
        for s in services:
            icon = difficulty_label(s.auth.type.value)
            lines.append(f"  {icon} {s.name} -- {s.description}")

    lines.append("\nDifficulty: [easy] no auth | [medium] API key | [complex] OAuth")

    return CapabilityResult(
        success=True,
        output="\n".join(lines),
        metadata={"totalServices": len(filtered)},
    )


async def service_disconnect(params: ServiceDisconnectParams) -> CapabilityResult:
    """Disconnect an external service."""
    log.info("service_disconnect", query=params.query)

    from rune.services.catalog import find_service
    from rune.services.connector import ServiceConnector

    service = find_service(params.query)
    if not service:
        return CapabilityResult(
            success=False,
            output=f'Service "{params.query}" not found.',
            error="Service not found",
        )

    registry = ServiceRegistry.get_instance()

    # Disconnect via connector (removes MCP config, clears tokens)
    success, tokens_cleared, config_removed = ServiceConnector.disconnect(service)

    # Remove from registry
    registry.disconnect(service.id)

    details: list[str] = []
    if tokens_cleared:
        details.append("auth tokens cleared")
    if config_removed:
        details.append("MCP config removed")
    if not details:
        details.append("disconnected")

    return CapabilityResult(
        success=True,
        output=f"{service.name} disconnected ({', '.join(details)}).",
        metadata={
            "serviceId": service.id,
            "tokenCleared": tokens_cleared,
            "configRemoved": config_removed,
        },
    )


async def service_reconnect(params: ServiceReconnectParams) -> CapabilityResult:
    """Reconnect an external service with a different account."""
    log.info("service_reconnect", query=params.query, channel=params.channel)

    from rune.services.catalog import find_service

    service = find_service(params.query)
    if not service:
        return CapabilityResult(
            success=False,
            output=f'Service "{params.query}" not found.',
            error="Service not found",
        )

    # Non-OAuth: direct user to update credentials
    if service.auth.type.value != "oauth":
        cred_key = service.auth.credentials[0].key if service.auth.credentials else None
        return CapabilityResult(
            success=False,
            output=(
                f"{service.name} uses API key authentication. "
                f"Update the credential environment variable"
                + (f" ({cred_key})" if cred_key else "")
                + " and reconnect."
            ),
            error="Not an OAuth service",
            suggestions=["credential_save"],
        )

    # 1. Disconnect first
    await service_disconnect(
        ServiceDisconnectParams(query=params.query)
    )
    log.info("service_reconnect_disconnected", service_id=service.id)

    # 2. Reconnect
    connect_result = await service_connect(
        ServiceConnectParams(query=params.query, channel=params.channel)
    )

    if connect_result.success:
        return CapabilityResult(
            success=True,
            output=f"{service.name} reconnected successfully!\n{connect_result.output}",
            metadata={"serviceId": service.id, "status": "reconnected", **connect_result.metadata},
        )

    return CapabilityResult(
        success=False,
        output=f"{service.name} reconnection failed. {connect_result.output}",
        error="Reconnect failed",
        metadata={"serviceId": service.id},
    )


# Registration

def register_service_capabilities(registry: CapabilityRegistry) -> None:
    """Register all service capabilities."""
    registry.register(CapabilityDefinition(
        name="service_connect",
        description=(
            "Connect an external service (Google Calendar, Google Drive, "
            "Notion, Todoist, GitHub, etc.). Handles OAuth auto-flow."
        ),
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=ServiceConnectParams,
        execute=service_connect,
    ))
    registry.register(CapabilityDefinition(
        name="service_status",
        description="Check connection status of external services.",
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=ServiceStatusParams,
        execute=service_status,
    ))
    registry.register(CapabilityDefinition(
        name="service_list",
        description="List all available external services that can be connected.",
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=ServiceListParams,
        execute=service_list,
    ))
    registry.register(CapabilityDefinition(
        name="service_disconnect",
        description="Disconnect an external service.",
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=ServiceDisconnectParams,
        execute=service_disconnect,
    ))
    registry.register(CapabilityDefinition(
        name="service_reconnect",
        description=(
            "Reconnect an external service with a different account. "
            "Clears existing OAuth tokens and starts fresh login."
        ),
        domain=Domain.NETWORK,
        risk_level=RiskLevel.LOW,
        group="web",
        parameters_model=ServiceReconnectParams,
        execute=service_reconnect,
    ))
