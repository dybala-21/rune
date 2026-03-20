"""MCP Bridge for RUNE.

Integrates MCP tools as RUNE capabilities by discovering tools from MCP
servers and registering them in the capability registry with dot-notation
names: ``mcp.<serverName>.<toolName>``.

Ported from ``src/mcp/bridge.ts``.

Bridge lifecycle:
1. ``initialize_mcp_bridge(configs, registry)`` - connect all servers,
   convert tools to capabilities, register in registry.
2. ``connect_single_server(name, config, registry)`` - runtime add.
3. ``shutdown_mcp_bridge()`` - disconnect all servers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rune.capabilities.registry import CapabilityRegistry, get_capability_registry
from rune.capabilities.types import TOOL_GROUPS, CapabilityDefinition
from rune.mcp.client import (
    MCPClientManager,
    MCPTool,
    MCPToolResult,
    get_mcp_client_manager,
)
from rune.mcp.config import MCPServerConfig
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# Types
# ============================================================================


@dataclass(slots=True)
class MCPBridgeResult:
    """Result of an MCP bridge initialization or server connection."""

    registered_count: int = 0
    connected_servers: int = 0
    failed_servers: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)


# ============================================================================
# Tool to Capability conversion
# ============================================================================


def _mcp_tool_to_capability(
    server_name: str,
    tool: MCPTool,
    manager: MCPClientManager,
) -> CapabilityDefinition:
    """Convert an MCP tool into a RUNE ``CapabilityDefinition``.

    The capability name follows dot notation: ``mcp.<server>.<tool>``.
    Risk level is ``medium`` for all MCP operations (external service access).
    The ``execute`` function delegates to ``MCPClientManager.call_tool()``.
    """
    capability_name = f"mcp.{server_name}.{tool.name}"
    description = tool.description or f"[MCP:{server_name}] {tool.name}"

    async def _execute(params: Any) -> CapabilityResult:
        """Execute the MCP tool via the client manager."""
        # Accept both dict and Pydantic model
        if hasattr(params, "model_dump"):
            args = params.model_dump()
        elif isinstance(params, dict):
            args = params
        else:
            args = dict(params) if params else {}

        try:
            result: MCPToolResult = await manager.call_tool(
                server_name, tool.name, args
            )

            # Convert MCP content to text output
            text_parts = [
                c.get("text", "")
                for c in result.content
                if c.get("type") == "text" and c.get("text")
            ]
            output = (
                "\n".join(text_parts)
                if text_parts
                else _serialize_content(result.content)
            )

            if result.is_error:
                return CapabilityResult(
                    success=False,
                    output=output,
                    error=f"MCP tool error: {output}",
                )

            return CapabilityResult(
                success=True,
                output=output,
                metadata={
                    "mcp_server": server_name,
                    "mcp_tool": tool.name,
                    "content_types": [c.get("type", "") for c in result.content],
                },
            )

        except Exception as exc:
            msg = str(exc)
            return CapabilityResult(
                success=False,
                output="",
                error=f"MCP call failed ({server_name}/{tool.name}): {msg}",
            )

    return CapabilityDefinition(
        name=capability_name,
        description=description,
        domain=Domain.NETWORK,
        risk_level=RiskLevel.MEDIUM,
        group="mcp",
        raw_json_schema=tool.input_schema if hasattr(tool, "input_schema") else None,
        execute=_execute,
    )


def _serialize_content(content: list[dict[str, Any]]) -> str:
    """Fallback serialization for non-text MCP content."""
    from rune.utils.fast_serde import json_encode

    try:
        return json_encode(content)
    except (TypeError, ValueError):
        return str(content)


# ============================================================================
# Bridge functions
# ============================================================================


async def initialize_mcp_bridge(
    configs: dict[str, MCPServerConfig] | None = None,
    registry: CapabilityRegistry | None = None,
) -> MCPBridgeResult:
    """Connect all MCP servers and register their tools as capabilities.

    Parameters
    ----------
    configs:
        Server configurations keyed by name.  If *None*, loads from the
        default config file via ``load_mcp_config()``.
    registry:
        Capability registry to register into.  Uses the module singleton
        if not provided.
    """
    if configs is None:
        from rune.mcp.config import load_mcp_config

        configs = load_mcp_config()

    if registry is None:
        registry = get_capability_registry()

    # Filter disabled servers
    active = {k: v for k, v in configs.items() if not v.disabled}
    if not active:
        log.debug("mcp_no_servers_configured")
        return MCPBridgeResult()

    manager = get_mcp_client_manager()
    failed_servers: list[str] = []
    capabilities: list[str] = []

    # Connect all servers in parallel
    tools_map = await manager.connect_all(active)

    # Register tools as capabilities
    for server_name, tools in tools_map.items():
        if not tools:
            failed_servers.append(server_name)
            continue

        for tool in tools:
            cap = _mcp_tool_to_capability(server_name, tool, manager)
            registry.register(cap)
            # Add to the mcp group in TOOL_GROUPS for policy resolution
            _ensure_mcp_group(cap.name)
            capabilities.append(cap.name)
            log.debug("mcp_capability_registered", name=cap.name)

    connected = manager.get_connected_count()
    log.info(
        "mcp_bridge_initialized",
        connected_servers=connected,
        capability_count=len(capabilities),
    )

    return MCPBridgeResult(
        registered_count=len(capabilities),
        connected_servers=connected,
        failed_servers=failed_servers,
        capabilities=capabilities,
    )


async def connect_single_server(
    name: str,
    config: MCPServerConfig,
    registry: CapabilityRegistry | None = None,
    timeout: float | None = None,
) -> MCPBridgeResult:
    """Connect a single MCP server at runtime and register its tools.

    Use this for dynamic server addition (e.g., after OAuth flow completion).
    """
    if registry is None:
        registry = get_capability_registry()

    manager = get_mcp_client_manager()

    try:
        tools = await manager.connect(name, config, timeout=timeout)

        if not tools:
            return MCPBridgeResult(
                failed_servers=[name],
            )

        capabilities: list[str] = []
        for tool in tools:
            cap = _mcp_tool_to_capability(name, tool, manager)
            registry.register(cap)
            _ensure_mcp_group(cap.name)
            capabilities.append(cap.name)

        log.info(
            "mcp_single_server_connected",
            server=name,
            tool_count=len(tools),
        )

        return MCPBridgeResult(
            registered_count=len(capabilities),
            connected_servers=1,
            capabilities=capabilities,
        )

    except Exception as exc:
        log.error(
            "mcp_single_server_failed",
            server=name,
            error=str(exc),
        )
        return MCPBridgeResult(failed_servers=[name])


async def shutdown_mcp_bridge() -> None:
    """Disconnect all MCP servers."""
    manager = get_mcp_client_manager()
    await manager.disconnect_all()
    log.info("mcp_bridge_shutdown")


def list_mcp_capabilities() -> list[str]:
    """Return names of all registered MCP capabilities."""
    registry = get_capability_registry()
    return [name for name in registry.list_names() if name.startswith("mcp.")]


def get_mcp_status() -> dict[str, Any]:
    """MCP server status summary (for CLI display)."""
    manager = get_mcp_client_manager()
    servers = manager.get_status()
    total_tools = sum(s["tool_count"] for s in servers)
    return {"servers": servers, "total_tools": total_tools}


# ============================================================================
# Helpers
# ============================================================================


def _ensure_mcp_group(capability_name: str) -> None:
    """Add a capability name to the ``mcp`` group in ``TOOL_GROUPS``.

    This allows policy profiles to reference ``mcp`` as a group.
    """
    group = TOOL_GROUPS.setdefault("mcp", [])
    if capability_name not in group:
        group.append(capability_name)
