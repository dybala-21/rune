"""MCP server management - CRUD API for MCP server configurations.

Endpoints:
  GET    /api/v1/mcp/servers          - list all configured servers
  POST   /api/v1/mcp/servers          - add a new server
  PUT    /api/v1/mcp/servers/{name}   - update a server
  DELETE /api/v1/mcp/servers/{name}   - remove a server
  POST   /api/v1/mcp/servers/{name}/test - test server connection
"""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

_VALID_SERVER_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,98}[a-zA-Z0-9]$")

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])
auth = TokenAuthDependency()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MCPServerRequest(BaseModel):
    """Request body for creating/updating an MCP server."""
    name: str = Field(max_length=100, description="Unique server name (alphanumeric, dots, hyphens, underscores)")
    command: str | None = Field(default=None, description="Command to start the server (stdio)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        default="stdio", description="Transport type"
    )
    url: str | None = Field(default=None, description="Server URL (sse/streamable-http)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers (sse/streamable-http)")
    disabled: bool = Field(default=False, description="Disable without removing")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not _VALID_SERVER_NAME.match(v):
            msg = "Server name must be alphanumeric with dots, hyphens, or underscores (2-100 chars)"
            raise ValueError(msg)
        return v


class MCPServerInfo(BaseModel):
    """Response model for a single MCP server."""
    name: str
    command: str | None = None
    args: list[str] = []
    transport: str = "stdio"
    url: str | None = None
    disabled: bool = False
    has_env: bool = False
    has_headers: bool = False


class MCPServerListResponse(BaseModel):
    servers: list[MCPServerInfo]
    count: int


class MCPTestResponse(BaseModel):
    name: str
    success: bool
    message: str
    tools_count: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/servers", response_model=MCPServerListResponse, dependencies=[Depends(auth)])
async def list_mcp_servers() -> MCPServerListResponse:
    """List all configured MCP servers."""
    from rune.mcp.config import load_mcp_config

    configs = load_mcp_config()
    servers = [
        MCPServerInfo(
            name=name,
            command=cfg.command,
            args=cfg.args,
            transport=cfg.transport,
            url=cfg.url,
            disabled=cfg.disabled,
            has_env=bool(cfg.env),
            has_headers=bool(cfg.headers),
        )
        for name, cfg in configs.items()
    ]
    return MCPServerListResponse(servers=servers, count=len(servers))


@router.post("/servers", response_model=MCPServerInfo, dependencies=[Depends(auth)])
async def add_mcp_server(req: MCPServerRequest) -> MCPServerInfo:
    """Add a new MCP server configuration."""
    from rune.mcp.config import MCPServerConfig, load_mcp_config, save_mcp_config

    configs = load_mcp_config()

    if req.name in configs:
        raise HTTPException(status_code=409, detail=f"Server '{req.name}' already exists")

    # Validate: stdio needs command, sse/streamable-http needs url
    if req.transport == "stdio" and not req.command:
        raise HTTPException(status_code=422, detail="stdio transport requires 'command'")
    if req.transport in ("sse", "streamable-http") and not req.url:
        raise HTTPException(status_code=422, detail=f"{req.transport} transport requires 'url'")

    configs[req.name] = MCPServerConfig(
        name=req.name,
        command=req.command,
        args=req.args,
        env=req.env,
        transport=req.transport,
        url=req.url,
        headers=req.headers,
        disabled=req.disabled,
    )
    save_mcp_config(configs)
    log.info("mcp_server_added", name=req.name, transport=req.transport)

    return MCPServerInfo(
        name=req.name,
        command=req.command,
        args=req.args,
        transport=req.transport,
        url=req.url,
        disabled=req.disabled,
        has_env=bool(req.env),
        has_headers=bool(req.headers),
    )


@router.put("/servers/{name}", response_model=MCPServerInfo, dependencies=[Depends(auth)])
async def update_mcp_server(name: str, req: MCPServerRequest) -> MCPServerInfo:
    """Update an existing MCP server configuration."""
    from rune.mcp.config import MCPServerConfig, load_mcp_config, save_mcp_config

    configs = load_mcp_config()

    if name not in configs:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    # If name changed, remove old entry
    if req.name != name:
        del configs[name]

    configs[req.name] = MCPServerConfig(
        name=req.name,
        command=req.command,
        args=req.args,
        env=req.env,
        transport=req.transport,
        url=req.url,
        headers=req.headers,
        disabled=req.disabled,
    )
    save_mcp_config(configs)
    log.info("mcp_server_updated", name=req.name)

    return MCPServerInfo(
        name=req.name,
        command=req.command,
        args=req.args,
        transport=req.transport,
        url=req.url,
        disabled=req.disabled,
        has_env=bool(req.env),
        has_headers=bool(req.headers),
    )


@router.delete("/servers/{name}", dependencies=[Depends(auth)])
async def delete_mcp_server(name: str) -> dict[str, Any]:
    """Remove an MCP server configuration."""
    from rune.mcp.config import load_mcp_config, save_mcp_config

    configs = load_mcp_config()

    if name not in configs:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    del configs[name]
    save_mcp_config(configs)
    log.info("mcp_server_deleted", name=name)

    return {"ok": True, "deleted": name}


@router.post("/servers/{name}/test", response_model=MCPTestResponse, dependencies=[Depends(auth)])
async def test_mcp_server(name: str) -> MCPTestResponse:
    """Test connection to an MCP server and discover tools."""
    from rune.mcp.config import load_mcp_config

    configs = load_mcp_config()

    if name not in configs:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    cfg = configs[name]

    try:
        from rune.mcp.client import MCPClient

        client = MCPClient(server_name=name, config=cfg)
        await client.connect()
        tools = await client.list_tools()
        await client.disconnect()

        return MCPTestResponse(
            name=name,
            success=True,
            message=f"Connected. {len(tools)} tools discovered.",
            tools_count=len(tools),
        )
    except Exception as exc:
        return MCPTestResponse(
            name=name,
            success=False,
            message=f"Connection failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
