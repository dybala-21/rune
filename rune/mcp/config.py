"""MCP server configuration for RUNE.

Loads and saves MCP server configurations from the RUNE config directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_MCP_CONFIG_FILE = "mcp_servers.json"

TransportType = Literal["stdio", "sse", "streamable-http"]


@dataclass(slots=True)
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: TransportType = "stdio"
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    disabled: bool = False


def _config_path() -> Path:
    return rune_home() / _MCP_CONFIG_FILE


def load_mcp_config() -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from disk.

    Reads from ~/.rune/mcp_servers.json. Returns an empty dict if
    the file doesn't exist or is malformed.

    The file format mirrors the TS config:
    ``{"mcpServers": {"name": {config...}}}``

    Also supports the legacy list format from the previous Python version.
    """
    path = _config_path()
    if not path.is_file():
        return {}

    try:
        data = json_decode(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("mcp_config_load_error", error=str(exc))
        return {}

    configs: dict[str, MCPServerConfig] = {}

    # New dict-keyed format: {"mcpServers": {"name": {...}}}
    if isinstance(data, dict) and "mcpServers" in data:
        raw = data["mcpServers"]
        if isinstance(raw, dict):
            for name, entry in raw.items():
                if not isinstance(entry, dict):
                    continue
                configs[name] = _parse_server_entry(name, entry)
            log.debug("mcp_config_loaded", count=len(configs))
            return configs

    # Legacy list format: {"servers": [{...}]} or [{...}]
    raw_servers: list[dict[str, Any]] = (
        data if isinstance(data, list) else data.get("servers", [])
    )
    for entry in raw_servers:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        name = entry["name"]
        configs[name] = _parse_server_entry(name, entry)

    log.debug("mcp_config_loaded", count=len(configs))
    return configs


def _parse_server_entry(name: str, entry: dict[str, Any]) -> MCPServerConfig:
    """Parse a single server config entry."""
    # Handle command as string or list
    command_raw = entry.get("command")
    if isinstance(command_raw, list):
        command = command_raw[0] if command_raw else None
        args = command_raw[1:] + entry.get("args", [])
    else:
        command = command_raw
        args = entry.get("args", [])

    return MCPServerConfig(
        name=name,
        command=command,
        args=args,
        env=entry.get("env", {}),
        transport=entry.get("transport", "stdio"),
        url=entry.get("url"),
        headers=entry.get("headers", {}),
        disabled=entry.get("disabled", False),
    )


def save_mcp_config(configs: dict[str, MCPServerConfig]) -> None:
    """Save MCP server configurations to disk."""
    path = _config_path()

    mcp_servers: dict[str, dict[str, Any]] = {}
    for name, cfg in configs.items():
        entry: dict[str, Any] = {"transport": cfg.transport}
        if cfg.command:
            entry["command"] = cfg.command
        if cfg.args:
            entry["args"] = cfg.args
        if cfg.env:
            entry["env"] = cfg.env
        if cfg.url:
            entry["url"] = cfg.url
        if cfg.headers:
            entry["headers"] = cfg.headers
        if cfg.disabled:
            entry["disabled"] = cfg.disabled
        mcp_servers[name] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"mcpServers": mcp_servers}, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("mcp_config_saved", count=len(configs))
