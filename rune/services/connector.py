"""Service connector for RUNE.

Ported from src/services/connector.ts - manages the lifecycle of
external service connections: status checks, configuration, connection,
disconnection, and write-operation detection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rune.services.catalog import (
    DEFAULT_WRITE_PATTERN,
    AuthType,
    ServiceDefinition,
    find_service_by_id,
    get_service_catalog,
)
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


# ============================================================================
# Types
# ============================================================================

@dataclass(slots=True)
class ServiceStatus:
    """Connection status for a single service."""

    service_id: str
    configured: bool = False
    connected: bool = False
    tool_count: int = 0
    missing_credentials: list[str] = field(default_factory=list)


# ============================================================================
# MCP config helpers
# ============================================================================

def _mcp_config_path(scope: str = "user") -> Path:
    """Return the path to the ``mcp.json`` config file."""
    if scope == "project":
        return Path.cwd() / ".rune" / "mcp.json"
    return rune_home() / "mcp.json"


def _load_mcp_config(scope: str = "user") -> dict[str, Any]:
    """Load the MCP configuration from disk."""
    path = _mcp_config_path(scope)
    if not path.exists():
        return {}
    try:
        return json_decode(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("mcp_config_load_failed", path=str(path), error=str(exc))
        return {}


def _save_mcp_config(config: dict[str, Any], scope: str = "user") -> Path:
    """Write the MCP configuration to disk and return the path."""
    path = _mcp_config_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


# ============================================================================
# ServiceConnector
# ============================================================================

class ServiceConnector:
    """Manages connections to external services via MCP servers."""

    # -- Status checking -----------------------------------------------------

    @staticmethod
    def check_status(service: ServiceDefinition) -> ServiceStatus:
        """Check the connection status of a service.

        Verifies credential availability and whether the service appears
        in the MCP configuration.
        """
        status = ServiceStatus(service_id=service.id)

        # Check credential environment variables
        for cred in service.auth.credentials:
            if not os.environ.get(cred.key):
                status.missing_credentials.append(cred.key)

        # Check MCP config presence
        config = _load_mcp_config("user")
        mcp_servers: dict[str, Any] = config.get("mcpServers", {})
        if service.id in mcp_servers:
            status.configured = True

        # A service is considered "connected" if configured and no
        # credentials are missing.
        if status.configured and not status.missing_credentials:
            status.connected = True

        return status

    # -- Connection lifecycle ------------------------------------------------

    @staticmethod
    def connect(
        service: ServiceDefinition,
        credentials: dict[str, str],
        *,
        scope: str = "user",
    ) -> tuple[bool, str]:
        """Configure and persist a service connection.

        Writes the service MCP server entry into ``mcp.json`` with the
        supplied credential environment variables.

        Returns ``(success, config_path)``.
        """
        try:
            config = _load_mcp_config(scope)
            if "mcpServers" not in config:
                config["mcpServers"] = {}

            env: dict[str, str] = {**credentials}

            config["mcpServers"][service.id] = {
                "transport": "stdio",
                "command": service.mcp_server.command,
                "args": service.mcp_server.args,
                "env": env,
            }

            path = _save_mcp_config(config, scope)
            log.info(
                "service_configured",
                service=service.id,
                scope=scope,
                path=str(path),
            )
            return True, str(path)
        except Exception as exc:
            log.error(
                "service_configure_failed",
                service=service.id,
                error=str(exc),
            )
            return False, ""

    @staticmethod
    def disconnect(
        service: ServiceDefinition,
        *,
        scope: str = "user",
        clear_tokens: bool = True,
    ) -> tuple[bool, bool, bool]:
        """Fully disconnect a service.

        * Removes the MCP config entry from ``mcp.json``
        * Optionally clears OAuth tokens

        Returns ``(success, tokens_cleared, config_removed)``.
        """
        tokens_cleared = False
        config_removed = False

        # 1. Remove OAuth tokens
        if clear_tokens and service.auth.type == AuthType.OAUTH:
            cred_dir = rune_home() / "credentials"

            # RUNE-managed token
            token_path = cred_dir / service.id / "tokens.json"
            if token_path.exists():
                try:
                    token_path.unlink()
                    tokens_cleared = True
                    log.debug("token_cleared", path=str(token_path))
                except OSError:
                    pass

            # Service default token paths
            oauth = service.auth.oauth
            if oauth is not None:
                home = str(Path.home())
                for default_path in oauth.token_default_paths:
                    expanded = default_path.replace("~", home, 1)
                    resolved = Path(expanded).resolve()
                    # Safety: only delete files under home directory with
                    # sufficient depth to prevent accidental broad deletion.
                    resolved_str = str(resolved)
                    if not resolved_str.startswith(home + "/") or len(resolved.parts) < 4:
                        log.warning(
                            "token_path_rejected",
                            path=expanded,
                            reason="not under home directory or too shallow",
                        )
                        continue
                    try:
                        resolved.unlink()
                        tokens_cleared = True
                        log.debug("token_cleared", path=expanded)
                    except (OSError, FileNotFoundError):
                        pass

        # 2. Remove from mcp.json
        config = _load_mcp_config(scope)
        mcp_servers: dict[str, Any] = config.get("mcpServers", {})
        if service.id in mcp_servers:
            del mcp_servers[service.id]
            _save_mcp_config(config, scope)
            config_removed = True

        log.info(
            "service_disconnected",
            service=service.id,
            tokens_cleared=tokens_cleared,
            config_removed=config_removed,
        )
        return True, tokens_cleared, config_removed

    # -- Listing -------------------------------------------------------------

    @staticmethod
    def get_status_all() -> list[ServiceStatus]:
        """Return connection status for every cataloged service."""
        return [
            ServiceConnector.check_status(svc)
            for svc in get_service_catalog()
        ]

    @staticmethod
    def list_connected() -> list[ServiceStatus]:
        """Return only services that are currently connected."""
        return [
            s for s in ServiceConnector.get_status_all() if s.connected
        ]


# ============================================================================
# Write detection (pure function, sync)
# ============================================================================

def is_mcp_write_operation(capability_name: str) -> bool:
    """Detect whether a MCP capability name represents a write operation.

    Used by the tool adapter to gate write operations behind approval.
    """
    if not capability_name.startswith("mcp."):
        return False

    parts = capability_name.split(".")
    if len(parts) < 3:
        return False

    service_id = parts[1]
    tool_name = ".".join(parts[2:])

    # Per-service override from catalog
    service = find_service_by_id(service_id)
    if service and service.write_patterns:
        return bool(service.write_patterns.search(tool_name))

    # Default pattern
    return bool(DEFAULT_WRITE_PATTERN.search(tool_name))
