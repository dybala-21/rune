"""MCP (Model Context Protocol) client and multi-server manager for RUNE.

Provides:
- ``MCPClient``: transport-agnostic communication with a single MCP server
  (stdio, sse, streamable-http).
- ``MCPClientManager``: manages multiple ``MCPClient`` instances keyed by
  server name, with connection deduplication, parallel connect, sensitive env
  filtering, and configurable timeouts.

Ported from ``src/mcp/client.ts``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from rune.mcp.config import MCPServerConfig
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)

# ============================================================================
# Types
# ============================================================================


@dataclass(slots=True)
class MCPTool:
    """MCP tool definition as reported by the server."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class MCPToolResult:
    """Result of an MCP tool invocation."""

    content: list[dict[str, Any]]
    is_error: bool = False


@dataclass(slots=True)
class ConnectedServer:
    """Tracks a connected MCP server and its client."""

    name: str
    config: MCPServerConfig
    client: MCPClient
    tools: list[MCPTool]
    connected: bool = True


# ============================================================================
# Timeouts (seconds)
# ============================================================================

CONNECT_TIMEOUT = 30.0
LIST_TOOLS_TIMEOUT = 15.0
CALL_TOOL_TIMEOUT = 60.0
CLOSE_TIMEOUT = 5.0

# ============================================================================
# Sensitive env filtering
# ============================================================================

_SENSITIVE_ENV_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^OPENAI_API_KEY$", re.I),
    re.compile(r"^ANTHROPIC_API_KEY$", re.I),
    re.compile(r"^BRAVE_API_KEY$", re.I),
    re.compile(r"^TELEGRAM_BOT_TOKEN$", re.I),
    re.compile(r"^DISCORD_BOT_TOKEN$", re.I),
    re.compile(r"^AWS_SECRET_ACCESS_KEY$", re.I),
    re.compile(r"^AWS_SESSION_TOKEN$", re.I),
    re.compile(r"^GITHUB_TOKEN$", re.I),
    re.compile(r"^GH_TOKEN$", re.I),
    re.compile(r"^NPM_TOKEN$", re.I),
    re.compile(r"^DATABASE_URL$", re.I),
    re.compile(r"API_KEY", re.I),
    re.compile(r"SECRET", re.I),
    re.compile(r"TOKEN$", re.I),
    re.compile(r"PASSWORD", re.I),
    re.compile(r"CREDENTIAL", re.I),
]


def filter_sensitive_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with sensitive keys removed."""
    filtered: dict[str, str] = {}
    for key, value in os.environ.items():
        if any(p.search(key) for p in _SENSITIVE_ENV_PATTERNS):
            continue
        filtered[key] = value
    return filtered


# ============================================================================
# MCPClient - single server
# ============================================================================


class MCPClient:
    """Transport-agnostic client for a single MCP server.

    Supports stdio, SSE, and streamable-http transports.
    """

    __slots__ = (
        "_server_name",
        "_config",
        "_connected",
        "_request_id",
        "_process",
        "_pending",
        "_reader_task",
    )

    def __init__(self, server_name: str, config: MCPServerConfig) -> None:
        self._server_name = server_name
        self._config = config
        self._connected = False
        self._request_id = 0
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None

    # Connection lifecycle

    async def connect(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Establish connection to the MCP server."""
        if self._connected:
            return

        transport = self._config.transport

        if transport == "stdio":
            await asyncio.wait_for(self._connect_stdio(), timeout=timeout)
        elif transport == "sse":
            await asyncio.wait_for(self._connect_sse(), timeout=timeout)
        elif transport == "streamable-http":
            await asyncio.wait_for(
                self._connect_streamable_http(), timeout=timeout
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        self._connected = True
        log.info(
            "mcp_connected",
            server=self._server_name,
            transport=transport,
        )

    async def disconnect(self) -> None:
        """Close the MCP connection."""
        if not self._connected:
            return

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None

        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(
                    self._process.wait(), timeout=CLOSE_TIMEOUT
                )
            except TimeoutError:
                self._process.kill()
            self._process = None

        # Resolve all pending futures with an error
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("MCP disconnected"))
        self._pending.clear()

        self._connected = False
        log.info("mcp_disconnected", server=self._server_name)

    # Public API

    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the MCP server."""
        result = await self._request("tools/list", {})
        raw_tools = result.get("tools", [])
        return [
            MCPTool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in raw_tools
            if t.get("name")
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool on the MCP server."""
        result = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        return MCPToolResult(
            content=result.get("content", []),
            is_error=bool(result.get("isError", False)),
        )

    # Transport: stdio

    # Shell interpreters that should not be spawned directly as MCP servers.
    _BLOCKED_COMMANDS = frozenset({
        "sh", "bash", "zsh", "fish", "csh", "tcsh", "dash", "ksh",
        "/bin/sh", "/bin/bash", "/bin/zsh", "/usr/bin/bash", "/usr/bin/zsh",
        "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh",
    })

    async def _connect_stdio(self) -> None:
        """Start the MCP server as a subprocess with stdio transport."""
        command = self._config.command
        if not command:
            raise ValueError(
                f'MCP server "{self._server_name}" missing command'
            )

        # Block direct shell interpreter execution - MCP servers should be
        # specific executables, not shells that can run arbitrary commands.
        cmd_basename = command.rsplit("/", 1)[-1].lower()
        if command.lower() in self._BLOCKED_COMMANDS or cmd_basename in self._BLOCKED_COMMANDS:
            raise ValueError(
                f'MCP server "{self._server_name}" uses blocked command "{command}" '
                f"— shell interpreters cannot be used as MCP servers"
            )

        log.info("mcp_server_spawn", server=self._server_name, command=command)

        # Build environment: filtered process env + config env overrides
        env = filter_sensitive_env()
        env.update(self._config.env)

        self._process = await asyncio.create_subprocess_exec(
            command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Start reading responses in background
        self._reader_task = asyncio.create_task(self._stdio_reader())

    async def _stdio_reader(self) -> None:
        """Read JSON-RPC responses from the subprocess stdout."""
        if self._process is None or self._process.stdout is None:
            return

        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break

                data = json_decode(line.decode("utf-8").strip())
                req_id = data.get("id")
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if "error" in data:
                        fut.set_exception(
                            RuntimeError(
                                f"MCP error: {data['error'].get('message', 'unknown')}"
                            )
                        )
                    else:
                        fut.set_result(data.get("result", {}))
            except json.JSONDecodeError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning(
                    "mcp_stdio_reader_error",
                    server=self._server_name,
                    error=str(exc),
                )
                break

    # Transport: SSE (legacy)

    async def _connect_sse(self) -> None:
        """Validate SSE transport prerequisites."""
        if not self._config.url:
            raise ValueError(
                f'MCP server "{self._server_name}" missing url for SSE'
            )
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "httpx is required for SSE transport: pip install httpx"
            ) from exc
        log.debug("mcp_sse_ready", server=self._server_name)

    # Transport: Streamable HTTP (MCP 2025+)

    async def _connect_streamable_http(self) -> None:
        """Validate streamable-http transport prerequisites."""
        if not self._config.url:
            raise ValueError(
                f'MCP server "{self._server_name}" missing url for streamable-http'
            )
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "httpx is required for streamable-http transport: pip install httpx"
            ) from exc
        log.debug("mcp_streamable_http_ready", server=self._server_name)

    # JSON-RPC request

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        if not self._connected:
            raise ConnectionError("MCP client is not connected")

        self._request_id += 1
        req_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        transport = self._config.transport

        if transport == "stdio":
            return await self._send_stdio(req_id, message)
        elif transport == "sse":
            return await self._send_sse(message)
        elif transport == "streamable-http":
            return await self._send_streamable_http(message)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    async def _send_stdio(
        self,
        req_id: int,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """Send via stdio and wait for the corresponding response."""
        if self._process is None or self._process.stdin is None:
            raise ConnectionError("MCP stdio process not available")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut

        payload = json_encode(message) + "\n"
        self._process.stdin.write(payload.encode("utf-8"))
        await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(fut, timeout=CALL_TOOL_TIMEOUT)
        except TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(
                f"MCP request {message['method']} timed out "
                f"({CALL_TOOL_TIMEOUT}s): {self._server_name}"
            ) from None

    async def _send_sse(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send via HTTP POST (SSE transport)."""
        import httpx

        headers = dict(self._config.headers) if self._config.headers else {}
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                self._config.url or "",
                json=message,
                headers=headers,
                timeout=CALL_TOOL_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise RuntimeError(
                f"MCP error: {data['error'].get('message', 'unknown')}"
            )
        return data.get("result", {})

    async def _send_streamable_http(
        self, message: dict[str, Any]
    ) -> dict[str, Any]:
        """Send via Streamable HTTP transport (MCP 2025+ standard).

        POSTs a JSON-RPC message to the server URL. The server may respond
        with a single JSON object or an NDJSON stream. We handle both:
        - If Content-Type is ``application/json``, parse as single response.
        - If Content-Type is ``text/event-stream`` or NDJSON, read lines and
          return the first JSON-RPC response matching our request id.
        """
        import httpx

        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._config.headers:
            headers.update(self._config.headers)

        async with httpx.AsyncClient() as http:
            resp = await http.post(
                self._config.url or "",
                json=message,
                headers=headers,
                timeout=CALL_TOOL_TIMEOUT,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            # Standard JSON response
            if "application/json" in content_type:
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(
                        f"MCP error: {data['error'].get('message', 'unknown')}"
                    )
                return data.get("result", {})

            # NDJSON / SSE stream - parse lines for our response
            for line in resp.text.splitlines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                # SSE data: prefix
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                try:
                    data = json_decode(line)
                except json.JSONDecodeError:
                    continue
                if data.get("id") == message.get("id"):
                    if "error" in data:
                        raise RuntimeError(
                            f"MCP error: {data['error'].get('message', 'unknown')}"
                        )
                    return data.get("result", {})

            raise RuntimeError(
                f"No matching response in streamable-http stream "
                f"for request {message.get('id')}"
            )

    # Properties

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def server_name(self) -> str:
        return self._server_name


# ============================================================================
# MCPClientManager - multi-server management
# ============================================================================


class MCPClientManager:
    """Manages multiple MCP server connections.

    Features:
    - Connection deduplication (returns cached tools or awaits in-flight task)
    - Parallel ``connect_all`` via ``asyncio.gather``
    - Sensitive env filtering for subprocess transports
    - Configurable timeouts
    """

    def __init__(self, *, close_timeout: float = CLOSE_TIMEOUT) -> None:
        self._servers: dict[str, ConnectedServer] = {}
        self._connecting: dict[str, asyncio.Task[list[MCPTool]]] = {}
        self._disconnecting: dict[str, asyncio.Task[None]] = {}
        self._close_timeout = max(0.1, min(close_timeout, 60.0))

    # Connect

    async def connect(
        self,
        name: str,
        config: MCPServerConfig,
        timeout: float | None = None,
    ) -> list[MCPTool]:
        """Connect to a single MCP server.

        Returns cached tools if already connected, or awaits an in-flight
        connection if one exists. Otherwise initiates a new connection.
        """
        if config.disabled:
            log.debug("mcp_server_disabled", server=name)
            return []

        # Already connected - return cached tools
        existing = self._servers.get(name)
        if existing is not None and existing.connected:
            return existing.tools

        # In-flight connection - await it
        in_flight = self._connecting.get(name)
        if in_flight is not None and not in_flight.done():
            return await in_flight

        # Start new connection
        task = asyncio.create_task(
            self._connect_internal(name, config, timeout)
        )
        self._connecting[name] = task
        try:
            return await task
        finally:
            self._connecting.pop(name, None)

    async def _connect_internal(
        self,
        name: str,
        config: MCPServerConfig,
        timeout: float | None,
    ) -> list[MCPTool]:
        """Internal connection logic."""
        connect_timeout = timeout or CONNECT_TIMEOUT
        try:
            client = MCPClient(name, config)
            await client.connect(timeout=connect_timeout)

            # Fetch tool list
            tools = await asyncio.wait_for(
                client.list_tools(), timeout=LIST_TOOLS_TIMEOUT
            )

            server = ConnectedServer(
                name=name,
                config=config,
                client=client,
                tools=tools,
                connected=True,
            )
            self._servers[name] = server

            log.info(
                "mcp_server_connected",
                server=name,
                tool_count=len(tools),
            )
            return tools

        except Exception as exc:
            log.error(
                "mcp_connect_failed",
                server=name,
                error=str(exc),
            )
            return []

    async def connect_all(
        self,
        configs: dict[str, MCPServerConfig],
    ) -> dict[str, list[MCPTool]]:
        """Connect to multiple servers in parallel.

        Uses ``asyncio.gather`` with ``return_exceptions=True`` so that
        one server's failure does not block the others.
        """
        names = list(configs.keys())

        async def _connect_one(name: str) -> tuple[str, list[MCPTool]]:
            tools = await self.connect(name, configs[name])
            return name, tools

        results_raw = await asyncio.gather(
            *[_connect_one(n) for n in names],
            return_exceptions=True,
        )

        results: dict[str, list[MCPTool]] = {}
        for i, result in enumerate(results_raw):
            name = names[i]
            if isinstance(result, Exception):
                log.error(
                    "mcp_connect_all_error",
                    server=name,
                    error=str(result),
                )
                results[name] = []
            else:
                results[name] = result[1]

        return results

    # Disconnect

    async def disconnect(self, name: str) -> None:
        """Disconnect a single server."""
        # Await in-flight disconnect
        pending = self._disconnecting.get(name)
        if pending is not None and not pending.done():
            await pending
            return

        task = asyncio.create_task(self._disconnect_internal(name))
        self._disconnecting[name] = task
        try:
            await task
        finally:
            self._disconnecting.pop(name, None)

    async def _disconnect_internal(self, name: str) -> None:
        """Internal disconnect logic."""
        # Wait for in-flight connect to settle first
        in_flight = self._connecting.get(name)
        if in_flight is not None and not in_flight.done():
            with contextlib.suppress(Exception):
                await in_flight

        server = self._servers.get(name)
        if server is None:
            return

        try:
            await asyncio.wait_for(
                server.client.disconnect(), timeout=self._close_timeout
            )
        except TimeoutError:
            log.warning(
                "mcp_disconnect_timeout",
                server=name,
                timeout=self._close_timeout,
            )
        except Exception as exc:
            log.warning(
                "mcp_disconnect_error",
                server=name,
                error=str(exc),
            )

        server.connected = False
        self._servers.pop(name, None)
        log.info("mcp_server_disconnected", server=name)

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        names = list(self._servers.keys())
        await asyncio.gather(
            *[self.disconnect(n) for n in names],
            return_exceptions=True,
        )

    # Tool operations

    def list_tools(self, name: str) -> list[MCPTool]:
        """Return cached tool list for a connected server."""
        server = self._servers.get(name)
        if server is None or not server.connected:
            return []
        return server.tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool on a connected server."""
        server = self._servers.get(server_name)
        if server is None or not server.connected:
            raise ConnectionError(
                f'MCP server "{server_name}" not connected'
            )

        log.debug(
            "mcp_call_tool",
            server=server_name,
            tool=tool_name,
        )

        return await asyncio.wait_for(
            server.client.call_tool(tool_name, arguments),
            timeout=CALL_TOOL_TIMEOUT,
        )

    # Query

    def get_all_tools(self) -> list[tuple[str, MCPTool]]:
        """Return ``(server_name, tool)`` pairs across all connected servers."""
        result: list[tuple[str, MCPTool]] = []
        for name, server in self._servers.items():
            if not server.connected:
                continue
            for tool in server.tools:
                result.append((name, tool))
        return result

    def find_server_for_tool(self, tool_name: str) -> str | None:
        """Find which connected server provides a given tool name."""
        for name, server in self._servers.items():
            if not server.connected:
                continue
            if any(t.name == tool_name for t in server.tools):
                return name
        return None

    def get_connected_count(self) -> int:
        """Number of currently connected servers."""
        return sum(1 for s in self._servers.values() if s.connected)

    def get_server(self, name: str) -> ConnectedServer | None:
        return self._servers.get(name)

    def get_status(self) -> list[dict[str, Any]]:
        """Connection status summary for all servers."""
        return [
            {
                "name": name,
                "connected": server.connected,
                "tool_count": len(server.tools),
            }
            for name, server in self._servers.items()
        ]


# ============================================================================
# Module singleton
# ============================================================================

_manager: MCPClientManager | None = None


def get_mcp_client_manager() -> MCPClientManager:
    """Return the module-level singleton ``MCPClientManager``."""
    global _manager
    if _manager is None:
        _manager = MCPClientManager()
    return _manager


def reset_mcp_client_manager() -> None:
    """Disconnect all and reset the singleton (for testing)."""
    global _manager
    if _manager is not None:
        # Fire-and-forget; caller should await disconnect_all() first
        # if they need graceful shutdown.
        pass
    _manager = None
