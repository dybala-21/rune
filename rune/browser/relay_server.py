"""CDP relay server for RUNE browser integration.

Combined HTTP + WebSocket server using ``websockets`` 16+ (already a
dependency via ``mcp``).  No extra packages required.

HTTP endpoints (for Chrome Extension auto-discovery):
- ``/discover`` - returns extension WebSocket URL
- ``/health``   - connection status

WebSocket endpoints:
- ``/extension`` - Chrome Extension connects here
- ``/cdp``       - Agent / Playwright connects here

Port discovery range: 19222-19231 (matches the Chrome Extension).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)

DISCOVERY_PORT_START = 19222
DISCOVERY_PORT_END = 19231


def _json_response(data: dict[str, Any], *, status: int = 200) -> Any:
    """Build a ``websockets.http11.Response`` for an HTTP JSON reply."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    body = json.dumps(data).encode()
    headers = Headers()
    headers["Content-Type"] = "application/json"
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Content-Length"] = str(len(body))
    return Response(status, "OK" if status == 200 else "Not Found", headers, body)


class RelayServer:
    """WebSocket + HTTP relay server for Chrome Extension integration.

    The extension auto-discovers this server by scanning ports 19222-19231
    for a ``/discover`` HTTP endpoint.
    """

    __slots__ = (
        "_port",
        "_server",
        "_running",
        "_extension_ws",
        "_command_futures",
        "_command_id",
    )

    def __init__(self, port: int | None = None) -> None:
        self._port = port or DISCOVERY_PORT_START
        self._server: Any = None
        self._running = False
        self._extension_ws: Any = None
        self._command_futures: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._command_id = 0

    # -- properties ---------------------------------------------------------

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_connected(self) -> bool:
        return self._extension_ws is not None

    @property
    def extension_endpoint(self) -> str:
        return f"ws://127.0.0.1:{self._port}/extension"

    @property
    def cdp_endpoint(self) -> str:
        return f"ws://127.0.0.1:{self._port}/cdp"

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the relay server, scanning the port range for an open port."""
        if self._running:
            return

        try:
            import websockets  # noqa: F401
        except ImportError:
            log.error("relay_server_requires_websockets", msg="pip install websockets")
            return

        last_exc: Exception | None = None
        for port in range(DISCOVERY_PORT_START, DISCOVERY_PORT_END + 1):
            try:
                await self._bind(port)
                self._port = port
                self._running = True
                log.info(
                    "relay_server_started",
                    port=port,
                    extension_endpoint=self.extension_endpoint,
                )
                return
            except OSError as exc:
                last_exc = exc
                continue

        log.error(
            "relay_server_no_available_port",
            range=f"{DISCOVERY_PORT_START}-{DISCOVERY_PORT_END}",
            last_error=str(last_exc),
        )

    async def _bind(self, port: int) -> None:
        """Bind to *port* using websockets 16+ API."""
        from websockets.asyncio.server import serve

        self._server = await serve(
            self._handle_ws,
            "127.0.0.1",
            port,
            process_request=self._process_request,
        )

    async def stop(self) -> None:
        """Stop the relay server and close all connections."""
        self._running = False

        if self._extension_ws is not None:
            with contextlib.suppress(Exception):
                await self._extension_ws.close()
            self._extension_ws = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for future in self._command_futures.values():
            if not future.done():
                future.cancel()
        self._command_futures.clear()

        log.info("relay_server_stopped")

    async def wait_for_extension(self, timeout: float = 10.0) -> bool:
        """Block until the extension connects or *timeout* expires."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self.is_connected:
                return True
            await asyncio.sleep(0.5)
        return False

    # -- HTTP interceptor (websockets 16 process_request) -------------------

    async def _process_request(self, connection: Any, request: Any) -> Any:
        """Intercept non-WebSocket HTTP requests for discovery/health.

        In websockets 16+, ``process_request`` receives
        ``(connection, request)`` and must return a ``Response`` to
        short-circuit, or ``None`` to proceed with WebSocket upgrade.
        """
        path = request.path

        if path == "/discover":
            return _json_response({
                "extensionEndpoint": self.extension_endpoint,
                "cdpEndpoint": self.cdp_endpoint,
                "extensionConnected": self.is_connected,
            })

        if path == "/health":
            return _json_response({
                "status": "ok",
                "extensionConnected": self.is_connected,
                "port": self._port,
            })

        # WebSocket paths proceed with upgrade.
        if path in ("/extension", "/cdp"):
            return None

        # Unknown path.
        return _json_response({"error": "Not Found"}, status=404)

    # -- WebSocket handlers -------------------------------------------------

    async def _handle_ws(self, websocket: Any) -> None:
        """Route incoming WebSocket connections by path.

        In websockets 16+, the handler receives only ``websocket``;
        the request path is available on ``websocket.request.path``.
        """
        path = websocket.request.path if websocket.request else "/"
        try:
            if path == "/extension":
                await self._handle_extension(websocket)
            elif path == "/cdp":
                await self._handle_agent(websocket)
            else:
                await websocket.close(4004, "Invalid path")
        except Exception as exc:
            log.error("relay_connection_error", path=path, error=str(exc))

    async def _handle_extension(self, websocket: Any) -> None:
        """Chrome Extension WebSocket handler."""
        log.info("extension_connected")
        self._extension_ws = websocket

        try:
            async for raw_message in websocket:
                try:
                    message = json_decode(raw_message)
                except (json.JSONDecodeError, ValueError):
                    log.warning("relay_invalid_json", source="extension")
                    continue

                cmd_id = message.get("id")
                if cmd_id is not None and cmd_id in self._command_futures:
                    future = self._command_futures.pop(cmd_id)
                    if not future.done():
                        future.set_result(message.get("result", {}))
                    log.debug("relay_response_received", id=cmd_id)
                else:
                    log.debug("relay_extension_event", method=message.get("method"))
        except Exception as exc:
            log.warning("extension_disconnected", error=str(exc))
        finally:
            self._extension_ws = None
            log.info("extension_disconnected")

    async def _handle_agent(self, websocket: Any) -> None:
        """Agent / Playwright CDP WebSocket handler."""
        log.debug("agent_client_connected")

        try:
            async for raw_message in websocket:
                try:
                    message = json_decode(raw_message)
                except (json.JSONDecodeError, ValueError):
                    log.warning("relay_invalid_json", source="agent")
                    continue

                method = message.get("method", "")
                params = message.get("params", {})
                msg_id = message.get("id")

                try:
                    result = await self.send_command(method, params)
                    response = {"id": msg_id, "result": result}
                except (ConnectionError, TimeoutError) as exc:
                    response = {"id": msg_id, "error": {"message": str(exc)}}

                await websocket.send(json_encode(response))
        except Exception as exc:
            log.debug("agent_client_disconnected", error=str(exc))

    # -- CDP command relay --------------------------------------------------

    async def send_command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a CDP command to the Chrome Extension and await the response."""
        if self._extension_ws is None:
            raise ConnectionError("No Chrome Extension connected to relay")

        self._command_id += 1
        cmd_id = self._command_id

        message = {
            "id": cmd_id,
            "method": method,
            "params": params or {},
        }

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._command_futures[cmd_id] = future

        try:
            await self._extension_ws.send(json_encode(message))
            log.debug("relay_command_sent", id=cmd_id, method=method)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except TimeoutError:
            self._command_futures.pop(cmd_id, None)
            raise TimeoutError(
                f"CDP command {method} (id={cmd_id}) timed out after {timeout}s"
            ) from None
        finally:
            self._command_futures.pop(cmd_id, None)
