"""Daemon Unix socket client for RUNE.

Ported from src/daemon/client.ts - async client that communicates with
the RUNE daemon over a Unix domain socket using newline-delimited JSON.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from rune.daemon.types import (
    CommandType,
    DaemonCommand,
    DaemonResponse,
    DaemonStatus,
)
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_DEFAULT_SOCKET_PATH = rune_home() / "daemon.sock"


class DaemonClient:
    """Async client for communicating with the RUNE daemon."""

    __slots__ = ("_socket_path", "_reader", "_writer")

    def __init__(self, socket_path: Path | str | None = None) -> None:
        self._socket_path = Path(socket_path) if socket_path else _DEFAULT_SOCKET_PATH
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Connect to the daemon's Unix socket."""
        if not self._socket_path.exists():
            raise FileNotFoundError(
                f"Daemon socket not found at {self._socket_path}. "
                "Is the daemon running?"
            )

        self._reader, self._writer = await asyncio.open_unix_connection(
            str(self._socket_path)
        )
        log.debug("daemon_client_connected", socket=str(self._socket_path))

    async def disconnect(self) -> None:
        """Disconnect from the daemon."""
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None
        log.debug("daemon_client_disconnected")

    async def send_command(self, command: DaemonCommand) -> DaemonResponse:
        """Send a command to the daemon and wait for a response."""
        if self._writer is None or self._reader is None:
            raise ConnectionError("Not connected to daemon")

        payload = json_encode({
            "type": command.type,
            "payload": command.payload,
            "request_id": command.request_id,
        })

        self._writer.write(payload.encode("utf-8") + b"\n")
        await self._writer.drain()

        # Read response (newline-delimited JSON)
        line = await asyncio.wait_for(self._reader.readline(), timeout=30.0)
        if not line:
            raise ConnectionError("Daemon closed the connection")

        data = json_decode(line.decode("utf-8"))
        return DaemonResponse(
            request_id=data.get("request_id", command.request_id),
            success=data.get("success", False),
            data=data.get("data"),
            error=data.get("error"),
        )

    async def status(self) -> DaemonStatus:
        """Query daemon status."""
        command = DaemonCommand(type=CommandType.STATUS)
        response = await self.send_command(command)

        if not response.success or response.data is None:
            raise RuntimeError(response.error or "Failed to get daemon status")

        d = response.data
        return DaemonStatus(
            running=d.get("running", False),
            uptime_seconds=d.get("uptime_seconds", 0.0),
            active_tasks=d.get("active_tasks", 0),
            queued_tasks=d.get("queued_tasks", 0),
            channels=d.get("channels", []),
        )

    async def execute(self, goal: str, sender_id: str = "") -> str:
        """Submit a goal for execution. Returns the request_id."""
        command = DaemonCommand(
            type=CommandType.EXECUTE,
            payload={"goal": goal, "sender_id": sender_id},
        )
        response = await self.send_command(command)

        if not response.success:
            raise RuntimeError(response.error or "Failed to submit goal")

        return response.request_id

    async def cancel(self, request_id: str) -> bool:
        """Cancel a running task. Returns True if cancellation was accepted."""
        command = DaemonCommand(
            type=CommandType.CANCEL,
            payload={"request_id": request_id},
        )
        response = await self.send_command(command)
        return response.success
