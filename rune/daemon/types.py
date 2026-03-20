"""Daemon type definitions for RUNE.

Ported from src/daemon/types.ts - command, response, and status types
for the RUNE daemon's Unix socket protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class CommandType(StrEnum):
    EXECUTE = "execute"
    STATUS = "status"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class DaemonCommand:
    """A command sent to the daemon over the Unix socket."""

    type: CommandType
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex[:16])


@dataclass(slots=True)
class DaemonResponse:
    """A response from the daemon to a command."""

    request_id: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class DaemonStatus:
    """Current status of the daemon."""

    running: bool
    uptime_seconds: float
    active_tasks: int
    queued_tasks: int
    channels: list[str] = field(default_factory=list)
