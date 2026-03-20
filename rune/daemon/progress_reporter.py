"""Real-time progress reporting for RUNE daemon clients.

Ported from src/daemon/progress-reporter.ts - broadcasts agent progress
updates to connected daemon clients via Unix domain sockets.

Phase-based narrative progression:
    thinking -> researching -> analyzing -> executing -> composing
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rune.utils.fast_serde import json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Phase system

class Phase(StrEnum):
    THINKING = "thinking"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    COMPOSING = "composing"


def detect_phase(tool_name: str, *, is_late_stage: bool = False) -> Phase:
    """Determine the current phase from a tool name.

    Mirrors the TypeScript ``detectPhase`` logic.
    """
    name = tool_name.replace("_", ".")

    # Execution / mutation tools
    if any(
        name.startswith(p)
        for p in ("file.edit", "file.write", "file.delete", "bash")
    ):
        return Phase.EXECUTING

    # Web / browser / search tools
    if name.startswith(("web.", "browser.")):
        return Phase.RESEARCHING

    # File read / search tools
    if any(
        name.startswith(p)
        for p in ("file.read", "file.search", "file.list", "project.map", "memory.search")
    ):
        return Phase.RESEARCHING

    # Code analysis tools
    if name.startswith("code."):
        return Phase.ANALYZING

    # Think - late stage means composing, early stage means thinking
    if name.startswith("think"):
        return Phase.COMPOSING if is_late_stage else Phase.THINKING

    # Delegate
    if name.startswith("delegate"):
        return Phase.ANALYZING

    return Phase.THINKING


# Config

@dataclass(slots=True)
class ProgressReporterConfig:
    """Configuration for throttled progress updates."""

    min_edit_interval_ms: float = 1000.0
    """Minimum interval between edits (milliseconds)."""

    max_steps: int = 200
    """Agent loop maximum step count."""


#: Per-channel defaults for edit throttling.
CHANNEL_DEFAULTS: dict[str, ProgressReporterConfig] = {
    "telegram": ProgressReporterConfig(min_edit_interval_ms=1000, max_steps=200),
    "discord": ProgressReporterConfig(min_edit_interval_ms=2000, max_steps=200),
    "slack": ProgressReporterConfig(min_edit_interval_ms=1500, max_steps=200),
}

#: Maximum message lengths per channel.
CHANNEL_MAX_LENGTH: dict[str, int] = {
    "telegram": 4096,
    "discord": 2000,
    "slack": 4000,
    "googlechat": 8000,
}


# Progress update dataclass

@dataclass(slots=True)
class ProgressUpdate:
    """A single progress update sent to daemon clients."""

    run_id: str
    phase: Phase
    action: str
    """Human-readable description of what the agent is doing."""
    step_number: int = 0
    total_steps: int = 0
    tool_name: str = ""
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "action": self.action,
            "step_number": self.step_number,
            "total_steps": self.total_steps,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            **self.extra,
        }


# ProgressReporter

class ProgressReporter:
    """Broadcasts agent progress to connected daemon clients.

    Each connected client is represented as an ``asyncio.StreamWriter``.
    Updates are throttled per the configured interval and serialized as
    newline-delimited JSON.
    """

    def __init__(
        self,
        config: ProgressReporterConfig | None = None,
        *,
        channel: str | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        elif channel and channel in CHANNEL_DEFAULTS:
            self._config = CHANNEL_DEFAULTS[channel]
        else:
            self._config = ProgressReporterConfig()

        self._clients: list[asyncio.StreamWriter] = []
        self._last_send_time: float = 0.0
        self._current_phase: Phase = Phase.THINKING
        self._step_count: int = 0

    # -- Client management ---------------------------------------------------

    def add_client(self, writer: asyncio.StreamWriter) -> None:
        """Register a client writer for progress broadcasts."""
        self._clients.append(writer)
        log.debug("progress_client_added", total=len(self._clients))

    def remove_client(self, writer: asyncio.StreamWriter) -> None:
        """Unregister a client writer."""
        with contextlib.suppress(ValueError):
            self._clients.remove(writer)
        log.debug("progress_client_removed", total=len(self._clients))

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- Sending -------------------------------------------------------------

    async def send_update(self, update: ProgressUpdate) -> None:
        """Send a progress update to all clients, respecting throttle.

        If the minimum edit interval has not elapsed since the last
        update, the call is silently dropped.
        """
        now = time.monotonic()
        elapsed_ms = (now - self._last_send_time) * 1000.0
        if elapsed_ms < self._config.min_edit_interval_ms:
            return

        self._last_send_time = now
        self._current_phase = update.phase
        self._step_count = update.step_number

        await self._broadcast_raw(update.to_dict())

    async def broadcast(self, event: str, data: dict[str, Any]) -> None:
        """Broadcast an arbitrary event dict to all connected clients.

        Unlike :meth:`send_update`, this is not throttled.
        """
        payload = {"event": event, "data": data}
        await self._broadcast_raw(payload)

    async def _broadcast_raw(self, payload: dict[str, Any]) -> None:
        """Serialize *payload* as JSON and write to every client."""
        if not self._clients:
            return

        data = json_encode(payload).encode("utf-8") + b"\n"
        dead: list[asyncio.StreamWriter] = []

        for writer in self._clients:
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionError, OSError):
                dead.append(writer)
            except Exception as exc:  # noqa: BLE001
                log.warning("progress_send_failed", error=str(exc))
                dead.append(writer)

        for writer in dead:
            self.remove_client(writer)

    # -- Convenience helpers -------------------------------------------------

    async def report_tool_call(
        self,
        run_id: str,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        step_number: int = 0,
        total_steps: int = 0,
    ) -> None:
        """Build and send a progress update for a tool invocation."""
        is_late = step_number > (total_steps * 0.7) if total_steps else False
        phase = detect_phase(tool_name, is_late_stage=is_late)
        action = _describe_action(tool_name, args or {})

        update = ProgressUpdate(
            run_id=run_id,
            phase=phase,
            action=action,
            step_number=step_number,
            total_steps=total_steps,
            tool_name=tool_name,
        )
        await self.send_update(update)

    async def report_phase_change(
        self, run_id: str, phase: Phase, action: str
    ) -> None:
        """Send a phase-change update (always sent, bypasses throttle)."""
        self._current_phase = phase
        payload: dict[str, Any] = {
            "event": "progress",
            "data": {
                "run_id": run_id,
                "phase": phase.value,
                "action": action,
            },
        }
        await self._broadcast_raw(payload)

    @property
    def current_phase(self) -> Phase:
        return self._current_phase


# Action description (narrative)

def _describe_action(tool_name: str, args: dict[str, Any]) -> str:
    """Convert a tool call to a short human-readable description."""
    name = tool_name.replace("_", ".")

    # -- File --
    if name.startswith("file.read"):
        p = str(args.get("path") or args.get("file_path") or "")
        filename = p.rsplit("/", 1)[-1] or "file"
        return f"Reading {filename}"

    if name.startswith("file.search"):
        q = str(args.get("query") or args.get("pattern") or "")[:25]
        return f'Searching "{q}"'

    if name.startswith("file.edit"):
        p = str(args.get("path") or args.get("file_path") or "")
        return f"Editing {p.rsplit('/', 1)[-1] or 'file'}"

    if name.startswith("file.write"):
        p = str(args.get("path") or args.get("file_path") or "")
        return f"Writing {p.rsplit('/', 1)[-1] or 'file'}"

    if name.startswith("file.list"):
        return "Exploring project structure"

    # -- Web --
    if name.startswith("web.search"):
        q = str(args.get("query") or "")[:30]
        return f'Searching web for "{q}"'

    if name.startswith("web.fetch"):
        url = str(args.get("url") or "")
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).hostname or ""
            domain = domain.removeprefix("www.")
            return f"Reading {domain}"
        except Exception:
            return "Reading web page"

    # -- Browser --
    if name.startswith("browser.navigate"):
        url = str(args.get("url") or "")
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).hostname or ""
            domain = domain.removeprefix("www.")
            return f"Navigating to {domain}"
        except Exception:
            return "Navigating to website"

    # -- Code analysis --
    if name.startswith("code."):
        return "Analyzing code"

    # -- Bash --
    if name.startswith("bash"):
        cmd = str(args.get("command") or "")[:40]
        return f"Running command: {cmd}" if cmd else "Running command"

    # -- Think --
    if name.startswith("think"):
        return "Thinking"

    # -- Delegate --
    if name.startswith("delegate"):
        return "Delegating sub-task"

    return f"Using {tool_name}"
