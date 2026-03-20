"""Environment sensor for real-time monitoring in RUNE.

Watches file changes, git status mutations, and idle periods,
emitting SensorEvent objects via EventEmitter.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from rune.utils.events import EventEmitter
from rune.utils.logger import get_logger

log = get_logger(__name__)

SensorEventType = Literal["file_change", "git_change", "process_change", "idle"]

# Idle detection threshold in seconds
_IDLE_THRESHOLD = 120.0

# Git poll interval in seconds
_GIT_POLL_INTERVAL = 30.0


@dataclass(slots=True)
class SensorEvent:
    """An event produced by the EnvironmentSensor."""

    type: SensorEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EnvironmentSensor(EventEmitter):
    """Monitors the workspace environment and emits change events.

    Events emitted:
        sensor_event: Fired with a SensorEvent whenever a change is detected.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        *,
        idle_threshold: float = _IDLE_THRESHOLD,
        git_poll_interval: float = _GIT_POLL_INTERVAL,
    ) -> None:
        super().__init__()
        self._workspace_root = workspace_root
        self._idle_threshold = idle_threshold
        self._git_poll_interval = git_poll_interval
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._last_activity: float = time.monotonic()
        self._last_git_status: str = ""

    # Public API

    async def start(self) -> None:
        """Start all monitoring loops (file watcher, git monitor, idle detector)."""
        if self._running:
            return
        self._running = True
        self._last_activity = time.monotonic()

        self._tasks.append(asyncio.create_task(self._watch_files()))
        self._tasks.append(asyncio.create_task(self._monitor_git()))
        self._tasks.append(asyncio.create_task(self._detect_idle()))

        log.info("environment_sensor_started", workspace=self._workspace_root)

    async def stop(self) -> None:
        """Stop all monitoring loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        log.info("environment_sensor_stopped")

    def record_activity(self) -> None:
        """Record a user activity event (resets idle timer)."""
        self._last_activity = time.monotonic()

    # Monitoring loops

    async def _watch_files(self) -> None:
        """Watch the workspace for file changes using watchfiles."""
        try:
            from watchfiles import Change, awatch
        except ImportError:
            log.debug("watchfiles_not_available", msg="file watching disabled")
            return

        try:
            async for changes in awatch(self._workspace_root, stop_event=None):
                if not self._running:
                    break

                change_list: list[dict[str, str]] = []
                for change_type, path in changes:
                    change_list.append({
                        "change": change_type.name if isinstance(change_type, Change) else str(change_type),
                        "path": path,
                    })

                event = SensorEvent(
                    type="file_change",
                    data={"changes": change_list},
                )
                await self.emit("sensor_event", event)
                self.record_activity()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("file_watcher_error", error=str(exc))

    async def _monitor_git(self) -> None:
        """Periodically check git status for changes."""
        try:
            while self._running:
                await asyncio.sleep(self._git_poll_interval)
                if not self._running:
                    break

                try:
                    proc = await asyncio.create_subprocess_exec(
                        "git", "status", "--short",
                        cwd=self._workspace_root,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
                    current_status = stdout.decode("utf-8", errors="replace").strip()

                    if current_status != self._last_git_status:
                        event = SensorEvent(
                            type="git_change",
                            data={
                                "previous": self._last_git_status,
                                "current": current_status,
                            },
                        )
                        self._last_git_status = current_status
                        await self.emit("sensor_event", event)

                except (TimeoutError, OSError) as exc:
                    log.debug("git_monitor_error", error=str(exc))

        except asyncio.CancelledError:
            pass

    async def _detect_idle(self) -> None:
        """Detect periods of user inactivity."""
        try:
            idle_emitted = False
            while self._running:
                await asyncio.sleep(10.0)
                if not self._running:
                    break

                elapsed = time.monotonic() - self._last_activity
                if elapsed >= self._idle_threshold and not idle_emitted:
                    event = SensorEvent(
                        type="idle",
                        data={"idle_seconds": elapsed},
                    )
                    await self.emit("sensor_event", event)
                    idle_emitted = True
                elif elapsed < self._idle_threshold:
                    idle_emitted = False

        except asyncio.CancelledError:
            pass
