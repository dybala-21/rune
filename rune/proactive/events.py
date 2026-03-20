"""Event-driven awareness for RUNE.

Transforms raw EnvironmentSensor events into higher-level AwarenessEvents
and schedules periodic time-based triggers.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from rune.proactive.sensor import EnvironmentSensor, SensorEvent
from rune.utils.events import EventEmitter
from rune.utils.logger import get_logger

log = get_logger(__name__)

EventType = Literal["idle", "time_trigger", "task_complete", "file_change", "error_detected"]

# Default interval for periodic time triggers (in seconds)
_TIME_TRIGGER_INTERVAL = 300.0


@dataclass(slots=True)
class AwarenessEvent:
    """A high-level awareness event derived from sensor data or timers."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EventDrivenAwareness(EventEmitter):
    """Bridges raw sensor events to awareness-level events.

    Events emitted:
        awareness_event: Fired with an AwarenessEvent.
    """

    def __init__(
        self,
        sensor: EnvironmentSensor,
        *,
        time_trigger_interval: float = _TIME_TRIGGER_INTERVAL,
    ) -> None:
        super().__init__()
        self._sensor = sensor
        self._time_trigger_interval = time_trigger_interval
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    # Public API

    async def start(self) -> None:
        """Start listening to sensor events and scheduling time triggers."""
        if self._running:
            return
        self._running = True

        self._sensor.on("sensor_event", self._on_sensor_event)
        self._tasks.append(asyncio.create_task(self._schedule_time_triggers()))

        log.info("event_driven_awareness_started")

    async def stop(self) -> None:
        """Stop listening and cancel scheduled triggers."""
        self._running = False
        self._sensor.off("sensor_event", self._on_sensor_event)

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        log.info("event_driven_awareness_stopped")

    async def emit_task_complete(self, task_data: dict[str, Any]) -> None:
        """Manually emit a task-completion awareness event."""
        event = AwarenessEvent(
            type="task_complete",
            data=task_data,
        )
        await self.emit("awareness_event", event)

    async def emit_error_detected(self, error_data: dict[str, Any]) -> None:
        """Manually emit an error-detection awareness event."""
        event = AwarenessEvent(
            type="error_detected",
            data=error_data,
        )
        await self.emit("awareness_event", event)

    # Internal handlers

    async def _on_sensor_event(self, event: SensorEvent) -> None:
        """Transform a SensorEvent into an AwarenessEvent."""
        mapping: dict[str, EventType] = {
            "file_change": "file_change",
            "git_change": "file_change",
            "idle": "idle",
            "process_change": "file_change",
        }
        awareness_type = mapping.get(event.type, "file_change")

        awareness_event = AwarenessEvent(
            type=awareness_type,
            data={"sensor_type": event.type, **event.data},
            timestamp=event.timestamp,
        )

        log.debug(
            "awareness_event_generated",
            sensor_type=event.type,
            awareness_type=awareness_type,
        )
        await self.emit("awareness_event", awareness_event)

    async def _schedule_time_triggers(self) -> None:
        """Emit periodic time-trigger awareness events."""
        try:
            while self._running:
                await asyncio.sleep(self._time_trigger_interval)
                if not self._running:
                    break

                event = AwarenessEvent(
                    type="time_trigger",
                    data={
                        "interval_seconds": self._time_trigger_interval,
                        "hour": datetime.now().hour,
                    },
                )
                await self.emit("awareness_event", event)
                log.debug("time_trigger_emitted")

        except asyncio.CancelledError:
            pass
