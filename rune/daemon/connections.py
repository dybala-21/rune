"""Connect optional services without delaying local readiness."""

import asyncio

from rune.daemon.lifecycle import bounded
from rune.utils.logger import get_logger

log = get_logger(__name__)


class Connections:
    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task] = {}
        self.work: dict[str, asyncio.Task] = {}
        self.status: dict[str, str] = {}

    def start(self, name, connect, *, timeout=75) -> None:
        self.status[name] = "connecting"

        async def run():
            try:
                self.work[name] = asyncio.create_task(connect())
                self.status[name] = await bounded(self.work[name], timeout) or "ready"
            except asyncio.CancelledError:
                self.status[name] = "stopped"
                raise
            except Exception as exc:
                self.status[name] = "failed"
                log.warning("optional_connection_failed", name=name, error=str(exc))

        self.tasks[name] = asyncio.create_task(run())

    async def stop(self, cleanup) -> None:
        for task in self.tasks.values():
            task.cancel()
        for name, task in self.tasks.items():
            await cleanup.cancel(f"connection_{name}", task)
        for name, task in self.work.items():
            await cleanup.cancel(f"connection_work_{name}", task)
