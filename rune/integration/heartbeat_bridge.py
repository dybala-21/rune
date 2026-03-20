"""Heartbeat integration for RUNE proactive system.

Bridges the HeartbeatScheduler with the ProactiveEngine by registering
periodic cron tasks that trigger proactive evaluation checks.
"""

from __future__ import annotations

from typing import Any

from rune.daemon.heartbeat import HeartbeatScheduler
from rune.proactive.engine import ProactiveEngine
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Default cron: every 5 minutes
_DEFAULT_PROACTIVE_CRON = "*/5 * * * *"


class HeartbeatEngineBridge:
    """Bridges HeartbeatScheduler and ProactiveEngine.

    Registers heartbeat tasks that periodically run the proactive
    engine's evaluation pipeline and process the resulting suggestions.
    """

    __slots__ = ("_heartbeat", "_proactive_engine", "_running", "_context_provider")

    def __init__(
        self,
        heartbeat: HeartbeatScheduler,
        proactive_engine: ProactiveEngine,
        context_provider: Any = None,
    ) -> None:
        self._heartbeat = heartbeat
        self._proactive_engine = proactive_engine
        self._running = False
        self._context_provider = context_provider

    async def start(self) -> None:
        """Start the bridge and register heartbeat tasks for proactive checks."""
        if self._running:
            return
        self._running = True

        self._heartbeat.add_task(
            name="proactive_check",
            cron_expr=_DEFAULT_PROACTIVE_CRON,
            callback=self._run_proactive_check,
        )

        log.info(
            "heartbeat_bridge_started",
            cron=_DEFAULT_PROACTIVE_CRON,
        )

    async def stop(self) -> None:
        """Stop the bridge and remove heartbeat tasks."""
        self._running = False
        self._heartbeat.remove_task("proactive_check")
        log.info("heartbeat_bridge_stopped")

    async def _run_proactive_check(self) -> None:
        """Execute a proactive evaluation check.

        Gathers context, runs the engine pipeline, and logs results.
        This is called by the heartbeat scheduler at the configured interval.
        """
        if not self._running:
            return

        try:
            # Build context for the engine
            context: dict[str, Any] = {}

            if self._context_provider is not None:
                if hasattr(self._context_provider, "gather"):
                    import asyncio

                    result = self._context_provider.gather()
                    if asyncio.iscoroutine(result):
                        awareness_ctx = await result
                    else:
                        awareness_ctx = result

                    context["awareness"] = awareness_ctx

            # Run the proactive engine
            suggestions = await self._proactive_engine.evaluate(context)

            if suggestions:
                log.info(
                    "heartbeat_proactive_suggestions",
                    count=len(suggestions),
                    types=[s.type for s in suggestions],
                )

        except Exception as exc:
            log.error("heartbeat_proactive_check_failed", error=str(exc))
