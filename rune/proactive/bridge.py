"""Proactive Agent Bridge for RUNE.

Connects the proactive engine's suggestions to actual agent execution,
providing polling, retry logic, rate limiting, and execution history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from rune.proactive.engine import ProactiveEngine
from rune.proactive.feedback import FeedbackLearner
from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Types

class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass(slots=True)
class ExecutionRecord:
    """Record of a suggestion execution attempt."""

    suggestion_id: str
    suggestion_title: str
    status: ExecutionStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    attempt: int = 1
    duration_ms: float = 0.0


@dataclass(slots=True)
class BridgeConfig:
    """Configuration for the proactive agent bridge."""

    poll_interval_seconds: float = 60.0
    max_retries: int = 2
    max_executions_per_hour: int = 5
    min_confidence: float = 0.5
    backoff_base_seconds: float = 2.0
    max_steps: int = 50
    timeout_ms: int = 180_000


# Type alias for agent factory: takes a goal string and returns a coroutine
# that resolves to a result dict.
AgentFactory = Callable[[str], Coroutine[Any, Any, dict[str, Any]]]


# ProactiveAgentBridge

class ProactiveAgentBridge:
    """Bridge that polls the proactive engine and executes actionable suggestions.

    The bridge runs a background polling loop that:
    1. Asks the engine for new suggestions.
    2. Filters by confidence threshold.
    3. Rate-limits executions (max N per hour).
    4. Creates an agent session via the agent factory for each suggestion.
    5. Retries on failure with exponential backoff (max 2 retries).
    6. Tracks execution history.
    """

    __slots__ = (
        "_engine",
        "_agent_factory",
        "_config",
        "_history",
        "_running",
        "_poll_task",
        "_context",
        "_feedback_learner",
        "_autonomous_executor",
    )

    def __init__(
        self,
        engine: ProactiveEngine,
        agent_factory: AgentFactory,
        config: BridgeConfig | None = None,
        context: dict[str, Any] | None = None,
        feedback_learner: FeedbackLearner | None = None,
        autonomous_executor: Any | None = None,
    ) -> None:
        self._engine = engine
        self._agent_factory = agent_factory
        self._config = config or BridgeConfig()
        self._history: list[ExecutionRecord] = []
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._context = context or {}
        self._feedback_learner = feedback_learner
        self._autonomous_executor = autonomous_executor

        # Subscribe to engine events for auto-execution dispatch
        self._engine.on("suggestion", self._on_suggestion_event)
        self._engine.on("intervention", self._on_intervention_event)

    # Properties

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def history(self) -> list[ExecutionRecord]:
        return list(self._history)

    @property
    def config(self) -> BridgeConfig:
        return self._config

    # Lifecycle

    def start(self) -> None:
        """Begin polling the engine for actionable suggestions."""
        if self._running:
            log.warning("bridge_already_running")
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info(
            "bridge_started",
            poll_interval=self._config.poll_interval_seconds,
            max_per_hour=self._config.max_executions_per_hour,
        )

    def stop(self) -> None:
        """Stop polling and cancel the background task."""
        if not self._running:
            return

        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None

        # Unsubscribe from engine events
        self._engine.off("suggestion", self._on_suggestion_event)
        self._engine.off("intervention", self._on_intervention_event)
        log.info("bridge_stopped", history_size=len(self._history))

    # Engine event handlers (auto-execution dispatch)

    def _on_suggestion_event(self, suggestions: list[Suggestion]) -> None:
        """Handle suggestions emitted by the engine.

        Dispatches high-confidence suggestions for automatic execution
        if the bridge is running.
        """
        if not self._running:
            return
        for suggestion in suggestions:
            if suggestion.confidence >= self._config.min_confidence:
                if not self._is_rate_limited():
                    asyncio.create_task(self.execute_suggestion(suggestion))

    def _on_intervention_event(self, interventions: list[Suggestion]) -> None:
        """Handle high-confidence intervention suggestions.

        Interventions bypass the normal confidence threshold and are executed
        immediately if the bridge is running.
        """
        if not self._running:
            return
        for suggestion in interventions:
            if not self._is_rate_limited():
                asyncio.create_task(self.execute_suggestion(suggestion))

    # Polling

    async def _poll_loop(self) -> None:
        """Main polling loop - runs until stopped."""
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("bridge_poll_error", error=str(exc))

            try:
                await asyncio.sleep(self._config.poll_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        """Single poll iteration: evaluate engine and execute suggestions."""
        suggestions = await self._engine.evaluate(self._context)

        for suggestion in suggestions:
            if suggestion.confidence < self._config.min_confidence:
                self._record(
                    suggestion,
                    ExecutionStatus.SKIPPED,
                    error="Below confidence threshold",
                )
                continue

            if self._is_rate_limited():
                log.debug("bridge_rate_limited")
                break

            await self.execute_suggestion(suggestion)

    # Execution with retry

    async def execute_suggestion(self, suggestion: Suggestion) -> ExecutionRecord:
        """Execute a suggestion by creating an agent session.

        Retries up to ``max_retries`` times with exponential backoff on failure.
        """
        max_attempts = 1 + self._config.max_retries
        last_record: ExecutionRecord | None = None

        for attempt in range(1, max_attempts + 1):
            start = datetime.now(UTC)
            try:
                goal = f"{suggestion.title}: {suggestion.description}"
                result = await self._agent_factory(goal)

                duration = (datetime.now(UTC) - start).total_seconds() * 1000
                success = result.get("success", True) if isinstance(result, dict) else True

                if success:
                    record = self._record(
                        suggestion,
                        ExecutionStatus.SUCCESS,
                        attempt=attempt,
                        duration_ms=duration,
                    )
                    self._engine.record_feedback(suggestion.id, True)
                    if self._feedback_learner is not None:
                        self._feedback_learner.record_feedback(suggestion, "accepted")
                    # Reflexion: record successful task outcome
                    try:
                        from rune.proactive.reflexion import get_reflexion_learner
                        get_reflexion_learner().record_task_outcome({
                            "domain": suggestion.type,
                            "success": True,
                            "goal": suggestion.title,
                            "steps_taken": attempt,
                            "duration_ms": duration,
                        })
                    except Exception:
                        pass
                    self._record_to_autonomous_executor(
                        suggestion, success=True, duration_ms=duration,
                        result_summary=str(result.get("output", ""))[:200] if isinstance(result, dict) else "",
                    )
                    return record
                else:
                    error_msg = (
                        result.get("error", "Agent returned failure")
                        if isinstance(result, dict)
                        else "Agent returned failure"
                    )
                    last_record = self._record(
                        suggestion,
                        ExecutionStatus.FAILURE,
                        error=error_msg,
                        attempt=attempt,
                        duration_ms=duration,
                    )

            except Exception as exc:
                duration = (datetime.now(UTC) - start).total_seconds() * 1000
                last_record = self._record(
                    suggestion,
                    ExecutionStatus.FAILURE,
                    error=str(exc),
                    attempt=attempt,
                    duration_ms=duration,
                )

            # Exponential backoff before retry (unless this was the last attempt)
            if attempt < max_attempts:
                backoff = self._config.backoff_base_seconds * (2 ** (attempt - 1))
                log.debug(
                    "bridge_retry_backoff",
                    suggestion=suggestion.title,
                    attempt=attempt,
                    backoff_seconds=backoff,
                )
                await asyncio.sleep(backoff)

        # All attempts exhausted
        self._engine.record_feedback(suggestion.id, False)
        if self._feedback_learner is not None:
            self._feedback_learner.record_feedback(suggestion, "dismissed")

        # Reflexion learning: record rejection + failed outcome
        try:
            from rune.proactive.reflexion import get_reflexion_learner
            learner = get_reflexion_learner()
            learner.record_rejection(
                event_type=suggestion.type,
                suggestion_type=suggestion.type,
                score=suggestion.confidence,
                reason="execution_failed",
            )
            learner.record_task_outcome({
                "domain": suggestion.type,
                "success": False,
                "goal": suggestion.title,
                "error": last_record.error if last_record else "unknown",
                "steps_taken": max_attempts,
                "duration_ms": last_record.duration_ms if last_record else 0.0,
            })
        except Exception:
            pass
        self._record_to_autonomous_executor(
            suggestion,
            success=False,
            duration_ms=last_record.duration_ms if last_record else 0.0,
            result_summary=last_record.error or "" if last_record else "",
        )
        assert last_record is not None
        return last_record

    # AutonomousExecutor recording

    def _record_to_autonomous_executor(
        self,
        suggestion: Suggestion,
        *,
        success: bool,
        duration_ms: float = 0.0,
        result_summary: str = "",
    ) -> None:
        """Record a full execution to the AutonomousExecutor ledger.

        Mirrors TS proactive-agent-bridge.ts lines 140-183 which builds
        a 12-field ``AutonomousExecution`` and calls
        ``autonomy.recordExecution(execution)``.
        """
        executor = self._autonomous_executor
        if executor is None:
            try:
                from rune.agent.autonomous import get_autonomous_executor
                executor = get_autonomous_executor()
            except Exception:
                return

        try:
            import time as _time
            from uuid import uuid4

            from rune.agent.autonomous import AutonomousExecution

            feedback = "approved" if success else "full_revert"
            domain = getattr(suggestion, "type", "unknown") or "unknown"
            # Map common suggestion types to TaskDomain literals
            domain_map = {
                "git": "git", "build": "build", "file": "file",
                "browser": "browser", "system": "system", "notify": "notify",
                "cleanup": "cleanup",
            }
            resolved_domain = domain_map.get(domain, "unknown")

            execution = AutonomousExecution(
                id=uuid4().hex[:16],
                timestamp=_time.monotonic(),
                level=executor._policy.domain_levels.get(resolved_domain, 0),
                domain=resolved_domain,
                description=suggestion.title,
                action=f"{suggestion.title}: {suggestion.description}"[:120],
                success=success,
                result_summary=result_summary[:200] if result_summary else "",
                duration_ms=duration_ms,
                reversible=False,
                user_feedback=feedback,
            )
            executor.record_execution(execution)
            log.debug(
                "bridge_autonomous_recorded",
                execution_id=execution.id,
                domain=resolved_domain,
                success=success,
                duration_ms=round(duration_ms, 1),
            )
        except Exception as exc:
            log.warning(
                "bridge_autonomous_record_failed",
                suggestion_id=suggestion.id,
                error=str(exc),
            )

    # Rate limiting

    def _is_rate_limited(self) -> bool:
        """Check if we have exceeded the hourly execution limit."""
        now = datetime.now(UTC)
        one_hour_ago = now.timestamp() - 3600.0

        recent_count = sum(
            1
            for r in self._history
            if r.status == ExecutionStatus.SUCCESS
            and r.timestamp.timestamp() > one_hour_ago
        )
        return recent_count >= self._config.max_executions_per_hour

    # History

    def _record(
        self,
        suggestion: Suggestion,
        status: ExecutionStatus,
        *,
        error: str | None = None,
        attempt: int = 1,
        duration_ms: float = 0.0,
    ) -> ExecutionRecord:
        """Create and store an execution record."""
        record = ExecutionRecord(
            suggestion_id=suggestion.id,
            suggestion_title=suggestion.title,
            status=status,
            error=error,
            attempt=attempt,
            duration_ms=duration_ms,
        )
        self._history.append(record)
        log.info(
            "bridge_execution",
            suggestion=suggestion.title,
            status=status.value,
            attempt=attempt,
            error=error,
        )
        return record

    def get_history(
        self,
        *,
        status: ExecutionStatus | None = None,
        limit: int = 50,
    ) -> list[ExecutionRecord]:
        """Retrieve execution history, optionally filtered by status."""
        records = self._history
        if status is not None:
            records = [r for r in records if r.status == status]
        return records[-limit:]

    def clear_history(self) -> None:
        """Clear all execution history."""
        self._history.clear()


# Module-level factory

_bridge: ProactiveAgentBridge | None = None


def initialize_proactive_bridge(
    engine: ProactiveEngine,
    agent_factory: AgentFactory,
    config: BridgeConfig | None = None,
    context: dict[str, Any] | None = None,
    feedback_learner: FeedbackLearner | None = None,
    autonomous_executor: Any | None = None,
) -> ProactiveAgentBridge:
    """Create or replace the singleton ProactiveAgentBridge.

    Parameters
    ----------
    engine:
        The proactive suggestion engine to poll.
    agent_factory:
        Async callable that takes a goal string and returns a result dict.
    config:
        Optional bridge configuration.
    context:
        Optional context dict passed to the engine on each poll.
    feedback_learner:
        Optional FeedbackLearner for recording feedback with full
        suggestion context (type, confidence, description).
    autonomous_executor:
        Optional :class:`~rune.agent.autonomous.AutonomousExecutor` for
        recording execution outcomes (feeds promotion/demotion logic).
    """
    global _bridge
    if _bridge is not None and _bridge.is_running:
        _bridge.stop()

    _bridge = ProactiveAgentBridge(
        engine, agent_factory, config, context, feedback_learner,
        autonomous_executor=autonomous_executor,
    )
    log.info("proactive_bridge_initialized")
    return _bridge


def get_proactive_bridge() -> ProactiveAgentBridge | None:
    """Return the current bridge instance, if any."""
    return _bridge
