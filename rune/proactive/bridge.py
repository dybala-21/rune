"""Deliver proactive suggestions and manage their execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from rune.proactive.engine import ProactiveEngine
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.feedback import FeedbackLearner
from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Types

class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    # Shown to the user; execution has not started.
    DELIVERED = "delivered"
    # Execution was claimed, but its outcome could not be confirmed.
    UNVERIFIED = "unverified"


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
    # When disabled, suggestions need an explicit accept before execution.
    auto_execute: bool = False
    # Enable only if replaying the whole goal cannot duplicate external actions.
    retry_safe: bool = False


# Factories accept a goal and optionally verification commands. Results include
# success and, when checked, verified or tests_passed.
AgentFactory = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


# ProactiveAgentBridge

class ProactiveAgentBridge:
    """Dispatch suggestions with shared execution claims and an hourly limit."""

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
        "_execution_store",
        "_inflight",
        "_delivered",
        "_subscribed",
    )

    def __init__(
        self,
        engine: ProactiveEngine,
        agent_factory: AgentFactory,
        config: BridgeConfig | None = None,
        context: dict[str, Any] | None = None,
        feedback_learner: FeedbackLearner | None = None,
        autonomous_executor: Any | None = None,
        execution_store: ExecutionStore | None = None,
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

        self._execution_store = execution_store or ExecutionStore()
        self._inflight: dict[str, asyncio.Task[ExecutionRecord]] = {}
        self._delivered: dict[str, ExecutionRecord] = {}
        self._subscribed = False
        self._subscribe()

    def _subscribe(self) -> None:
        if not self._subscribed:
            self._engine.on("suggestion", self._on_suggestion_event)
            self._subscribed = True

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

        self._subscribe()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info(
            "bridge_started",
            poll_interval=self._config.poll_interval_seconds,
            max_per_hour=self._config.max_executions_per_hour,
        )

    def stop(self) -> None:
        """Stop polling and cancel the background task."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        self._poll_task = None
        for task in self._inflight.values():
            task.cancel()
        if self._subscribed:
            self._engine.off("suggestion", self._on_suggestion_event)
            self._subscribed = False
        log.info("bridge_stopped", history_size=len(self._history))

    def _on_suggestion_event(self, suggestions: list[Suggestion]) -> None:
        if self._running:
            for suggestion in suggestions:
                if suggestion.confidence >= self._config.min_confidence:
                    self._dispatch(suggestion)

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

    async def _call_factory(self, goal: str, verification: list[str]) -> object:
        """Pass verification commands when the factory supports the argument."""
        import inspect

        # Check the signature first to avoid retrying after an internal TypeError.
        try:
            signature = inspect.signature(self._agent_factory)
        except (TypeError, ValueError):
            signature = None
        accepts_verification = True
        if signature is not None:
            try:
                signature.bind(goal, verification=verification)
            except TypeError:
                accepts_verification = False
        coro = self._agent_factory(goal, verification=verification) if accepts_verification else (
            self._agent_factory(goal)
        )
        return await coro

    def _outcome_from_result(self, result: object) -> ExecutionStatus:
        """Require a success flag and a passing check to record SUCCESS.

        A truthy success flag without a passing check is UNVERIFIED.
        Missing or false success flags, and non-dict results, are FAILURE.
        """
        if not isinstance(result, dict) or not bool(result.get("success", False)):
            return ExecutionStatus.FAILURE
        verified = result.get("verified")
        if verified is None:
            verified = result.get("tests_passed")
        if verified is True:
            return ExecutionStatus.SUCCESS
        return ExecutionStatus.UNVERIFIED

    async def execute_suggestion(
        self, suggestion: Suggestion, *, force: bool = False
    ) -> ExecutionRecord:
        """Deliver a suggestion, or execute it when auto_execute or force is set.

        Concurrent callers share the execution. Failed attempts are retried only
        when retry_safe is enabled, up to max_retries with exponential backoff.
        """
        return await asyncio.shield(self._dispatch(suggestion, force=force))

    @staticmethod
    def _ready(record: ExecutionRecord) -> asyncio.Future[ExecutionRecord]:
        future: asyncio.Future[ExecutionRecord] = asyncio.get_running_loop().create_future()
        future.set_result(record)
        return future

    def _dispatch(
        self, suggestion: Suggestion, *, force: bool = False
    ) -> asyncio.Future[ExecutionRecord]:
        """Claim execution before yielding so concurrent callers can join it.

        Delivered suggestions remain eligible for a later explicit accept.
        """
        if not (self._config.auto_execute or force):
            record = self._delivered.get(suggestion.id)
            if record is None:
                record = self._record(suggestion, ExecutionStatus.DELIVERED, attempt=0)
                self._delivered[suggestion.id] = record
            return self._ready(record)

        fingerprint = hashlib.sha256(json.dumps(
            [suggestion.title, suggestion.description, suggestion.verification],
            ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest()
        decision = self._execution_store.claim(
            suggestion.id, fingerprint, self._config.max_executions_per_hour
        )
        if decision == "rate_limited":
            return self._ready(self._record(
                suggestion, ExecutionStatus.SKIPPED, error="Hourly execution limit", attempt=0,
            ))
        if decision == "exists":
            existing = self._execution_store.get(suggestion.id)
            assert existing is not None
            previous_fingerprint, saved = existing
            if previous_fingerprint != fingerprint:
                return self._ready(self._record(
                    suggestion, ExecutionStatus.SKIPPED,
                    error="Operation ID already belongs to different work", attempt=0,
                ))
            if suggestion.id in self._inflight:
                return self._inflight[suggestion.id]
            if saved is not None:
                return self._ready(ExecutionRecord(
                    suggestion_id=suggestion.id, suggestion_title=suggestion.title,
                    status=ExecutionStatus(saved["status"]), error=saved.get("error"),
                    attempt=saved["attempt"], duration_ms=saved["duration_ms"],
                    timestamp=datetime.fromisoformat(saved["timestamp"]),
                ))
            return self._ready(ExecutionRecord(
                suggestion_id=suggestion.id, suggestion_title=suggestion.title,
                status=ExecutionStatus.UNVERIFIED, attempt=0,
                error="An execution was already claimed. Inspect its outcome before retrying.",
            ))

        snapshot = replace(suggestion, verification=list(suggestion.verification))
        task = asyncio.create_task(self._run_claimed(snapshot))
        self._inflight[suggestion.id] = task

        def finished(done: asyncio.Task[ExecutionRecord]) -> None:
            self._inflight.pop(suggestion.id, None)
            if not done.cancelled() and (error := done.exception()) is not None:
                log.error("proactive_execution_error", suggestion_id=suggestion.id, error=str(error))

        task.add_done_callback(finished)
        return task

    async def _run_claimed(self, suggestion: Suggestion) -> ExecutionRecord:
        try:
            record = await self._execute_attempts(suggestion)
        except asyncio.CancelledError:
            record = self._record(
                suggestion, ExecutionStatus.UNVERIFIED,
                error="Execution interrupted; inspect external state before retrying.",
            )
            self._save_record(record)
            raise
        self._save_record(record)
        return record

    def _save_record(self, record: ExecutionRecord) -> None:
        self._execution_store.finish(record.suggestion_id, {
            "status": record.status.value, "error": record.error,
            "attempt": record.attempt, "duration_ms": record.duration_ms,
            "timestamp": record.timestamp.isoformat(),
        })

    async def _execute_attempts(self, suggestion: Suggestion) -> ExecutionRecord:
        max_attempts = 1 + (self._config.max_retries if self._config.retry_safe else 0)
        last_record: ExecutionRecord | None = None

        for attempt in range(1, max_attempts + 1):
            start = datetime.now(UTC)
            try:
                goal = f"{suggestion.title}: {suggestion.description}"
                result = await self._call_factory(goal, suggestion.verification)

                duration = (datetime.now(UTC) - start).total_seconds() * 1000
                outcome = self._outcome_from_result(result)

                if outcome == ExecutionStatus.SUCCESS:
                    record = self._record(
                        suggestion,
                        ExecutionStatus.SUCCESS,
                        attempt=attempt,
                        duration_ms=duration,
                    )
                    self._engine.record_feedback(suggestion.id, True)
                    if self._feedback_learner is not None:
                        self._feedback_learner.record_feedback(suggestion, "accepted")
                    try:
                        from rune.proactive.reflexion import get_reflexion_learner
                        get_reflexion_learner().record_task_outcome({
                            "domain": suggestion.type,
                            "success": True,
                            "goal": suggestion.title,
                            "steps_taken": attempt,
                            "duration_ms": duration,
                        })
                    except Exception as exc:
                        log.debug("reflexion_record_success_failed", error=str(exc)[:100])
                    self._record_to_autonomous_executor(
                        suggestion, success=True, duration_ms=duration,
                        result_summary=str(result.get("output", ""))[:200] if isinstance(result, dict) else "",
                    )
                    return record
                elif outcome == ExecutionStatus.UNVERIFIED:
                    # An unknown outcome must not trigger retries or success learning.
                    log.info("proactive_unverified", suggestion=suggestion.title)
                    return self._record(
                        suggestion,
                        ExecutionStatus.UNVERIFIED,
                        attempt=attempt,
                        duration_ms=duration,
                    )
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

            if attempt < max_attempts:
                backoff = self._config.backoff_base_seconds * (2 ** (attempt - 1))
                log.debug(
                    "bridge_retry_backoff",
                    suggestion=suggestion.title,
                    attempt=attempt,
                    backoff_seconds=backoff,
                )
                await asyncio.sleep(backoff)

        self._engine.record_feedback(suggestion.id, False)
        if self._feedback_learner is not None:
            self._feedback_learner.record_feedback(suggestion, "dismissed")

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
        except Exception as exc:
            log.debug("reflexion_record_failure_failed", error=str(exc)[:100])
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
        """Record the outcome for autonomy promotion and demotion decisions."""
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
        since = datetime.now(UTC).timestamp() - 3600
        return self._execution_store.started_since(since) >= self._config.max_executions_per_hour

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
    execution_store: ExecutionStore | None = None,
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
    if _bridge is not None:
        _bridge.stop()

    _bridge = ProactiveAgentBridge(
        engine, agent_factory, config, context, feedback_learner,
        autonomous_executor=autonomous_executor,
        execution_store=execution_store,
    )
    log.info("proactive_bridge_initialized")
    return _bridge


def get_proactive_bridge() -> ProactiveAgentBridge | None:
    """Return the current bridge instance, if any."""
    return _bridge
