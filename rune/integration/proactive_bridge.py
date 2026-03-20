"""Bridge connecting ProactiveEngine and NativeAgentLoop for RUNE.

Subscribes to engine suggestion events and decides whether to
auto-execute (via AutonomousExecutor) or notify the user.
"""

from __future__ import annotations

import time
from typing import Any

from rune.proactive.engine import ProactiveEngine
from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Type alias for the execution feedback values accepted by AutonomousExecutor
ExecutionFeedback = str  # "approved" | "full_revert" | "none"

# Confidence threshold for auto-execution
_AUTO_EXECUTE_THRESHOLD = 0.8


class ProactiveAgentBridge:
    """Bridges the ProactiveEngine with the agent loop and autonomous executor.

    Listens for suggestion events from the engine and routes them:
    - High-confidence suggestions from safe types -> auto-execute
    - Everything else -> notify the user via the agent loop
    """

    __slots__ = ("_engine", "_agent_loop", "_autonomous_executor", "_running")

    def __init__(
        self,
        engine: ProactiveEngine,
        agent_loop: Any,
        autonomous_executor: Any = None,
    ) -> None:
        self._engine = engine
        self._agent_loop = agent_loop
        self._autonomous_executor = autonomous_executor
        self._running = False

    async def start(self) -> None:
        """Start the bridge and subscribe to engine events."""
        if self._running:
            return
        self._running = True
        log.info("proactive_bridge_started")

    async def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        log.info("proactive_bridge_stopped")

    async def on_suggestions(self, suggestions: list[Suggestion]) -> None:
        """Handle a batch of suggestions from the proactive engine.

        Decides per-suggestion whether to auto-execute or notify.

        Parameters:
            suggestions: The list of suggestions to process.
        """
        if not self._running:
            return

        for suggestion in suggestions:
            await self._on_suggestion(suggestion)

    async def _on_suggestion(self, suggestion: Suggestion) -> None:
        """Process a single suggestion.

        Auto-executes if the suggestion is high-confidence and the
        autonomous executor permits it. Otherwise notifies the user.
        """
        if not self._running:
            return

        can_auto_execute = (
            suggestion.confidence >= _AUTO_EXECUTE_THRESHOLD
            and suggestion.type in ("optimization", "followup")
            and self._autonomous_executor is not None
        )

        if can_auto_execute:
            try:
                decision = self._autonomous_executor.decide(
                    command=suggestion.description,
                    domain="notify",
                    risk_score=1.0 - suggestion.confidence,
                )

                # Only auto-execute at JUST_DO or INFORM_DO level
                from rune.agent.autonomous import AutonomyLevel

                if decision.level >= AutonomyLevel.INFORM_DO:
                    log.info(
                        "proactive_auto_execute",
                        suggestion_id=suggestion.id,
                        type=suggestion.type,
                        level=decision.level.name,
                    )
                    # Execute via agent and record outcome to the ledger
                    success = await self._execute_and_record(
                        suggestion, decision
                    )
                    await self._notify_user(
                        suggestion, auto_executed=True, success=success
                    )
                    return
            except Exception as exc:
                log.warning(
                    "proactive_auto_execute_failed",
                    suggestion_id=suggestion.id,
                    error=str(exc),
                )
                # Record failure to the ledger so demotion logic can fire
                self._record_to_ledger(
                    suggestion, decision=None, success=False
                )

        # Default: notify the user
        await self._notify_user(suggestion, auto_executed=False)

    # Execution + ledger recording

    async def _execute_and_record(
        self,
        suggestion: Suggestion,
        decision: Any,
    ) -> bool:
        """Execute a suggestion via the agent loop and record the outcome.

        Returns ``True`` on success, ``False`` on failure.
        """
        start = time.monotonic()
        success = False

        try:
            if hasattr(self._agent_loop, "run_goal"):
                goal = f"{suggestion.title}: {suggestion.description}"
                result = await self._agent_loop.run_goal(goal)
                success = bool(
                    result.get("success", True) if isinstance(result, dict) else True
                )
            else:
                # Agent loop does not support run_goal -- treat as success
                # (notification-only path)
                success = True
        except Exception as exc:
            log.warning(
                "proactive_execute_error",
                suggestion_id=suggestion.id,
                error=str(exc),
            )
            success = False

        duration_ms = (time.monotonic() - start) * 1000
        self._record_to_ledger(suggestion, decision, success, duration_ms)
        return success

    def _record_to_ledger(
        self,
        suggestion: Suggestion,
        decision: Any,
        success: bool,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an execution outcome to the AutonomousExecutor ledger.

        This feeds the promotion/demotion logic so the executor learns
        from proactive execution outcomes.
        """
        if self._autonomous_executor is None:
            return

        pattern_key = f"proactive:{suggestion.type}:{suggestion.title[:40]}"
        feedback: ExecutionFeedback = "approved" if success else "full_revert"

        try:
            self._autonomous_executor.record_feedback(pattern_key, feedback)
            log.debug(
                "proactive_ledger_recorded",
                suggestion_id=suggestion.id,
                pattern_key=pattern_key,
                feedback=feedback,
                duration_ms=round(duration_ms, 1),
            )
        except Exception as exc:
            log.warning(
                "proactive_ledger_record_failed",
                suggestion_id=suggestion.id,
                error=str(exc),
            )

    # User notification

    async def _notify_user(
        self,
        suggestion: Suggestion,
        *,
        auto_executed: bool,
        success: bool | None = None,
    ) -> None:
        """Send a suggestion notification to the user via the agent loop.

        Parameters:
            suggestion: The suggestion to notify about.
            auto_executed: Whether the action was already auto-executed.
            success: Outcome of auto-execution (None if not executed).
        """
        import asyncio

        if hasattr(self._agent_loop, "emit"):
            action = "auto_executed" if auto_executed else "suggested"
            payload: dict[str, Any] = {
                "suggestion_id": suggestion.id,
                "type": suggestion.type,
                "title": suggestion.title,
                "description": suggestion.description,
                "confidence": suggestion.confidence,
                "action": action,
            }
            if success is not None:
                payload["success"] = success
            result = self._agent_loop.emit(
                "proactive_suggestion",
                payload,
            )
            if asyncio.iscoroutine(result):
                await result

        log.debug(
            "proactive_notified",
            suggestion_id=suggestion.id,
            auto_executed=auto_executed,
        )
