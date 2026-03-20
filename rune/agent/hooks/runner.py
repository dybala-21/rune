"""Hook runner - execute registered hooks for agent lifecycle events.

Ported from src/agent/hooks/runner.ts (63 lines) - typed hook registry
with support for pre_tool_use, post_tool_use, and task_completed events.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types (ported from hooks/types.ts)

HookEvent = Literal["pre_tool_use", "post_tool_use", "task_completed"]
HookDecision = Literal["pass", "warn", "block", "retry"]


@dataclass(slots=True)
class HookResult:
    """Result from a single hook execution."""

    decision: HookDecision
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreToolUseContext:
    """Context passed to pre_tool_use hooks."""

    goal: str
    capability: str
    params: dict[str, Any] = field(default_factory=dict)
    step_number: int = 0


@dataclass(slots=True)
class PostToolUseContext:
    """Context passed to post_tool_use hooks."""

    goal: str
    capability: str
    params: dict[str, Any] = field(default_factory=dict)
    result_success: bool = True
    result_output: str = ""
    result_error: str = ""
    step_number: int = 0


@dataclass(slots=True)
class TaskCompletedContext:
    """Context passed to task_completed hooks."""

    goal: str
    success: bool
    answer: str = ""
    iterations: int = 0
    duration_ms: float = 0.0
    changed_files: list[str] = field(default_factory=list)
    requires_code_verification: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Context type map
HookContextMap = {
    "pre_tool_use": PreToolUseContext,
    "post_tool_use": PostToolUseContext,
    "task_completed": TaskCompletedContext,
}

# Handler type: sync or async callable returning HookResult
HookHandler = Callable[..., HookResult | Awaitable[HookResult]]


@dataclass(slots=True)
class HookSummary:
    """Aggregated result from running all hooks for an event."""

    blocked: bool = False
    should_retry: bool = False
    warnings: list[str] = field(default_factory=list)
    results: list[HookResult] = field(default_factory=list)


# Registered hook entry

@dataclass(slots=True)
class _RegisteredHook:
    name: str
    handler: HookHandler


# HookRunner

class HookRunner:
    """Typed hook registry and execution runner.

    Supports registering handlers for pre_tool_use, post_tool_use,
    and task_completed events.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[_RegisteredHook]] = {
            "pre_tool_use": [],
            "post_tool_use": [],
            "task_completed": [],
        }

    def register(
        self,
        event: HookEvent,
        name: str,
        handler: HookHandler,
    ) -> None:
        """Register a hook handler for an event type."""
        self._hooks[event].append(_RegisteredHook(name=name, handler=handler))

    async def run(
        self,
        event: HookEvent,
        context: Any,
    ) -> HookSummary:
        """Execute all registered hooks for an event.

        Returns an aggregated summary with blocked/retry/warnings status.
        """
        registered = self._hooks.get(event, [])
        warnings: list[str] = []
        results: list[HookResult] = []
        blocked = False
        should_retry = False

        for hook in registered:
            try:
                result = hook.handler(context)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    result = await result
                results.append(result)

                if result.decision == "block":
                    blocked = True
                elif result.decision == "retry":
                    should_retry = True
                elif result.decision == "warn" and result.message:
                    warnings.append(f"[{hook.name}] {result.message}")

            except Exception as exc:
                message = str(exc)[:300]
                log.warning(
                    "hook_execution_failed",
                    event=event,
                    hook_name=hook.name,
                    error=message,
                )
                warnings.append(f"[{hook.name}] hook failure: {message}")

        return HookSummary(
            blocked=blocked,
            should_retry=should_retry and not blocked,
            warnings=warnings,
            results=results,
        )
