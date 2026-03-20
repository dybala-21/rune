"""Hook event types, decision types, and context dataclasses.

Extracted from the runner module for cleaner imports.  These types are
also used by the TS codebase at src/agent/hooks/types.ts.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

# Core enums (as Literal unions for lightweight typing)

HookEvent = Literal["pre_tool_use", "post_tool_use", "task_completed"]
"""Agent lifecycle hook events."""

HookDecision = Literal["pass", "warn", "block", "retry"]
"""Decisions a hook can return to influence the agent loop."""


# Hook result

@dataclass(slots=True)
class HookResult:
    """Result from a single hook execution."""

    decision: HookDecision
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Context objects - one per event type

@dataclass(slots=True)
class PreToolUseContext:
    """Context passed to ``pre_tool_use`` hooks.

    Allows the hook to inspect (and potentially block) a tool call
    *before* it executes.
    """

    goal: str
    capability: str
    params: dict[str, Any] = field(default_factory=dict)
    step_number: int = 0


@dataclass(slots=True)
class PostToolUseContext:
    """Context passed to ``post_tool_use`` hooks.

    Provides the result of a tool call so hooks can decide whether
    to retry, warn, or take other corrective action.
    """

    goal: str
    capability: str
    params: dict[str, Any] = field(default_factory=dict)
    result_success: bool = True
    result_output: str = ""
    result_error: str = ""
    step_number: int = 0


@dataclass(slots=True)
class TaskCompletedContext:
    """Context passed to ``task_completed`` hooks.

    Fired once after the agent loop finishes (successfully or not).
    """

    goal: str
    success: bool
    answer: str = ""
    iterations: int = 0
    duration_ms: float = 0.0
    changed_files: list[str] = field(default_factory=list)
    requires_code_verification: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Context type map (event -> context class)

HOOK_CONTEXT_MAP: dict[HookEvent, type] = {
    "pre_tool_use": PreToolUseContext,
    "post_tool_use": PostToolUseContext,
    "task_completed": TaskCompletedContext,
}

# Handler callable type

HookHandler = Callable[..., HookResult | Awaitable[HookResult]]
"""A hook handler is a sync or async callable that receives a context
object and returns a :class:`HookResult`."""
