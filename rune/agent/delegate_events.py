"""Delegate events - event types for multi-agent delegation.

Ported from src/agent/delegate-events.ts (43 lines) - defines event
types and runtime hooks for delegate.task / delegate.orchestrate
capabilities used in multi-agent coordination.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

# Constants

DELEGATE_RUNTIME_HOOKS_KEY = "__delegateRuntimeHooks"
DELEGATE_SANDBOX_KEY = "__delegateSandboxInstance"

# Types

DelegateCapabilityName = Literal["delegate.task", "delegate.orchestrate"]

DelegateProgressStage = Literal[
    "start",
    "planning",
    "planned",
    "plan_gate_review",
    "plan_gate_waiting_approval",
    "plan_gate_approved",
    "plan_gate_denied",
    "executing",
    "task_claimed",
    "task_complete",
    "replanning",
    "integrating",
    "completed",
    "failed",
]


@dataclass(slots=True)
class DelegateProgressEvent:
    """Progress event emitted during delegation."""

    capability: DelegateCapabilityName
    stage: DelegateProgressStage
    message: str = ""
    task_id: str = ""
    role: str = ""
    success: bool | None = None
    total_tasks: int | None = None
    completed_tasks: int | None = None


@dataclass(slots=True)
class DelegateRuntimeHooks:
    """Runtime hooks injected by tool-adapter for delegate capabilities.

    These hooks are never exposed to the model-facing parameter schema.
    """

    on_progress_event: Callable[[DelegateProgressEvent], None] | None = None
    on_plan_approval_required: (
        Callable[[dict[str, Any]], Awaitable[bool]] | None
    ) = None
