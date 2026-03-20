"""Agent configuration and result types.

Ported from src/agent/types.ts. Covers AgentLoopConfig, StepResult,
LoopResult, and all supporting types for the agentic loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

# Agent loop configuration

@dataclass(slots=True)
class AgentLoopConfig:
    """Configuration for a single agent loop run.

    Mirrors the TS ``AgentConfig`` interface.
    """

    max_iterations: int = 200
    """Maximum iterations, as a safety net (model completion + timeout are primary exit conditions)."""

    max_retries: int = 3
    """Retry count on transient failures."""

    timeout_ms: int = 1_800_000  # 30 minutes
    """Overall timeout in milliseconds."""

    enable_reflexion: bool = True
    """Whether the agent should self-critique between steps."""

    stream_output: bool = True
    """Enable streaming output to the caller."""

    abort_event: Any | None = None
    """Optional asyncio.Event (or similar) to signal early abort."""


DEFAULT_AGENT_CONFIG = AgentLoopConfig()


# Agent status / step enums

class AgentStatus(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


AgentStepType = Literal["thought", "action", "observation", "reflection"]
MessageRole = Literal["system", "user", "assistant", "tool"]


# Step / Message types

@dataclass(slots=True)
class AgentStep:
    """A single step in the agent's execution trace."""

    type: AgentStepType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """A tool call request from the model."""

    id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallResult:
    """Result from executing a tool call."""

    tool_call_id: str
    success: bool
    output: str
    error: str | None = None


@dataclass(slots=True)
class AgentMessage:
    """Single message in the agent conversation."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


# Agent action / think result

@dataclass(slots=True)
class AgentAction:
    """An action the agent wants to execute."""

    capability: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionResult:
    """Result from executing an agent action."""

    success: bool
    output: str
    error: str | None = None
    suggestions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ThinkResult:
    """Result of the agent's reasoning step."""

    reasoning: str
    decision: Literal["continue", "complete", "error"]
    next_action: AgentAction | None = None
    final_answer: str | None = None
    error_message: str | None = None


# Agent state

@dataclass(slots=True)
class AgentState:
    """Full mutable state of a running agent loop."""

    goal: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    history: list[AgentStep] = field(default_factory=list)
    scratchpad: list[str] = field(default_factory=list)
    iteration: int = 0
    status: AgentStatus = AgentStatus.IDLE
    started_at: datetime = field(default_factory=datetime.now)


# Completion trace

@dataclass(slots=True)
class CompletionRequirementTrace:
    id: str
    description: str
    required: bool = True
    status: Literal["done", "blocked", "skipped"] = "done"
    evidence: list[str] = field(default_factory=list)
    failure_reason: str | None = None


@dataclass(slots=True)
class CompletionContractTrace:
    kind: str = ""
    tool_requirement: Literal["none", "read", "write"] = "none"
    grounding_requirement: Literal["none", "recommended", "required"] = "none"
    source: str = ""
    resolved: bool = True
    unresolved_reason: str | None = None


@dataclass(slots=True)
class CompletionContractPlanTrace:
    objective: str = ""
    action_plan: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    verification_candidates: list[str] = field(default_factory=list)
    probe_candidates: list[str] | None = None


@dataclass(slots=True)
class CompletionEvidenceTrace:
    reads: int = 0
    writes: int = 0
    executions: int = 0
    verifications: int = 0
    browser_reads: int = 0
    browser_writes: int = 0
    changed_files: int = 0
    structured_writes: int | None = None
    structured_write_samples: list[str] | None = None
    service_task: dict[str, int] | None = None
    service_task_samples: dict[str, list[str]] | None = None


@dataclass(slots=True)
class CompletionTrace:
    """Full trace of how the completion gate evaluated the run."""

    outcome: Literal["verified", "partial", "blocked"] | None = None
    success: bool = True
    error: str | None = None
    workspace_root: str | None = None
    requested_workspace_path: str | None = None
    primary_execution_root: str | None = None
    execution_roots: list[str] | None = None
    workspace_warning: str | None = None
    evidence: CompletionEvidenceTrace | None = None
    hard_failures: list[str] = field(default_factory=list)
    missing_requirement_ids: list[str] = field(default_factory=list)
    requirements: list[CompletionRequirementTrace] = field(default_factory=list)
    contract: CompletionContractTrace | None = None
    contract_plan: CompletionContractPlanTrace | None = None


# Classification hint (attached to LoopResult for downstream memory)

@dataclass(slots=True)
class ClassificationHint:
    category: str = ""
    domain: str = ""
    requires_code: bool = False
    requires_execution: bool = False
    is_continuation: bool = False


# Loop result - top-level return from agent.run()

@dataclass(slots=True)
class StepResult:
    """Result of a single iteration (step) within the loop."""

    step_type: AgentStepType
    content: str
    action: AgentAction | None = None
    action_result: ActionResult | None = None
    duration_ms: float = 0.0


@dataclass(slots=True)
class LoopResult:
    """Final result of an agent run."""

    success: bool
    answer: str
    history: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    aborted: bool = False
    tokens_consumed: int | None = None
    completion_trace: CompletionTrace | None = None
    classification_hint: ClassificationHint | None = None


# Execution context

@dataclass(slots=True)
class ExecutionContext:
    """Runtime context passed to the agent loop."""

    cwd: str = "."
    preferences: dict[str, Any] = field(default_factory=dict)
    recent_commands: list[str] = field(default_factory=list)
    memory_context: str = ""


# Failover

FailoverReason = Literal[
    "auth",
    "rate_limit",
    "billing",
    "timeout",
    "context_overflow",
    "format",
    "unknown",
]


@dataclass(slots=True)
class FailoverResult:
    success: bool
    reason: FailoverReason | None = None
    retries_used: int = 0
    final_profile: str | None = None


# Session / Transcript

SessionScope = Literal["main", "task", "conversation"]


@dataclass(slots=True)
class SessionConfig:
    agent_id: str = ""
    scope: SessionScope = "main"
    identifier: str | None = None
    workspace_path: str | None = None
    session_dir: str | None = None


@dataclass(slots=True)
class SessionMetadata:
    id: str = ""
    key: str = ""
    agent_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    cwd: str = "."
    message_count: int = 0
    compaction_count: int = 0
