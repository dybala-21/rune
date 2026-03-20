"""Core type definitions for RUNE.

Ported from src/core/types.ts - all shared types used across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

# Domain & Risk

class Domain(StrEnum):
    FILE = "file"
    BROWSER = "browser"
    PROCESS = "process"
    NETWORK = "network"
    GIT = "git"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    GENERAL = "general"
    SCHEDULE = "schedule"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


RISK_SCORE: dict[str, int] = {
    "safe": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# Intent & Planning

@dataclass(slots=True)
class Intent:
    domain: Domain
    action: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    ambiguous: bool = False
    alternatives: list[str] = field(default_factory=list)


class RollbackAction(StrEnum):
    MOVE_BACK = "move_back"
    DELETE = "delete"
    RESTORE = "restore"


@dataclass(slots=True)
class RollbackPlan:
    action: RollbackAction
    target: str
    backup_path: str | None = None
    expires_at: datetime | None = None


@dataclass(slots=True)
class Step:
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    description: str = ""
    domain: Domain = Domain.GENERAL
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    timeout: int = 60_000  # ms
    expected_output: str | None = None
    rollback: RollbackPlan | None = None


@dataclass(slots=True)
class Plan:
    steps: list[Step] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = 1.0
    requires_approval: bool = False
    description: str = ""


# Task Lifecycle

class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TaskProgress:
    current_step: int = 0
    total_steps: int = 0
    percent: float = 0.0
    current_action: str = ""


@dataclass(slots=True)
class Task:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    plan: Plan | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# Tool Execution

@dataclass(slots=True)
class ToolResult:
    success: bool
    output: str = ""
    data: Any = None
    error: str | None = None
    rollback_data: dict[str, Any] | None = None
    duration_ms: float = 0.0


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    domain: Domain = Domain.GENERAL
    risk_level: RiskLevel = RiskLevel.LOW
    actions: list[str] = field(default_factory=list)
    parameters_schema: dict[str, Any] = field(default_factory=dict)


# Policy & Approval

class PolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(slots=True)
class PolicyViolation:
    rule: str
    description: str
    severity: RiskLevel = RiskLevel.MEDIUM


@dataclass(slots=True)
class PolicyResult:
    decision: PolicyDecision
    violations: list[PolicyViolation] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class ApprovalToken:
    token_id: str = field(default_factory=lambda: uuid4().hex[:16])
    scope: str = ""
    allowed_actions: list[str] = field(default_factory=list)
    max_files: int = 100
    max_size_bytes: int = 10_485_760  # 10MB
    expires_at: datetime | None = None


# UI / App State

class AppStatus(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    ERROR = "error"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass(slots=True)
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppState:
    status: AppStatus = AppStatus.IDLE
    messages: list[Message] = field(default_factory=list)
    current_task: Task | None = None
    pending_approval: ApprovalToken | None = None


# Agent

class AgentStatus(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"


@dataclass(slots=True)
class AgentConfig:
    max_iterations: int = 200
    timeout_seconds: int = 1800  # 30 min
    model: str = "gpt-5.4"
    provider: str = "openai"
    temperature: float = 0.0
    max_tokens: int = 16_384
    _overridden: bool = False  # True when --model or --provider CLI flags are used


@dataclass(slots=True)
class CompletionTrace:
    """Tracks how the agent decided to finish."""
    reason: str = ""
    final_step: int = 0
    total_tokens_used: int = 0
    evidence_score: float = 0.0


# LLM

class ModelTier(StrEnum):
    BEST = "best"
    CODING = "coding"
    FAST = "fast"


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AZURE = "azure"
    OLLAMA = "ollama"


@dataclass(slots=True)
class LLMAvailabilityStatus:
    ready: bool = False
    available_providers: list[Provider] = field(default_factory=list)
    blocked_reason: str = "none"  # none | no_provider | unreachable
    details: dict[str, dict[str, Any]] = field(default_factory=dict)


# Session / Transcript

class TranscriptEntryType(StrEnum):
    HEADER = "header"
    MESSAGE = "message"
    ACTION = "action"
    OBSERVATION = "observation"
    COMPACTION = "compaction"


@dataclass(slots=True)
class TranscriptEntry:
    type: TranscriptEntryType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# Capability Result (used by all capability execute() functions)

@dataclass(slots=True)
class CapabilityResult:
    success: bool
    output: str = ""
    error: str | None = None
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
