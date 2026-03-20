"""Gateway API Protocol.

Ported from src/api/protocol.ts - RPC method definitions,
request/response envelope types, error types.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "v1"


# Envelope

class ApiRequest(BaseModel):
    """RPC request envelope."""
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ApiError(BaseModel):
    """Structured error payload."""
    code: str
    message: str
    details: Any | None = None


class ApiResponse(BaseModel):
    """RPC response envelope."""
    success: bool
    data: Any | None = None
    error: ApiError | None = None
    timestamp: str
    id: str | None = None


# Run status

RunStatus = Literal["queued", "running", "completed", "failed", "aborted"]


# Agent domain

class AgentRequestParams(BaseModel):
    goal: str
    session_id: str | None = Field(None, alias="sessionId")
    cwd: str | None = None
    sender_name: str | None = Field(None, alias="senderName")

    model_config = ConfigDict(populate_by_name=True)


class AgentRequestResult(BaseModel):
    run_id: str = Field(alias="runId")
    session_id: str = Field(alias="sessionId")

    model_config = ConfigDict(populate_by_name=True)


class AgentAbortParams(BaseModel):
    run_id: str = Field(alias="runId")

    model_config = ConfigDict(populate_by_name=True)


class AgentStatusParams(BaseModel):
    run_id: str = Field(alias="runId")

    model_config = ConfigDict(populate_by_name=True)


class TokenUsage(BaseModel):
    input: int
    output: int
    cache_read: int | None = Field(None, alias="cacheRead")

    model_config = ConfigDict(populate_by_name=True)


class AgentStatusResult(BaseModel):
    run_id: str = Field(alias="runId")
    status: RunStatus
    answer: str | None = None
    error: str | None = None
    elapsed_ms: int | None = Field(None, alias="elapsedMs")
    usage: TokenUsage | None = None

    model_config = ConfigDict(populate_by_name=True)


class AgentApprovalParams(BaseModel):
    approval_id: str = Field(alias="approvalId")
    decision: Literal["approve_once", "approve_always", "deny"]
    user_guidance: str | None = Field(None, alias="userGuidance")

    model_config = ConfigDict(populate_by_name=True)


class AgentQuestionParams(BaseModel):
    question_id: str = Field(alias="questionId")
    answer: str
    selected_index: int | None = Field(None, alias="selectedIndex")

    model_config = ConfigDict(populate_by_name=True)


# Sessions domain

class SessionInfo(BaseModel):
    id: str
    user_id: str = Field(alias="userId")
    title: str
    status: Literal["active", "archived"]
    channel: str
    turn_count: int = Field(alias="turnCount")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class SessionListParams(BaseModel):
    status: Literal["active", "archived"] | None = None
    limit: int | None = 20
    offset: int | None = 0


class SessionListResult(BaseModel):
    sessions: list[SessionInfo]
    total: int


class SessionGetParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    include_turns: bool | None = Field(None, alias="includeTurns")
    max_turns: int | None = Field(None, alias="maxTurns")

    model_config = ConfigDict(populate_by_name=True)


class TurnInfo(BaseModel):
    role: str
    content: str
    channel: str
    timestamp: str


class SessionGetResult(SessionInfo):
    turns: list[TurnInfo] | None = None


class SessionDeleteParams(BaseModel):
    session_id: str = Field(alias="sessionId")

    model_config = ConfigDict(populate_by_name=True)


class SessionArchiveParams(BaseModel):
    session_id: str = Field(alias="sessionId")

    model_config = ConfigDict(populate_by_name=True)


class SessionEventsParams(BaseModel):
    session_id: str = Field(alias="sessionId")
    run_id: str | None = Field(None, alias="runId")
    include_tools: bool | None = Field(True, alias="includeTools")
    include_thinking: bool | None = Field(True, alias="includeThinking")

    model_config = ConfigDict(populate_by_name=True)


class EventLogEntry(BaseModel):
    event: str
    data: Any
    timestamp: str


class SessionEventsResult(BaseModel):
    events: list[EventLogEntry]
    runs: list[str]


# Channels domain

class ChannelInfo(BaseModel):
    name: str
    status: Literal["disconnected", "connecting", "connected", "error"]
    type: Literal["in-process", "api-client"]
    session_count: int = Field(0, alias="sessionCount")

    model_config = ConfigDict(populate_by_name=True)


class ChannelListResult(BaseModel):
    channels: list[ChannelInfo]


class ChannelRestartParams(BaseModel):
    name: str


class ChannelSendParams(BaseModel):
    channel_id: str = Field(alias="channelId")
    message: str

    model_config = ConfigDict(populate_by_name=True)


# Config domain

class ActiveModelInfo(BaseModel):
    provider: str
    model: str
    source: str


class ConfigGetResult(BaseModel):
    proactive_enabled: bool = Field(alias="proactiveEnabled")
    gateway_channels: list[str] = Field(alias="gatewayChannels")
    max_concurrency: int = Field(alias="maxConcurrency")
    version: str
    active_model: ActiveModelInfo | None = Field(None, alias="activeModel")
    memory_tuning: dict[str, Any] | None = Field(None, alias="memoryTuning")
    safety_tuning: dict[str, Any] | None = Field(None, alias="safetyTuning")

    model_config = ConfigDict(populate_by_name=True)


class ConfigPatchParams(BaseModel):
    proactive_enabled: bool | None = Field(None, alias="proactiveEnabled")
    memory_tuning: dict[str, Any] | None = Field(None, alias="memoryTuning")
    safety_tuning: dict[str, Any] | None = Field(None, alias="safetyTuning")

    model_config = ConfigDict(populate_by_name=True)


# Cron domain

class CronJobInfo(BaseModel):
    id: str
    name: str
    schedule: str
    command: str
    enabled: bool
    created_at: str = Field(alias="createdAt")
    last_run_at: str | None = Field(None, alias="lastRunAt")
    run_count: int = Field(0, alias="runCount")
    max_runs: int | None = Field(None, alias="maxRuns")
    type: str | None = None
    conditions: dict[str, Any] | None = None
    depends_on: list[str] | None = Field(None, alias="dependsOn")
    actor: str | None = None
    target: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class CronListParams(BaseModel):
    pass


class CronListResult(BaseModel):
    jobs: list[CronJobInfo]
    builtin_tasks: list[dict[str, Any]] = Field(default_factory=list, alias="builtinTasks")

    model_config = ConfigDict(populate_by_name=True)


class CronCreateParams(BaseModel):
    name: str
    schedule: str
    command: str
    enabled: bool = True
    max_runs: int | None = Field(None, alias="maxRuns")
    type: str | None = None
    conditions: dict[str, Any] | None = None
    depends_on: list[str] | None = Field(None, alias="dependsOn")
    actor: str | None = None
    target: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class CronCreateResult(BaseModel):
    job: CronJobInfo


class CronUpdateParams(BaseModel):
    id: str
    name: str | None = None
    schedule: str | None = None
    command: str | None = None
    enabled: bool | None = None

    model_config = ConfigDict(populate_by_name=True)


class CronUpdateResult(BaseModel):
    job: CronJobInfo


class CronDeleteParams(BaseModel):
    id: str


class CronDeleteResult(BaseModel):
    deleted: bool


# Env domain

EnvCategory = Literal[
    "llm", "logging", "search", "telegram", "discord",
    "slack", "mattermost", "line", "whatsapp", "google-chat", "other",
]


class EnvVarInfo(BaseModel):
    key: str
    masked_value: str = Field(alias="maskedValue")
    scope: Literal["user", "project"]
    is_secret: bool = Field(alias="isSecret")
    category: EnvCategory

    model_config = ConfigDict(populate_by_name=True)


class EnvListParams(BaseModel):
    scope: Literal["user", "project"] | None = None


class EnvListResult(BaseModel):
    variables: list[EnvVarInfo]
    paths: dict[str, str]


class EnvSetParams(BaseModel):
    key: str
    value: str
    scope: Literal["user", "project"]


class EnvUnsetParams(BaseModel):
    key: str
    scope: Literal["user", "project"]


# Health domain

class SchedulerStats(BaseModel):
    queued: int
    running: int
    max_concurrency: int = Field(alias="maxConcurrency")

    model_config = ConfigDict(populate_by_name=True)


class SubsystemStatus(BaseModel):
    memory: Literal["ok", "error"]
    proactive: Literal["ok", "disabled"]
    gateway: Literal["ok", "error"]
    mcp: Literal["ok", "disabled"]
    scheduler: SchedulerStats


class HealthResult(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    uptime: float
    subsystems: SubsystemStatus


# Proactive domain

class ProactiveDashboardParams(BaseModel):
    limit: int | None = 20


class SuggestionAction(BaseModel):
    command: str
    auto_executable: bool = Field(alias="autoExecutable")

    model_config = ConfigDict(populate_by_name=True)


class PendingSuggestion(BaseModel):
    id: str
    type: str
    priority: str
    title: str
    description: str
    confidence: float
    created_at: str = Field(alias="createdAt")
    action: SuggestionAction | None = None

    model_config = ConfigDict(populate_by_name=True)


class ProactiveDashboardResult(BaseModel):
    stats: dict[str, Any]
    patterns: list[dict[str, Any]]
    recent_executions: list[dict[str, Any]] = Field(alias="recentExecutions")
    engine: dict[str, Any]
    pending_suggestions: list[PendingSuggestion] = Field(alias="pendingSuggestions")
    governance: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)


class ProactiveRespondParams(BaseModel):
    suggestion_id: str = Field(alias="suggestionId")
    response: str

    model_config = ConfigDict(populate_by_name=True)


# Runs domain

class RunsListParams(BaseModel):
    client_id: str | None = Field(None, alias="clientId")
    conversation_id: str | None = Field(None, alias="conversationId")
    status: RunStatus | None = None
    limit: int | None = 20
    offset: int | None = 0

    model_config = ConfigDict(populate_by_name=True)


class RunRecord(BaseModel):
    run_id: str = Field(alias="runId")
    session_id: str = Field(alias="sessionId")
    client_id: str = Field(alias="clientId")
    conversation_id: str | None = Field(None, alias="conversationId")
    status: RunStatus
    goal: str
    started_at: str = Field(alias="startedAt")
    completed_at: str | None = Field(None, alias="completedAt")
    result_success: bool | None = Field(None, alias="resultSuccess")
    result_answer: str | None = Field(None, alias="resultAnswer")
    error: str | None = None
    usage_input: int | None = Field(None, alias="usageInput")
    usage_output: int | None = Field(None, alias="usageOutput")

    model_config = ConfigDict(populate_by_name=True)


class RunsListResult(BaseModel):
    runs: list[RunRecord]
    total: int


class RunsGetParams(BaseModel):
    run_id: str = Field(alias="runId")

    model_config = ConfigDict(populate_by_name=True)


# Skills domain

class SkillInfo(BaseModel):
    name: str
    description: str
    scope: Literal["user", "project", "builtin"]
    file_path: str | None = Field(None, alias="filePath")

    model_config = ConfigDict(populate_by_name=True)


class SkillListParams(BaseModel):
    scope: Literal["user", "project", "builtin"] | None = None


class SkillListResult(BaseModel):
    skills: list[SkillInfo]
    project_path: str = Field(alias="projectPath")
    user_path: str = Field(alias="userPath")

    model_config = ConfigDict(populate_by_name=True)


class SkillGetParams(BaseModel):
    name: str


class SkillDetailResult(BaseModel):
    name: str
    description: str
    body: str
    scope: Literal["user", "project", "builtin"]
    file_path: str | None = Field(None, alias="filePath")

    model_config = ConfigDict(populate_by_name=True)


class SkillCreateParams(BaseModel):
    name: str
    description: str
    body: str
    scope: Literal["user", "project"] = "project"


class SkillCreateResult(BaseModel):
    name: str
    file_path: str = Field(alias="filePath")

    model_config = ConfigDict(populate_by_name=True)


class SkillUpdateParams(BaseModel):
    name: str
    description: str | None = None
    body: str | None = None


class SkillDeleteParams(BaseModel):
    name: str


# Tokens domain

Permission = Literal[
    "agent:execute", "agent:abort",
    "sessions:read", "sessions:write",
    "channels:read", "channels:admin",
    "config:read", "config:write",
    "env:read", "env:write",
    "skills:read", "skills:write",
    "proactive:read", "proactive:write",
    "admin",
]


class TokenCreateParams(BaseModel):
    label: str
    permissions: list[Permission]
    expires_at: str | None = Field(None, alias="expiresAt")

    model_config = ConfigDict(populate_by_name=True)


class TokenCreateResult(BaseModel):
    token: str
    id: str
    label: str
    permissions: list[Permission]


class TokenListEntry(BaseModel):
    id: str
    label: str
    permissions: list[Permission]
    created_at: str = Field(alias="createdAt")
    expires_at: str | None = Field(None, alias="expiresAt")

    model_config = ConfigDict(populate_by_name=True)


class TokenListResult(BaseModel):
    tokens: list[TokenListEntry]


class TokenRevokeParams(BaseModel):
    id: str
