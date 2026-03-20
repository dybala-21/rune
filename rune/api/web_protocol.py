"""Web channel protocol - SSE event types + REST request types.

Ported 1:1 from src/daemon/web-protocol.ts - defines the SSE event
vocabulary for Server→Client streaming and REST request types for
Client→Server interactions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Token usage (shared across multiple events)

class TokenUsage(BaseModel):
    total: int = 0
    input: int = 0
    output: int = 0
    cache_read: int | None = Field(None, alias="cacheRead")
    cache_creation: int | None = Field(None, alias="cacheCreation")
    model_config = ConfigDict(populate_by_name=True)


# SSE Event Types (Server → Client)

SseEventType = Literal[
    # Connection
    "connected",
    # Agent lifecycle (legacy - backward compat)
    "agent_start", "agent_complete", "agent_error", "agent_aborted",
    # v1 run lifecycle (runId required)
    "run_started", "run_completed", "run_error", "run_aborted", "progress",
    # Real-time streaming
    "step_start", "thinking", "tool_call", "tool_result", "text_delta",
    # Interactive (client must respond via REST)
    "approval_request", "question",
    # Proactive lifecycle
    "suggestion_created", "proactive_execution_started",
    "proactive_execution_completed", "autonomy_level_changed",
    # Meta
    "context_compaction", "delegate_event",
    # Heartbeat
    "heartbeat",
]


class SseEvent(BaseModel):
    """A single SSE event to broadcast to clients."""
    event: SseEventType
    data: dict[str, Any] = Field(default_factory=dict)

    def format_sse(self, event_id: int | None = None) -> str:
        """Format as SSE wire protocol string."""
        from rune.utils.fast_serde import json_encode
        lines: list[str] = []
        if event_id is not None:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {self.event}")
        lines.append(f"data: {json_encode(self.data)}")
        lines.append("")
        lines.append("")
        return "\n".join(lines)


# Predefined event constructors

def connected_event(client_id: str) -> SseEvent:
    return SseEvent(event="connected", data={"clientId": client_id})

def agent_start_event(goal: str, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"goal": goal}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="agent_start", data=d)

def agent_complete_event(
    success: bool, answer: str, duration_ms: int,
    usage: dict[str, Any] | None = None, run_id: str | None = None,
) -> SseEvent:
    d: dict[str, Any] = {"success": success, "answer": answer, "durationMs": duration_ms}
    if usage:
        d["usage"] = usage
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="agent_complete", data=d)

def agent_error_event(error: str, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"error": error}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="agent_error", data=d)

def agent_aborted_event(run_id: str | None = None) -> SseEvent:
    return SseEvent(event="agent_aborted", data={"runId": run_id} if run_id else {})

def run_started_event(run_id: str, session_id: str, goal: str) -> SseEvent:
    return SseEvent(event="run_started", data={
        "runId": run_id, "sessionId": session_id, "goal": goal,
    })

def run_completed_event(
    run_id: str, success: bool, answer: str, duration_ms: int,
    usage: dict[str, Any] | None = None,
) -> SseEvent:
    d: dict[str, Any] = {
        "runId": run_id, "success": success, "answer": answer, "durationMs": duration_ms,
    }
    if usage:
        d["usage"] = usage
    return SseEvent(event="run_completed", data=d)

def run_error_event(run_id: str, error: str) -> SseEvent:
    return SseEvent(event="run_error", data={"runId": run_id, "error": error})

def run_aborted_event(run_id: str) -> SseEvent:
    return SseEvent(event="run_aborted", data={"runId": run_id})

def progress_event(run_id: str, phase: str, action: str) -> SseEvent:
    return SseEvent(event="progress", data={"runId": run_id, "phase": phase, "action": action})

def step_start_event(step_number: int, tokens: int, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"stepNumber": step_number, "tokens": tokens}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="step_start", data=d)

def thinking_event(text: str, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"text": text}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="thinking", data=d)

def tool_call_event(tool_name: str, args: dict[str, Any], run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"toolName": tool_name, "args": args}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="tool_call", data=d)

def tool_result_event(
    tool_name: str, result: str, success: bool, run_id: str | None = None,
) -> SseEvent:
    d: dict[str, Any] = {"toolName": tool_name, "result": result, "success": success}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="tool_result", data=d)

def text_delta_event(text: str, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"text": text}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="text_delta", data=d)

def approval_request_event(
    approval_id: str, command: str, risk_level: str,
    timeout_ms: int = 300_000, reason: str | None = None,
    run_id: str | None = None,
) -> SseEvent:
    d: dict[str, Any] = {
        "id": approval_id, "command": command,
        "riskLevel": risk_level, "timeoutMs": timeout_ms,
    }
    if reason:
        d["reason"] = reason
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="approval_request", data=d)

def question_event(
    question_id: str, question: str,
    options: list[dict[str, str]] | None = None,
    input_mode: str = "text", run_id: str | None = None,
) -> SseEvent:
    d: dict[str, Any] = {"id": question_id, "question": question}
    if options:
        d["options"] = options
    d["inputMode"] = input_mode
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="question", data=d)

def suggestion_created_event(
    suggestion_id: str, type: str, description: str,
    priority: str, confidence: float,
    action: dict[str, Any] | None = None,
) -> SseEvent:
    d: dict[str, Any] = {
        "id": suggestion_id, "type": type, "description": description,
        "priority": priority, "confidence": confidence,
    }
    if action:
        d["action"] = action
    return SseEvent(event="suggestion_created", data=d)

def context_compaction_event(message: str, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"message": message}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="context_compaction", data=d)

def delegate_event(stage: str, message: str, run_id: str | None = None) -> SseEvent:
    d: dict[str, Any] = {"stage": stage, "message": message}
    if run_id:
        d["runId"] = run_id
    return SseEvent(event="delegate_event", data=d)

def heartbeat_event() -> SseEvent:
    return SseEvent(event="heartbeat", data={})


# REST Request Types (Client → Server)

class MessageRequest(BaseModel):
    """POST /api/message"""
    text: str

class ApprovalRequest(BaseModel):
    """POST /api/approval"""
    id: str
    decision: Literal["approve_once", "approve_always", "deny"]
    user_guidance: str | None = Field(None, alias="userGuidance")
    model_config = ConfigDict(populate_by_name=True)

class QuestionRequest(BaseModel):
    """POST /api/question"""
    id: str
    answer: str
    selected_index: int | None = Field(None, alias="selectedIndex")
    model_config = ConfigDict(populate_by_name=True)
