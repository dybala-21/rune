"""WebSocket / SSE protocol types for RUNE daemon-web communication.

Ported from src/daemon/web-protocol.ts - defines SSE event types
(server -> client) and REST request types (client -> server) used by
the web channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from rune.utils.fast_serde import json_decode, json_encode

# ============================================================================
# Token usage
# ============================================================================

@dataclass(slots=True)
class TokenUsage:
    """Token consumption counters for a single run."""

    total: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "total": self.total,
            "input": self.input,
            "output": self.output,
        }
        if self.cache_read:
            d["cacheRead"] = self.cache_read
        if self.cache_creation:
            d["cacheCreation"] = self.cache_creation
        return d


# ============================================================================
# SSE Event types (Server -> Client)
# ============================================================================

class SseEventType(StrEnum):
    """All SSE event types the daemon can emit."""

    # Connection
    CONNECTED = "connected"

    # Legacy agent lifecycle (backward-compatible)
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    AGENT_ABORTED = "agent_aborted"

    # v1 run lifecycle (run_id required)
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_ERROR = "run_error"
    RUN_ABORTED = "run_aborted"
    PROGRESS = "progress"

    # Real-time streaming (NativeEventHandlers mapping)
    STEP_START = "step_start"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TEXT_DELTA = "text_delta"

    # Interactive (client must respond via POST)
    APPROVAL_REQUEST = "approval_request"
    QUESTION = "question"

    # Proactive lifecycle
    SUGGESTION_CREATED = "suggestion_created"
    PROACTIVE_EXECUTION_STARTED = "proactive_execution_started"
    PROACTIVE_EXECUTION_COMPLETED = "proactive_execution_completed"
    AUTONOMY_LEVEL_CHANGED = "autonomy_level_changed"

    # Meta
    CONTEXT_COMPACTION = "context_compaction"
    DELEGATE_EVENT = "delegate_event"


# ============================================================================
# SSE event dataclass
# ============================================================================

@dataclass(slots=True)
class SseEvent:
    """A single Server-Sent Event."""

    event: SseEventType
    data: dict[str, Any]
    id: str | None = None

    def serialize(self) -> str:
        """Serialize to the SSE wire format (``event:`` + ``data:`` lines).

        Each event is terminated by a blank line per the SSE spec.
        """
        lines: list[str] = []
        if self.id is not None:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event.value}")
        lines.append(f"data: {json_encode(self.data)}")
        lines.append("")  # blank line terminator
        return "\n".join(lines) + "\n"

    @classmethod
    def connected(cls, client_id: str) -> SseEvent:
        return cls(event=SseEventType.CONNECTED, data={"clientId": client_id})

    @classmethod
    def run_started(
        cls, run_id: str, session_id: str, goal: str
    ) -> SseEvent:
        return cls(
            event=SseEventType.RUN_STARTED,
            data={"runId": run_id, "sessionId": session_id, "goal": goal},
        )

    @classmethod
    def run_completed(
        cls,
        run_id: str,
        *,
        success: bool,
        answer: str,
        duration_ms: float,
        usage: TokenUsage | None = None,
    ) -> SseEvent:
        d: dict[str, Any] = {
            "runId": run_id,
            "success": success,
            "answer": answer,
            "durationMs": duration_ms,
        }
        if usage is not None:
            d["usage"] = usage.to_dict()
        return cls(event=SseEventType.RUN_COMPLETED, data=d)

    @classmethod
    def run_error(cls, run_id: str, error: str) -> SseEvent:
        return cls(
            event=SseEventType.RUN_ERROR,
            data={"runId": run_id, "error": error},
        )

    @classmethod
    def run_aborted(cls, run_id: str) -> SseEvent:
        return cls(
            event=SseEventType.RUN_ABORTED, data={"runId": run_id}
        )

    @classmethod
    def progress(cls, run_id: str, phase: str, action: str) -> SseEvent:
        return cls(
            event=SseEventType.PROGRESS,
            data={"runId": run_id, "phase": phase, "action": action},
        )

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        args: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> SseEvent:
        d: dict[str, Any] = {"toolName": tool_name, "args": args}
        if run_id is not None:
            d["runId"] = run_id
        return cls(event=SseEventType.TOOL_CALL, data=d)

    @classmethod
    def tool_result(
        cls,
        tool_name: str,
        result: str,
        *,
        success: bool = True,
        run_id: str | None = None,
    ) -> SseEvent:
        d: dict[str, Any] = {
            "toolName": tool_name,
            "result": result,
            "success": success,
        }
        if run_id is not None:
            d["runId"] = run_id
        return cls(event=SseEventType.TOOL_RESULT, data=d)

    @classmethod
    def text_delta(cls, text: str, *, run_id: str | None = None) -> SseEvent:
        d: dict[str, Any] = {"text": text}
        if run_id is not None:
            d["runId"] = run_id
        return cls(event=SseEventType.TEXT_DELTA, data=d)

    @classmethod
    def thinking(cls, text: str, *, run_id: str | None = None) -> SseEvent:
        d: dict[str, Any] = {"text": text}
        if run_id is not None:
            d["runId"] = run_id
        return cls(event=SseEventType.THINKING, data=d)

    @classmethod
    def approval_request(
        cls,
        *,
        request_id: str,
        command: str,
        risk_level: str,
        timeout_ms: int,
        reason: str | None = None,
        run_id: str | None = None,
    ) -> SseEvent:
        d: dict[str, Any] = {
            "id": request_id,
            "command": command,
            "riskLevel": risk_level,
            "timeoutMs": timeout_ms,
        }
        if reason is not None:
            d["reason"] = reason
        if run_id is not None:
            d["runId"] = run_id
        return cls(event=SseEventType.APPROVAL_REQUEST, data=d)

    @classmethod
    def question(
        cls,
        *,
        question_id: str,
        question: str,
        options: list[dict[str, str]] | None = None,
        input_mode: str | None = None,
        run_id: str | None = None,
    ) -> SseEvent:
        d: dict[str, Any] = {"id": question_id, "question": question}
        if options is not None:
            d["options"] = options
        if input_mode is not None:
            d["inputMode"] = input_mode
        if run_id is not None:
            d["runId"] = run_id
        return cls(event=SseEventType.QUESTION, data=d)


# ============================================================================
# REST request types (Client -> Server)
# ============================================================================

@dataclass(slots=True)
class MessageRequest:
    """POST /api/message - client sends a new message/goal."""

    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageRequest:
        return cls(text=str(data.get("text", "")))


@dataclass(slots=True)
class ApprovalResponse:
    """POST /api/approval - client responds to an approval request."""

    id: str
    decision: str  # "approve_once" | "approve_always" | "deny"
    user_guidance: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalResponse:
        return cls(
            id=str(data["id"]),
            decision=str(data["decision"]),
            user_guidance=data.get("userGuidance"),
        )

    def is_approved(self) -> bool:
        return self.decision in ("approve_once", "approve_always")


@dataclass(slots=True)
class QuestionResponse:
    """POST /api/question - client responds to a question."""

    id: str
    answer: str
    selected_index: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionResponse:
        return cls(
            id=str(data["id"]),
            answer=str(data.get("answer", "")),
            selected_index=data.get("selectedIndex"),
        )


# ============================================================================
# WebSocket handshake helpers
# ============================================================================

@dataclass(slots=True)
class HandshakeMessage:
    """Initial handshake exchanged when a WebSocket client connects."""

    client_id: str = field(default_factory=lambda: uuid4().hex[:16])
    protocol_version: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clientId": self.client_id,
            "protocolVersion": self.protocol_version,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandshakeMessage:
        return cls(
            client_id=data.get("clientId", uuid4().hex[:16]),
            protocol_version=int(data.get("protocolVersion", 1)),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def serialize(self) -> str:
        return json_encode(self.to_dict())

    @classmethod
    def deserialize(cls, raw: str | bytes) -> HandshakeMessage:
        data = json_decode(raw)
        return cls.from_dict(data)


def create_handshake_response(client_id: str) -> str:
    """Build the JSON string the server sends after accepting a WS handshake."""
    return SseEvent.connected(client_id).serialize()
