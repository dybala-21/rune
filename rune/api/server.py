"""FastAPI server for RUNE.

Ported from src/api/server.ts - REST + SSE + WebSocket API with token-based
auth, CORS, streaming agent execution, and session management.

Supports three real-time protocols:
- SSE (GET /api/v1/events) - Server→Client event stream
- NDJSON (POST /api/v1/agent/execute) - Streaming execution
- WebSocket (/ws) - Bidirectional real-time communication
"""

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

from rune.agent.timing import timed
from rune.api.questions import PendingQuestion, question_payload
from rune.api.trust import build_cancelled_trust
from rune.api.trust import build_trust_payload as build_trust_payload
from rune.capabilities.ask_user import AskUserParams, UserResponse, user_response
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)


def join_steps(collected: list[str], step_starts: list[int]) -> str:
    """Join collected deltas, separating each step's narration with a blank line.

    Deltas stream token by token, so within a step they concatenate directly.
    Across steps they must not: each step is a separate remark ("Opening the
    page." / "Now searching.") and running them together produces one unreadable
    paragraph. Steps that produced no text drop out.
    """
    # Force a 0 boundary: an empty step_starts, or one that begins past the
    # first delta, would otherwise leave that text outside every window and
    # return it as nothing.
    bounds = sorted({0, *step_starts, len(collected)})
    parts = []
    for begin, end in pairwise(bounds):
        chunk = "".join(collected[begin:end]).strip()
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts)


class StreamJoiner:
    """Incremental equivalent of :func:`join_steps` for a live stream.

    ``join_steps`` re-joins everything collected so far on every delta: O(n) per
    token, O(n^2) over an answer. Only the open step can grow and stripping is
    prefix-stable, so the new text follows from the tail alone and the full
    transcript is built only when asked for.

    :attr:`text` is byte-identical to ``join_steps`` at every point.
    """

    __slots__ = ("_closed", "_open_parts", "_seen_nonws", "_pending_ws")

    def __init__(self) -> None:
        self._closed: list[str] = []      # finished steps, already stripped
        self._open_parts: list[str] = []  # raw deltas of the current step
        self._seen_nonws = False          # has the open step produced text yet
        self._pending_ws = ""             # whitespace held back after the last emit

    def start_step(self) -> None:
        """Close the current step. A step that produced no text drops out."""
        chunk = "".join(self._open_parts).strip()
        if chunk:
            self._closed.append(chunk)
        self._open_parts = []
        self._seen_nonws = False
        self._pending_ws = ""

    def append(self, delta: str) -> str:
        """Add a delta and return only the text it made visible.

        Costs O(len(delta) + held-back whitespace), not O(transcript).
        """
        self._open_parts.append(delta)

        if not self._seen_nonws:
            # Leading whitespace in a step is dropped, so nothing is visible
            # until the first real character arrives.
            head = delta.lstrip()
            if not head:
                return ""
            self._seen_nonws = True
            visible = head.rstrip()
            self._pending_ws = head[len(visible):]
            # Steps are separated by a blank line, and the separator only
            # exists once a later step actually has text.
            return ("\n\n" + visible) if self._closed else visible

        candidate = self._pending_ws + delta
        visible = candidate.rstrip()
        self._pending_ws = candidate[len(visible):]
        return visible

    @property
    def text(self) -> str:
        """The whole transcript so far. Built on demand, not per delta."""
        parts = list(self._closed)
        tail = "".join(self._open_parts).strip()
        if tail:
            parts.append(tail)
        return "\n\n".join(parts)


def split_answer(collected: list[str], step_starts: list[int]) -> tuple[str, str]:
    """(full transcript, user-facing answer) for a finished run.

    The answer is the narration of the last step that produced any text. A
    multi-pass run otherwise concatenates every pass's narration and the final
    message shows duplicated summaries. Crucially, a trailing step with no text
    of its own — a tool-only round, or a re-observe/verify gate that ends the
    run silently — must not blank the answer or fall back to dumping the whole
    (duplicated) transcript, so walk back to the last step that actually spoke.
    Memory extraction still gets the full transcript. One helper so the rule
    can't drift between the SSE, NDJSON, and interrupted-stream paths.
    """
    # `full` feeds memory extraction, not the chat, so it stays a raw transcript.
    full = "".join(collected)
    for start in reversed(step_starts):
        if "".join(collected[start:]).strip():
            rel = [s - start for s in step_starts if s >= start] or [0]
            return full, join_steps(collected[start:], rel)
    return full, full


# How long an approval card stays actionable before the run gives up.
_APPROVAL_TIMEOUT_MS = 120_000

# Clients send the decision the UI offered — approve_once / approve_always /
# deny (see ApprovalRequestModel). Plain "approve" is accepted for older
# callers. Anything else, including an empty payload, denies.
_APPROVE_DECISIONS = frozenset({"approve", "approve_once", "approve_always"})


def approval_granted(result: dict[str, Any] | None) -> bool:
    """Whether an approval response allows the operation to proceed."""
    if not result:
        return False
    return str(result.get("decision", "")).strip().lower() in _APPROVE_DECISIONS


# SSE Client Manager


class SseClientManager:
    """Manages connected SSE clients with heartbeat and broadcasting."""

    def __init__(self) -> None:
        self._clients: dict[str, asyncio.Queue[str]] = {}
        self._event_counter = 0

    def add_client(self, client_id: str) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=512)
        self._clients[client_id] = queue
        return queue

    def remove_client(self, client_id: str) -> None:
        self._clients.pop(client_id, None)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def broadcast(self, event: str, data: dict[str, Any]) -> None:
        """Broadcast an SSE event to all connected clients."""
        self._event_counter += 1
        eid = self._event_counter
        formatted = (
            f"id: {eid}\nevent: {event}\n"
            f"data: {json_encode(data)}\n\n"
        )
        for queue in list(self._clients.values()):
            try:
                queue.put_nowait(formatted)
            except asyncio.QueueFull:
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait('event: resync_required\ndata: {}\n\n')

    def send_to(self, client_id: str, event: str, data: dict[str, Any]) -> None:
        """Send an SSE event to a specific client."""
        self._event_counter += 1
        eid = self._event_counter
        formatted = (
            f"id: {eid}\nevent: {event}\n"
            f"data: {json_encode(data)}\n\n"
        )
        queue = self._clients.get(client_id)
        if queue:
            with suppress(asyncio.QueueFull):
                queue.put_nowait(formatted)


# WebSocket Client Manager


class WsClientManager:
    """Manages connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}  # client_id -> WebSocket

    def add_client(self, client_id: str, ws: Any) -> None:
        self._clients[client_id] = ws

    def remove_client(self, client_id: str) -> None:
        self._clients.pop(client_id, None)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def broadcast(self, event: str, data: dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSocket clients."""
        msg = json_encode({"event": event, "data": data})
        disconnected: list[str] = []
        # Copy the clients before awaiting sends; connections may change meanwhile.
        for cid, ws in list(self._clients.items()):
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(cid)
        for cid in disconnected:
            self._clients.pop(cid, None)

    async def send_to(
        self, client_id: str, event: str, data: dict[str, Any]
    ) -> None:
        """Send a message to a specific WebSocket client."""
        ws = self._clients.get(client_id)
        if ws:
            try:
                await ws.send_text(
                    json_encode(
                        {"event": event, "data": data}, ensure_ascii=False
                    )
                )
            except Exception:
                self._clients.pop(client_id, None)


def create_app() -> Any:
    """Create and configure the FastAPI application.

    Returns a FastAPI instance with all routes, middleware, and event
    handlers configured.  All internal helpers are defined inside this
    function so they share the same closure over application state.
    """
    try:
        from fastapi import (
            Depends,
            FastAPI,
            HTTPException,
            Request,
            WebSocket,
            WebSocketDisconnect,
        )
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise ImportError(
            "FastAPI and uvicorn are required for the API server. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    from rune.api.auth import TokenAuthDependency

    # Active tasks - declared early so the lifespan can reference them.
    _active_tasks: dict[str, asyncio.Task[Any]] = {}
    _shutting_down = False
    from rune.api.computer import Computers, computer_router
    _computers = Computers(record=lambda event, data: _run_snapshots.record(event, data))
    # A final snapshot may add receipts after the initial stop notification.
    _aborted_runs: dict[str, dict[str, Any]] = {}

    def _finish_run(rid: str) -> None:
        _active_tasks.pop(rid, None)
        _aborted_runs.pop(rid, None)
    # Heartbeats may repeat suggestions already sent to clients.
    _broadcast_suggestion_ids: set[str] = set()

    def _on_proactive_suggestion(suggestions: list[Any]) -> None:
        """Broadcast each suggestion once for display in the chat."""
        for s in suggestions:
            sid = getattr(s, "id", "")
            if sid and sid in _broadcast_suggestion_ids:
                continue
            if sid:
                _broadcast_suggestion_ids.add(sid)
            conf = getattr(s, "confidence", 0.0)
            priority = ("high" if conf >= 0.8
                        else "medium" if conf >= 0.6 else "low")
            _sse_manager.broadcast("suggestion_created", {
                "id": getattr(s, "id", ""),
                "type": getattr(s, "type", "insight"),
                "title": getattr(s, "title", ""),
                "description": getattr(s, "description", ""),
                "priority": priority,
                "confidence": conf,
                "source": getattr(s, "source", ""),
            })

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[arg-type]
        nonlocal _shutting_down
        _run_snapshots.open()
        _maintenance.start()
        computer_maintenance = asyncio.create_task(_computers.maintain())
        log.info("api_server_started")
        _proactive_engine = None
        try:
            from rune.proactive.engine import get_proactive_engine
            _proactive_engine = get_proactive_engine()
            _proactive_engine.on("suggestion", _on_proactive_suggestion)
        except Exception as exc:
            log.debug("proactive_sse_subscribe_failed", error=str(exc))
        try:
            yield
        finally:
            _shutting_down = True
            if _proactive_engine is not None:
                with contextlib.suppress(Exception):
                    _proactive_engine.off("suggestion", _on_proactive_suggestion)
            try:
                tasks = list(_active_tasks.values())
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                _run_snapshots.interrupt_active("server_shutdown")
            finally:
                computer_maintenance.cancel()
                await asyncio.gather(computer_maintenance, return_exceptions=True)
                await _computers.close()
                await _maintenance.close()
                _run_snapshots.close()
                log.info("api_server_stopped")

    app = FastAPI(
        title="RUNE API",
        description="RUNE AI Development Environment API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware

    # CORS - use restrictive policy from cors_policy module.
    # Default (no RUNE_CORS_ORIGINS env): same-origin only.
    # To allow specific origins: RUNE_CORS_ORIGINS=http://localhost:3000,https://app.example.com
    from rune.api.cors_policy import get_allowed_origins_from_env

    _cors_origins_env = get_allowed_origins_from_env()
    if _cors_origins_env and _cors_origins_env.strip() == "*":
        # Explicit wildcard - no credentials
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )
    elif _cors_origins_env:
        _allowed = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )
    else:
        # Default: no cross-origin allowed (same-origin only)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[],
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # Auth dependency
    auth = TokenAuthDependency()
    app.include_router(computer_router(_computers, auth))
    from rune.api.desktop import desktop_router
    app.include_router(desktop_router(_computers, auth, stop_run=lambda rid: _stop_run(rid)))

    # REST API routers
    from rune.api.handlers.mcp import router as mcp_router
    app.include_router(mcp_router, prefix="/api/v1")

    # Request / Response models

    class ExecuteRequest(BaseModel):
        goal: str = Field(max_length=100_000)
        sender_id: str = Field(default="", max_length=256)
        session_id: str | None = Field(default=None, max_length=256)
        model: str | None = Field(default=None, max_length=256)
        stream: bool = False

    class ExecuteResponse(BaseModel):
        request_id: str
        status: str
        result: str | None = None

    class SessionInfo(BaseModel):
        session_id: str
        goal: str = ""
        started_at: str = ""

    class HealthResponse(BaseModel):
        status: str
        version: str
        uptime_seconds: float

    class MessageAttachment(BaseModel):
        name: str
        mimeType: str = ""
        data: str = ""

    class MessageRequest(BaseModel):
        text: str = ""
        attachments: list[MessageAttachment] | None = None
        # Conversation pin; without it the server-side sticky conversation
        # keeps live-chat continuity.
        session_id: str | None = Field(default=None, alias="sessionId")

        model_config = ConfigDict(populate_by_name=True)

    class ApprovalRequestModel(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        id: str
        decision: str
        user_guidance: str | None = Field(default=None, alias="userGuidance")
        response_id: str = Field(default="", alias="responseId", max_length=128)

    class QuestionRequestModel(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        id: str
        answer: str
        selected_index: int | None = Field(default=None, alias="selectedIndex", strict=True, ge=-1)
        response_id: str = Field(default="", alias="responseId", max_length=128)

    # State

    _start_time = time.monotonic()
    _active_loops: dict[str, Any] = {}  # run_id -> NativeAgentLoop
    _sse_manager = SseClientManager()
    _ws_manager = WsClientManager()
    _pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}
    _pending_questions: dict[str, PendingQuestion] = {}
    from rune.agent.execution_journal import (
        ExecutionJournal,
        RecoveryBlocked,
        journal_scope,
        recovery_context,
        recovery_written_files,
    )
    from rune.api.run_maintenance import RunMaintenance
    from rune.api.run_recovery import ResumeRequest, RunRecovery
    from rune.api.run_snapshot import RunSnapshots
    from rune.api.run_store import RunStore
    _run_store = RunStore()
    _run_snapshots = RunSnapshots(_run_store)
    _run_recovery = RunRecovery(_run_snapshots, _run_store)
    _maintenance = RunMaintenance(_run_store)

    # Broadcast helper

    async def _broadcast(event: str, data: dict[str, Any]) -> None:
        """Broadcast to both SSE and WebSocket clients."""
        data = _run_snapshots.record(event, data)
        if data is None:
            return
        _sse_manager.broadcast(event, data)
        await _ws_manager.broadcast(event, data)

    def _trust_payload(trace: Any) -> dict[str, Any]:
        return build_trust_payload(trace)

    async def _broadcast_aborted(run_id: str, trace: Any = None) -> None:
        if _shutting_down:
            return
        loop = _active_loops.get(run_id)
        trust = build_cancelled_trust(trace, artifact_receipts=getattr(loop, "artifact_receipts", []))
        payload = {"runId": run_id, "trust": trust}
        if _aborted_runs.get(run_id) != payload:
            _aborted_runs[run_id] = payload
            await _broadcast("agent_aborted", payload)

    # Agent execution helpers (inside create_app for closure access)

    async def _run_agent_for_client(
        goal: str,
        run_id: str,
        client_id: str | None,
        attachments: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        sticky: bool = False,
        agent_config: Any = None,
        resume_from: dict[str, Any] | None = None,
        resume_records: list[dict[str, Any]] | None = None,
    ) -> str:
        """Run the agent and broadcast progress to SSE/WS clients.

        session_id selects the conversation used for history. With no ID,
        sticky reuses the default web conversation; headless callers leave it off.
        """
        trace = None
        computer = None
        try:
            from rune.agent.agent_context import (
                PrepareContextOptions,
                prepare_agent_context,
            )
            from rune.agent.loop import NativeAgentLoop
            from rune.api import conversation_wiring as conv_wiring

            conv_manager = conv_wiring.get_conv_manager()
            conv_id: str | None = None
            if conv_manager is not None:
                try:
                    conv_id = await conv_wiring.resolve_conversation(
                        conv_manager, session_id, sticky=sticky,
                    )
                except Exception as exc:
                    log.debug("web_conv_resolve_failed", error=str(exc)[:100])
                    conv_manager = None
            computer = await _computers.claim(conv_id or session_id or run_id, run_id)
            if conv_manager is not None and conv_id and resume_from is None:
                conv_wiring.record_user_turn(conv_manager, conv_id, goal, attachments)
                await conv_manager._store.save(conv_manager._active[conv_id], embed=False)

            _run_snapshots.start(run_id, conv_id or session_id or "", goal)

            workspace = await conv_wiring.get_workspace(conv_id or "")
            # Use the user workspace rather than the daemon's launch directory.
            # Leave pinned_cwd unset so @path can still override it.
            from rune.utils.paths import user_workspace
            turn_cwd = workspace or str(user_workspace())
            if resume_from is not None:
                turn_cwd = resume_from["workspace"]
                workspace = turn_cwd
            _run_recovery.workspace_available(run_id, turn_cwd, resuming=resume_from is not None)
            _run_snapshots.record("run_context", {"runId": run_id, "workspace": turn_cwd})

            agent_ctx = await prepare_agent_context(
                PrepareContextOptions(
                    goal=goal,
                    channel="web",
                    cwd=turn_cwd,
                    pinned_cwd=workspace,
                    attachments=attachments or [],
                    conversation_id=conv_id or "",
                ),
                conversation_manager=conv_manager,
            )
            _run_recovery.workspace_available(run_id, agent_ctx.workspace_root, resuming=resume_from is not None)
            if conv_manager is not None and conv_id and not workspace:
                await conv_wiring.set_workspace(conv_id, agent_ctx.workspace_root)

            loop = NativeAgentLoop(config=agent_config) if agent_config else NativeAgentLoop()
            _active_loops[run_id] = loop

            approval_lock = asyncio.Lock()

            @timed("approval")
            async def _request_approval(command: str, reason: str) -> bool:
                approval_id = f"approval:{run_id}:{uuid4().hex}"
                approval_future: asyncio.Future[dict[str, Any]] = (
                    asyncio.get_running_loop().create_future()
                )
                _pending_approvals[approval_id] = approval_future
                await _broadcast(
                    "approval_request",
                    {
                        "id": approval_id,
                        "command": command,
                        # No risk level is supplied by this callback.
                        "riskLevel": "",
                        "reason": reason,
                        "timeoutMs": _APPROVAL_TIMEOUT_MS,
                        "expiresAt": time.time() * 1000 + _APPROVAL_TIMEOUT_MS,
                        "runId": run_id,
                    },
                )
                try:
                    result = await asyncio.wait_for(
                        approval_future, timeout=_APPROVAL_TIMEOUT_MS / 1000
                    )
                    return approval_granted(result)
                except TimeoutError:
                    return False
                finally:
                    _pending_approvals.pop(approval_id, None)
                    await _broadcast("approval_closed", {"id": approval_id, "runId": run_id})

            async def _web_approval_callback(command: str, reason: str) -> bool:
                async with approval_lock:
                    return await _request_approval(command, reason)

            loop.set_approval_callback(_web_approval_callback)

            @timed("question")
            async def _web_ask_user_callback(
                params: AskUserParams,
            ) -> UserResponse:
                from rune.agent.loop import current_tool_call_id

                question_id = f"question:{run_id}:{uuid4().hex}"
                pending = PendingQuestion(params)
                _pending_questions[question_id] = pending
                try:
                    await _broadcast("question", {
                        **question_payload(params, question_id, run_id, current_tool_call_id()),
                        "expiresAt": time.time() * 1000 + 300_000,
                    })
                    return await asyncio.wait_for(pending.future, timeout=300.0)
                except TimeoutError:
                    raise TimeoutError("Question expired without a user response") from None
                finally:
                    _pending_questions.pop(question_id, None)
                    if not pending.future.done():
                        pending.future.cancel()
                    await _broadcast("question_closed", {"id": question_id, "runId": run_id})

            loop.set_ask_user_callback(_web_ask_user_callback)

            journal = ExecutionJournal(_run_store, run_id, agent_ctx.workspace_root,
                                       previous=resume_records if resume_from is not None else None,
                                       approval=_web_approval_callback)
            computer.journal = journal
            _run_snapshots.record("run_context", {
                "runId": run_id, "workspace": agent_ctx.workspace_root, "recoveryVersion": 1,
                "execution": {"attachments": attachments or []},
            })

            collected: list[str] = []

            _run_start_time = time.monotonic()
            # split_answer uses these offsets to separate the final answer from commentary.
            _step_starts = [0]

            # Send deltas to avoid rebuilding and broadcasting the full text per token.
            _joiner = StreamJoiner()

            async def _on_step(step: int) -> None:
                _step_starts.append(len(collected))
                _joiner.start_step()
                await _broadcast(
                    "step_start",
                    {"stepNumber": step, "tokens": 0, "runId": run_id},
                )

            async def _on_text(delta: str) -> None:
                collected.append(delta)
                chunk = _joiner.append(delta)
                if chunk:
                    await _broadcast(
                        "text_delta", {"delta": chunk, "runId": run_id}
                    )

            async def _on_tool(info: dict[str, Any]) -> None:
                await _broadcast(
                    "tool_call",
                    {
                        "toolName": info.get("name", ""),
                        "args": info.get("params", {}),
                        # Concurrent tool results can arrive out of order.
                        "callId": info.get("callId", ""),
                        "runId": run_id,
                    },
                )

            async def _on_tool_result(info: dict[str, Any]) -> None:
                await _broadcast(
                    "tool_result",
                    {
                        "toolName": info.get("name", ""),
                        "result": info.get("output_head", "")
                        or info.get("error_head", ""),
                        "success": info.get("success", True),
                        "checkStatus": info.get("check_status"),
                        "outputTruncated": info.get("output_truncated", False),
                        "callId": info.get("callId", ""),
                        "runId": run_id,
                        "artifactReceipts": getattr(loop, "artifact_receipts", []),
                        "fileChange": info.get("fileChange"),
                    },
                )

            loop.on("step", _on_step)
            loop.on("text_delta", _on_text)
            loop.on("tool_call", _on_tool)
            loop.on("tool_result", _on_tool_result)

            # Wire orchestrator events if the loop delegates to one
            def _hook_orchestrator(orchestrator: Any) -> None:
                """Relay orchestrator events to SSE/WS clients."""

                async def _on_plan(plan: Any) -> None:
                    tc = len(plan.tasks) if hasattr(plan, "tasks") else 0
                    await _broadcast(
                        "orchestration_started",
                        {"runId": run_id, "taskCount": tc,
                         "description": getattr(plan, "description", "")},
                    )

                async def _on_progress(
                    completed: int, total: int, task_id: str, success: bool,
                    description: str = "", role: str = "",
                ) -> None:
                    await _broadcast(
                        "orchestration_task_progress",
                        {"runId": run_id, "taskId": task_id,
                         "completed": completed, "total": total,
                         "success": success,
                         "description": description, "role": role},
                    )

                async def _on_retry(
                    task_id: str, failure_type: str, attempt: int, error: str,
                ) -> None:
                    await _broadcast(
                        "orchestration_task_retry",
                        {"runId": run_id, "taskId": task_id,
                         "failureType": failure_type, "attempt": attempt,
                         "error": error[:200]},
                    )

                async def _on_orch_done(result: Any) -> None:
                    results = getattr(result, "results", [])
                    ok = sum(1 for r in results if getattr(r, "success", False))
                    await _broadcast(
                        "orchestration_completed",
                        {"runId": run_id,
                         "success": getattr(result, "success", False),
                         "durationMs": round(getattr(result, "duration_ms", 0), 1),
                         "completedCount": ok,
                         "failedCount": len(results) - ok},
                    )

                orchestrator.on("plan_ready", _on_plan)
                orchestrator.on("progress", _on_progress)
                orchestrator.on("subtask_retry", _on_retry)
                orchestrator.on("completed", _on_orch_done)

            # Expose hook so delegate capability can call it
            loop._web_orchestrator_hook = _hook_orchestrator  # type: ignore[attr-defined]

            await _broadcast(
                "agent_start",
                {
                    "runId": run_id,
                    # Use the original goal; expanded @references can contain whole files.
                    "goal": agent_ctx.original_goal or agent_ctx.goal,
                    "sessionId": conv_id or session_id,
                    "fileChanges": (_run_snapshots.get(run_id) or {}).get("fileChanges", []),
                },
            )

            continuation = {}
            run_context = {"workspace_root": agent_ctx.workspace_root,
                           "original_goal": agent_ctx.original_goal or agent_ctx.goal,
                           "attachments": agent_ctx.metadata.get("attachments") or []}
            if resume_from is not None:
                continuation["extra_system_context"] = recovery_context(resume_from, resume_records or [])
                run_context["recovery_written_files"] = recovery_written_files(resume_records or [])
            with journal_scope(journal):
                async with _computers.bind(computer):
                    trace = await loop.run(
                        agent_ctx.goal, context=run_context,
                        message_history=agent_ctx.messages if agent_ctx.messages else None,
                        **continuation,
                    )
                journal.check()
            loop_finished = time.monotonic()
            full_text, answer = split_answer(collected, _step_starts)
            answer = getattr(loop, "_last_answer_text", "") or answer
            duration_ms = int((time.monotonic() - _run_start_time) * 1000)

            if conv_manager is not None and conv_id:
                for instruction in computer.instructions:
                    conv_wiring.record_user_turn(conv_manager, conv_id, instruction)
                await conv_wiring.record_assistant_turn(
                    conv_manager, conv_id, loop, answer,
                    reason=trace.reason or "",
                    embed=False, require_save=True,
                )

            _maintenance.enqueue(
                run_id, agent_ctx, trace, full_text, duration_ms,
                classification_hint=getattr(loop, "_last_goal_type", "") or None,
            )
            duration_ms = int((time.monotonic() - _run_start_time) * 1000)
            log.info("agent_delivery_ready", run_id=run_id,
                     completion_tail_ms=round((time.monotonic() - loop_finished) * 1000, 1))

            if trace.reason == "cancelled" or run_id in _aborted_runs:
                await _broadcast_aborted(run_id, trace)
                return answer

            await _broadcast(
                "agent_complete",
                {
                    "runId": run_id,
                    "success": trace.reason == "completed",
                    "answer": answer,
                    "durationMs": duration_ms,
                    "timings": {**getattr(trace, "timings", {}),
                                "deliveryMs": round((time.monotonic() - loop_finished) * 1000, 1)},
                    "trust": _trust_payload(trace),
                },
            )

            return answer

        except asyncio.CancelledError:
            await _broadcast_aborted(run_id, trace)
            raise
        except Exception as exc:
            log.error("agent_execution_error", run_id=run_id, error=str(exc))
            await _broadcast(
                "agent_error",
                {"runId": run_id, "error": f"Agent execution failed: {type(exc).__name__}"},
            )
            return f"error: {type(exc).__name__}"
        finally:
            if computer is not None:
                _computers.finish(computer, run_id)
            _active_loops.pop(run_id, None)

    async def _ndjson_execution(
        goal: str, run_id: str, session_id: str | None = None
    ) -> AsyncGenerator[str]:
        """Stream a run as NDJSON, sharing history only with an explicit session.

        Save the answer before emitting completion. The finally block also
        saves partial answers when a disconnect interrupts the stream.
        """
        conv_manager: Any | None = None
        conv_id: str | None = None
        loop = None
        run_task: asyncio.Task[Any] | None = None
        assistant_save_attempted = False
        collected: list[str] = []
        _step_starts = [0]
        _ws_joiner = StreamJoiner()

        def _encode_event(frame: dict[str, Any]) -> str:
            data = _run_snapshots.record(frame["event"], frame["data"])
            return json_encode({**frame, "data": data}) if data is not None else ""

        try:
            from rune.agent.agent_context import (
                PrepareContextOptions,
                prepare_agent_context,
            )
            from rune.agent.loop import NativeAgentLoop
            from rune.api import conversation_wiring as conv_wiring

            # 0. Resolve the conversation and record the user turn
            conv_manager = conv_wiring.get_conv_manager()
            if conv_manager is not None:
                try:
                    conv_id = await conv_wiring.resolve_conversation(
                        conv_manager, session_id, sticky=False,
                    )
                except Exception as exc:
                    log.debug("ndjson_conv_resolve_failed", error=str(exc)[:100])
                    conv_manager = None
            if conv_manager is not None and conv_id:
                conv_wiring.record_user_turn(conv_manager, conv_id, goal)
                await conv_manager._store.save(conv_manager._active[conv_id], embed=False)

            # 1. Prepare agent context (loads prior turns as history)
            agent_ctx = await prepare_agent_context(
                PrepareContextOptions(
                    goal=goal, channel="web", conversation_id=conv_id or "",
                ),
                conversation_manager=conv_manager,
            )

            _run_snapshots.record("run_context", {
                "runId": run_id, "sessionId": conv_id or session_id or "",
                "workspace": agent_ctx.workspace_root,
            })

            yield (
                _encode_event(
                    {
                        "event": "agent_start",
                        "data": {
                            "runId": run_id,
                            "goal": agent_ctx.original_goal or agent_ctx.goal,
                            "sessionId": session_id,
                        },
                    }
                )
                + "\n"
            )

            loop = NativeAgentLoop()
            _active_loops[run_id] = loop

            # Use a queue so event callbacks can feed the generator
            event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def _enqueue(frame: dict[str, Any]) -> None:
                data = _run_snapshots.record(frame["event"], frame["data"])
                if data is not None:
                    await event_queue.put({**frame, "data": data})

            # 2. NDJSON is unidirectional (server→client) - no way to receive
            #    approval/question responses. Use auto-approve + autonomous mode.
            async def _ndjson_approval_cb(command: str, risk_level: str) -> bool:
                await _enqueue(
                    {
                        "event": "approval_request",
                        "data": {
                            "id": f"ndjson:{run_id}:{uuid4().hex}",
                            "command": command,
                            "riskLevel": risk_level,
                            "runId": run_id,
                            "autoApproved": True,
                        },
                    }
                )
                return True

            loop.set_approval_callback(_ndjson_approval_cb)

            async def _ndjson_ask_user_cb(
                params: AskUserParams,
            ) -> UserResponse:
                await _enqueue(
                    {
                        "event": "question",
                        "data": {**question_payload(params, f"ndjson:{run_id}:{uuid4().hex}", run_id),
                                 "autonomous": True},
                    }
                )
                return user_response(params, "")

            loop.set_ask_user_callback(_ndjson_ask_user_cb)

            async def _on_step(step: int) -> None:
                _step_starts.append(len(collected))
                _ws_joiner.start_step()
                await _enqueue(
                    {
                        "event": "step_start",
                        "data": {"stepNumber": step, "tokens": 0, "runId": run_id},
                    }
                )

            async def _on_text(delta: str) -> None:
                # /ws is documented as sending the whole transcript, so the
                # format stays. The joiner says cheaply whether this delta made
                # anything visible; when it did not, skip the frame and the
                # rebuild it would cost.
                collected.append(delta)
                if not _ws_joiner.append(delta):
                    return
                await _enqueue(
                    {
                        "event": "text_delta",
                        "data": {"text": _ws_joiner.text, "runId": run_id},
                    }
                )

            async def _on_tool(info: dict[str, Any]) -> None:
                await _enqueue(
                    {
                        "event": "tool_call",
                        "data": {
                            "toolName": info.get("name", ""),
                            "args": info.get("params", {}),
                            "callId": info.get("callId", ""),
                            "runId": run_id,
                        },
                    }
                )

            async def _on_tool_result(info: dict[str, Any]) -> None:
                await _enqueue(
                    {
                        "event": "tool_result",
                        "data": {
                            "toolName": info.get("name", ""),
                            "result": info.get("output_head", "")
                            or info.get("error_head", ""),
                            "success": info.get("success", True),
                            "callId": info.get("callId", ""),
                            "artifactReceipts": getattr(loop, "artifact_receipts", []),
                            "runId": run_id,
                        },
                    }
                )

            loop.on("step", _on_step)
            loop.on("text_delta", _on_text)
            loop.on("tool_call", _on_tool)
            loop.on("tool_result", _on_tool_result)

            _run_start_time = time.monotonic()
            run_task = asyncio.create_task(
                loop.run(
                    agent_ctx.goal,
                    # No attachments here: this endpoint takes text only.
                    context={"workspace_root": agent_ctx.workspace_root,
                             "original_goal": agent_ctx.original_goal or agent_ctx.goal},
                    message_history=(
                        agent_ctx.messages if agent_ctx.messages else None
                    ),
                )
            )
            _active_tasks[run_id] = run_task
            run_task.add_done_callback(lambda _: event_queue.put_nowait(None))

            while True:
                try:
                    evt = await asyncio.wait_for(
                        event_queue.get(), timeout=2.0
                    )
                    if evt is None:
                        break
                    yield json_encode(evt) + "\n"
                except TimeoutError:
                    yield (
                        json_encode({"event": "heartbeat", "data": {}}) + "\n"
                    )

            trace = None if run_task.cancelled() else run_task.result()
            loop_finished = time.monotonic()
            cancelled = (run_task.cancelled() or getattr(trace, "reason", "") == "cancelled"
                         or run_id in _aborted_runs)
            full_text, answer = split_answer(collected, _step_starts)
            answer = getattr(loop, "_last_answer_text", "") or answer
            duration_ms = int((time.monotonic() - _run_start_time) * 1000)

            if conv_manager is not None and conv_id:
                assistant_save_attempted = True
                await conv_wiring.record_assistant_turn(
                    conv_manager, conv_id, loop, answer,
                    reason=getattr(trace, "reason", "cancelled"),
                    embed=False, require_save=True,
                )

            if trace is not None:
                _maintenance.enqueue(
                    run_id, agent_ctx, trace, full_text, duration_ms,
                    classification_hint=getattr(loop, "_last_goal_type", "") or None,
                )
            duration_ms = int((time.monotonic() - _run_start_time) * 1000)
            if cancelled:
                await _broadcast_aborted(run_id, trace)

            yield (
                _encode_event(
                    {
                        "event": "agent_aborted" if cancelled else "agent_complete",
                        "data": {
                            "runId": run_id,
                            "success": not cancelled and getattr(trace, "reason", "") == "completed",
                            "answer": answer,
                            "durationMs": duration_ms,
                            "timings": {**getattr(trace, "timings", {}),
                                        "deliveryMs": round((time.monotonic() - loop_finished) * 1000, 1)},
                            "trust": build_cancelled_trust(trace, artifact_receipts=getattr(loop, "artifact_receipts", []))
                            if cancelled else _trust_payload(trace),
                        },
                    }
                )
                + "\n"
            )

        except Exception as exc:
            log.error("agent_stream_error", run_id=run_id, error=str(exc))
            yield (
                _encode_event(
                    {
                        "event": "agent_error",
                        "data": {"runId": run_id, "error": f"Agent execution failed: {type(exc).__name__}"},
                    }
                )
                + "\n"
            )
        finally:
            _run_snapshots.record("agent_interrupted", {"runId": run_id, "interruptionReason": "stream_disconnected"})
            if run_task is not None and not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    log.debug("ndjson_run_cancelled", run_id=run_id)
                except Exception as exc:
                    log.warning("ndjson_cleanup_failed", run_id=run_id, error=str(exc))
            _active_loops.pop(run_id, None)
            _finish_run(run_id)
            # A disconnect can skip the save before the completion event.
            if not assistant_save_attempted and conv_manager is not None and conv_id and loop is not None:
                _, last_step_text = split_answer(collected, _step_starts)
                from rune.api import conversation_wiring as conv_wiring

                await conv_wiring.record_assistant_turn(
                    conv_manager, conv_id, loop, last_step_text,
                    reason="stream interrupted",
                    embed=False,
                )

    def _make_action_run_agent(session_id: str | None) -> Any:
        """Closure for slash-command actions (e.g. /escalate) that need to run
        a full agent turn on the live conversation with an optional per-run
        model override — without mutating global config."""

        async def _run(goal: str, agent_config: Any = None) -> str:
            run_id = uuid4().hex[:16]
            task = asyncio.create_task(
                _run_agent_for_client(
                    goal=goal, run_id=run_id, client_id=None,
                    session_id=session_id, sticky=True,
                    agent_config=agent_config,
                )
            )
            _active_tasks[run_id] = task
            task.add_done_callback(
                lambda _t, _rid=run_id: _finish_run(_rid)
            )
            return await task

        return _run

    # Health (no auth)

    @app.get("/health", response_model=HealthResponse)
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            uptime_seconds=time.monotonic() - _start_time,
        )

    # SSE Events endpoint (GET /api/v1/events)

    from fastapi import Query

    @app.get("/api/v1/files/download", dependencies=[Depends(auth)])
    async def download_workspace_file(path: str, session_id: str = Query(alias="sessionId")) -> Any:
        from rune.api.files import download_file

        return await download_file(session_id, path)

    @app.get("/api/v1/events", dependencies=[Depends(auth)])
    @app.get("/api/events", dependencies=[Depends(auth)])
    async def sse_events(request: Request) -> StreamingResponse:
        client_id = uuid4().hex[:16]
        queue = _sse_manager.add_client(client_id)

        async def _generate() -> AsyncGenerator[str]:
            # Send connected event
            yield (
                f"event: connected\n"
                f'data: {json_encode({"clientId": client_id})}\n\n'
            )

            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(
                            queue.get(), timeout=30.0
                        )
                        yield msg
                    except TimeoutError:
                        # Heartbeat every 30 s
                        yield (
                            f"event: heartbeat\n"
                            f'data: {json_encode({"ts": time.time()})}\n\n'
                        )
            finally:
                _sse_manager.remove_client(client_id)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/runs/snapshot", dependencies=[Depends(auth)])
    async def run_snapshot(session_id: str = Query(alias="sessionId", min_length=1)) -> dict[str, Any]:
        snapshot = _run_snapshots.latest(session_id)
        if snapshot:
            from rune.api.conversation_wiring import get_conv_manager

            manager = get_conv_manager()
            conversation = manager._active.get(session_id) if manager else None
            if conversation is None and manager is not None:
                conversation = await manager._store.load(session_id)
            if conversation is not None:
                snapshot["history"] = [
                    {"id": f"history:{session_id}:{index}", "role": turn.role,
                     "content": turn.content, "timestamp": turn.timestamp.timestamp() * 1000}
                    for index, turn in enumerate(conversation.turns[-1200:])
                ]
        return {"run": snapshot}

    @app.post("/api/runs/resume", dependencies=[Depends(auth)])
    async def resume_run(req: ResumeRequest) -> dict[str, Any]:
        try:
            child, source, records = _run_recovery.begin(req.run_id)
        except (RecoveryBlocked, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if source is not None:
            task = asyncio.create_task(_run_agent_for_client(
                goal=source["goal"], run_id=child["runId"], client_id=None,
                session_id=source["sessionId"], resume_from=source, resume_records=records,
                attachments=source.get("execution", {}).get("attachments") or None,
            ))
            _active_tasks[child["runId"]] = task
            task.add_done_callback(lambda _t, rid=child["runId"]: _finish_run(rid))
        return {"ok": True, "runId": child["runId"], "sessionId": child["sessionId"]}

    # Embedded terminal WebSocket (/ws/terminal) — opt-in, token-gated.
    # Protocol (terminado-style JSON arrays): client→ ["stdin", text] /
    # ["set_size", rows, cols]; server→ ["stdout", text] / ["disconnect", 1].

    @app.websocket("/ws/terminal")
    async def terminal_endpoint(ws: WebSocket) -> None:
        from rune.api import terminal as term
        from rune.api.local_auth_guard import (
            is_localhost_request,
            is_trusted_local_bypass_request,
        )

        if not term.is_enabled():
            await ws.close(code=4003, reason="Terminal is disabled")
            return

        # Loopback only.
        if not is_localhost_request(ws.client.host if ws.client else ""):
            await ws.close(code=4001, reason="Terminal is local-only")
            return
        # Origin/CSRF check (browsers don't apply same-origin to WebSockets),
        # mirroring /ws — defense-in-depth against a cross-site handshake on top
        # of the single-use token.
        server_port = ws.scope.get("server", (None, 0))[1] or 0
        headers = {k.decode(): v.decode() for k, v in ws.scope.get("headers", [])}
        if not is_trusted_local_bypass_request(headers, server_port):
            await ws.close(code=4001, reason="Cross-origin terminal handshake refused")
            return
        # Short-lived, single-use token minted via the auth-gated terminal.token
        # RPC. NOTE: same-origin renderer XSS can mint one — see terminal.py.
        workspace = term.redeem_token(ws.query_params.get("token", ""))
        if workspace is None:
            await ws.close(code=4001, reason="Invalid or spent terminal token")
            return

        await ws.accept()
        session = term.TerminalSession(workspace)
        try:
            session.start()
        except Exception as exc:
            log.warning("terminal_start_failed", error=str(exc)[:150])
            await ws.send_text(json_encode(["disconnect", 1]))
            await ws.close()
            return

        async def _pump_out() -> None:
            while True:
                chunk = await session.out_queue.get()
                session.notify_consumed()  # re-arm PTY reader if it was paused
                if chunk is None:
                    with suppress(Exception):
                        await ws.send_text(json_encode(["disconnect", 1]))
                    return
                with suppress(Exception):
                    await ws.send_text(
                        json_encode(["stdout", chunk.decode("utf-8", "replace")])
                    )

        pump = asyncio.create_task(_pump_out())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json_decode(raw)
                except Exception:
                    continue
                if not isinstance(msg, list) or not msg:
                    continue
                if msg[0] == "stdin" and len(msg) > 1:
                    session.write(str(msg[1]))
                elif msg[0] == "set_size" and len(msg) >= 3:
                    with suppress(Exception):
                        session.resize(int(msg[1]), int(msg[2]))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.debug("terminal_ws_error", error=str(exc)[:100])
        finally:
            pump.cancel()
            session.close()

    # WebSocket endpoint (/ws)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        # Authenticate WebSocket connections using the same local auth
        # guard as HTTP endpoints.  Non-localhost connections require a
        # Bearer token via the ``token`` query parameter.
        from rune.api.auth import verify_token
        from rune.api.local_auth_guard import (
            is_localhost_request,
            is_trusted_local_bypass_request,
        )

        client_host = ws.client.host if ws.client else ""
        is_local = is_localhost_request(client_host)

        if is_local:
            server_port = ws.scope.get("server", (None, 0))[1] or 0
            headers = {k.decode(): v.decode() for k, v in ws.scope.get("headers", [])}
            if not is_trusted_local_bypass_request(headers, server_port):
                # Local but cross-origin - require token
                token = ws.query_params.get("token", "")
                if not verify_token(token):
                    await ws.close(code=4001, reason="Unauthorized")
                    return
        else:
            token = ws.query_params.get("token", "")
            if not verify_token(token):
                await ws.close(code=4001, reason="Unauthorized")
                return

        await ws.accept()
        client_id = uuid4().hex[:16]
        _ws_manager.add_client(client_id, ws)

        # Send connected event
        await ws.send_text(
            json_encode(
                {"event": "connected", "data": {"clientId": client_id}}
            )
        )

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json_decode(raw)
                except json.JSONDecodeError:
                    await ws.send_text(
                        json_encode(
                            {
                                "event": "error",
                                "data": {"error": "Invalid JSON"},
                            }
                        )
                    )
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "message":
                    text = msg.get("text", "")
                    if text:
                        run_id = uuid4().hex[:16]
                        _run_snapshots.start(run_id, msg.get("sessionId") or "", text)
                        ws_attachments = msg.get("attachments") or None
                        task = asyncio.create_task(
                            _run_agent_for_client(
                                goal=text,
                                run_id=run_id,
                                client_id=client_id,
                                attachments=ws_attachments,
                                session_id=msg.get("sessionId") or None,
                                sticky=True,
                            )
                        )
                        _active_tasks[run_id] = task
                        task.add_done_callback(lambda _t, _rid=run_id: _finish_run(_rid))
                        await ws.send_text(
                            json_encode(
                                {
                                    "event": "agent_start",
                                    "data": {
                                        "runId": run_id,
                                        "goal": text,
                                    },
                                }
                            )
                        )

                elif msg_type == "abort":
                    run_id = msg.get("runId", "")
                    await _stop_run(run_id)

                elif msg_type == "approval":
                    try:
                        await api_approval(ApprovalRequestModel.model_validate(msg))
                    except (HTTPException, ValueError) as exc:
                        await ws.send_text(json_encode({"event": "error", "data": {"message": str(exc)}}))

                elif msg_type == "question":
                    try:
                        await api_question(QuestionRequestModel.model_validate(msg))
                    except (HTTPException, ValueError) as exc:
                        await ws.send_text(json_encode({"event": "error", "data": {"message": str(exc)}}))

                elif msg_type == "ping":
                    await ws.send_text(
                        json_encode({"event": "pong", "data": {}})
                    )

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.warning(
                "ws_client_error",
                client_id=client_id,
                error=str(exc)[:100],
            )
        finally:
            _ws_manager.remove_client(client_id)

    # Agent execution (POST /execute and POST /api/v1/agent/execute)

    @app.post(
        "/execute",
        response_model=ExecuteResponse,
        dependencies=[Depends(auth)],
    )
    @app.post("/api/v1/agent/execute", dependencies=[Depends(auth)])
    async def execute(req: ExecuteRequest) -> Any:
        run_id = uuid4().hex[:16]
        _run_snapshots.start(run_id, req.session_id or "", req.goal)

        if req.stream:
            # NDJSON streaming response
            return StreamingResponse(
                _ndjson_execution(req.goal, run_id, session_id=req.session_id),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Headless calls share conversation history only with an explicit session_id.
        task = asyncio.create_task(
            _run_agent_for_client(
                goal=req.goal, run_id=run_id, client_id=None,
                session_id=req.session_id,
            )
        )
        _active_tasks[run_id] = task

        try:
            result = await task
        finally:
            _active_tasks.pop(run_id, None)

        snapshot = _run_snapshots.get(run_id)
        return ExecuteResponse(
            request_id=run_id,
            status=snapshot["status"] if snapshot else "failed",
            result=result,
        )

    # Legacy REST endpoints

    @app.post("/api/message", dependencies=[Depends(auth)])
    async def api_message(req: MessageRequest) -> dict[str, Any]:
        # Slash commands run without a chat turn; __ACTION__ markers execute
        # server-side and the result goes out as a command_result SSE event.
        if req.text.startswith("/"):
            from rune.api import command_actions
            from rune.slash_commands import COMMANDS, parse_slash_command
            parsed = parse_slash_command(req.text)
            if parsed:
                cmd_name, args = parsed
                cmd = COMMANDS.get(cmd_name)
                if cmd:
                    try:
                        output = await command_actions.handle_direct_command(
                            cmd_name, args,
                        )
                        if output is None:
                            result = await cmd.handler(args)
                            output = result or f"{cmd_name} executed."
                            if isinstance(output, str) and output.startswith(
                                "__ACTION__:"
                            ):
                                action = output[len("__ACTION__:"):]
                                ctx = command_actions.ActionContext(
                                    broadcast=_broadcast,
                                    workspace=Path.cwd(),
                                    session_id=req.session_id,
                                    run_agent=_make_action_run_agent(
                                        req.session_id
                                    ),
                                    active_run_count=lambda: len(_active_loops),
                                    started_at=_start_time,
                                )
                                output = await command_actions.execute_action(
                                    action, ctx,
                                )
                    except Exception as exc:
                        output = f"Error: {exc}"
                    if output:
                        await _broadcast(
                            "command_result",
                            {
                                "command": cmd_name,
                                "output": output,
                                # Let clients filter commands from other conversations.
                                "requestSessionId": req.session_id,
                            },
                        )
                    return {"ok": True, "command": cmd_name}

        run_id = uuid4().hex[:16]
        raw_attachments = [
            {
                "name": os.path.basename(a.name) if a.name else f"attachment_{i}",
                "mimeType": a.mimeType,
                "data": a.data,
            }
            for i, a in enumerate(req.attachments or [])
        ]
        _run_snapshots.start(run_id, req.session_id or "", req.text)
        task = asyncio.create_task(
            _run_agent_for_client(
                goal=req.text, run_id=run_id, client_id=None,
                attachments=raw_attachments or None,
                session_id=req.session_id,
                sticky=True,
            )
        )
        _active_tasks[run_id] = task
        task.add_done_callback(lambda _t, _rid=run_id: _finish_run(_rid))
        # The client uses runId to route events and target Stop.
        return {"ok": True, "runId": run_id}

    @app.post("/api/voice/transcribe", dependencies=[Depends(auth)])
    async def api_voice_transcribe(request: Request) -> dict[str, Any]:
        """Transcribe uploaded audio (base64 JSON body) to text.

        Brings the CLI --voice capability to the app: the client records via
        MediaRecorder and sends {audio: <base64>, mimeType}.
        """
        import base64

        try:
            body = await request.json()
            audio_b64 = body.get("audio", "") if isinstance(body, dict) else ""
            if not audio_b64:
                return {"ok": False, "error": "No audio data."}
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return {"ok": False, "error": "Invalid request body."}

        try:
            from rune.voice.service import get_voice_service

            svc = get_voice_service()
            if not svc.has_stt:
                from rune.voice.availability import get_voice_install_hint

                hint = get_voice_install_hint() or (
                    "Set DEEPGRAM_API_KEY / OPENAI_API_KEY, or install "
                    "sherpa-onnx for local STT."
                )
                return {
                    "ok": False,
                    "error": f"No speech-to-text provider available. {hint}",
                }
            text = await svc.transcribe(audio_bytes)
            return {"ok": True, "text": text}
        except Exception as exc:
            log.warning("voice_transcribe_failed", error=str(exc)[:150])
            return {"ok": False, "error": f"Transcription failed: {type(exc).__name__}"}

    async def _stop_run(rid: str) -> None:
        await _computers.stop(rid)
        agent_loop = _active_loops.get(rid)
        if agent_loop:
            with contextlib.suppress(Exception):
                await agent_loop.cancel()
        task = _active_tasks.get(rid)
        if task and not task.done():
            task.cancel()
            # Let tool receipts settle before the run becomes terminal in the store.
            await asyncio.wait({task}, timeout=5)
        snapshot = _run_snapshots.get(rid)
        if (task is None or task.done()) and snapshot and snapshot["status"] not in {"completed", "failed", "cancelled"}:
            await _broadcast_aborted(rid)

    @app.post("/api/abort", dependencies=[Depends(auth)])
    async def api_abort(
        request: Request,
    ) -> dict[str, Any]:
        # Parse body if present; frontend may send empty body
        run_id = ""
        try:
            body = await request.json()
            run_id = body.get("runId", "") if isinstance(body, dict) else ""
        except Exception:
            pass

        if run_id:
            # Never fall back to another run when the requested run has already ended.
            if run_id in _active_loops or run_id in _active_tasks:
                await _stop_run(run_id)
                return {"ok": True}
            return {"ok": True, "stopped": False, "reason": "run is not active"}

        # No runId given (an older client): abort the most recent active run.
        rid = (list(_active_loops.keys())[-1] if _active_loops
               else list(_active_tasks.keys())[-1] if _active_tasks else "")
        if rid:
            await _stop_run(rid)
        return {"ok": True}

    @app.post("/api/approval", dependencies=[Depends(auth)])
    async def api_approval(req: ApprovalRequestModel) -> dict[str, Any]:
        if req.decision not in _APPROVE_DECISIONS | {"deny"}:
            raise HTTPException(status_code=422, detail="Unknown approval decision")
        payload = {"decision": req.decision, "userGuidance": req.user_guidance or ""}
        try:
            if _run_snapshots.replay(req.id, req.response_id, payload):
                return {"ok": True}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        future = _pending_approvals.get(req.id)
        if future is None or future.done():
            raise HTTPException(status_code=410, detail="Approval has already been answered or closed")
        try:
            _run_snapshots.accept(req.id, req.response_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        future.set_result(payload)
        return {"ok": True}

    @app.post("/api/question", dependencies=[Depends(auth)])
    async def api_question(req: QuestionRequestModel) -> dict[str, Any]:
        payload = {"answer": req.answer, "selectedIndex": req.selected_index}
        try:
            if _run_snapshots.replay(req.id, req.response_id, payload):
                return {"ok": True}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        pending = _pending_questions.get(req.id)
        if pending is None or pending.future.done():
            raise HTTPException(status_code=410, detail="Question has already been answered or closed")
        try:
            response = user_response(pending.params, req.answer, req.selected_index)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            _run_snapshots.accept(req.id, req.response_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        pending.future.set_result(response)
        return {"ok": True}

    # SSE streaming endpoint (legacy /stream/{request_id})

    @app.get("/stream/{request_id}", dependencies=[Depends(auth)])
    async def stream(request_id: str) -> StreamingResponse:
        task = _active_tasks.get(request_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        async def _generate() -> AsyncGenerator[str]:
            while not task.done():
                yield f"data: {json_encode({'status': 'running'})}\n\n"
                await asyncio.sleep(1.0)
            result = (
                task.result() if not task.cancelled() else "cancelled"
            )
            yield (
                f"data: {json_encode({'status': 'completed', 'result': str(result)})}\n\n"
            )

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Session management

    @app.get("/sessions", dependencies=[Depends(auth)])
    @app.get("/api/v1/sessions", dependencies=[Depends(auth)])
    async def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
        from rune.agent.session import SessionManager

        return SessionManager().list_sessions(limit=limit)

    @app.get("/sessions/{session_id}", dependencies=[Depends(auth)])
    @app.get("/api/v1/sessions/{session_id}", dependencies=[Depends(auth)])
    async def get_session(session_id: str) -> dict[str, Any]:
        from rune.agent.session import SessionManager

        entries = SessionManager().load_session(session_id)
        if not entries:
            raise HTTPException(
                status_code=404, detail="Session not found"
            )
        return {
            "session_id": session_id,
            "entries": [
                {
                    "type": e.type,
                    "content": e.content,
                    "timestamp": e.timestamp.isoformat(),
                    "metadata": e.metadata,
                }
                for e in entries
            ],
        }

    # RPC endpoint - dispatches to handler functions

    from rune.api.protocol import ApiRequest as _RpcRequest

    @app.post("/api/v1/rpc", dependencies=[Depends(auth)])
    async def rpc_dispatch(req: _RpcRequest) -> dict[str, Any]:  # type: ignore[type-arg]
        """Unified RPC endpoint for web UI.

        Accepts ``{method, params}`` and dispatches to the appropriate
        handler function, returning ``{success, data, error, timestamp}``.
        """
        method = req.method
        params = req.params or {}
        ts = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()

        def _ok(data: Any) -> dict[str, Any]:
            # Pydantic models → dict with camelCase aliases
            if hasattr(data, "model_dump"):
                data = data.model_dump(by_alias=True)
            return {"success": True, "data": data, "timestamp": ts}

        def _err(code: str, message: str) -> dict[str, Any]:
            return {
                "success": False,
                "error": {"code": code, "message": message},
                "timestamp": ts,
            }

        try:
            # sessions — canonical conversation store, same id space as
            # /sessions, /load and the live-chat sessionId.
            if method == "sessions.list":
                from rune.api import conversation_wiring

                convs = await conversation_wiring.list_web_conversations(
                    limit=params.get("limit", 20),
                )
                manager = conversation_wiring.get_conv_manager()
                sessions = []
                for c in convs:
                    turn_count = 0
                    if manager is not None:
                        try:
                            turn_count = await manager._store.get_turn_count(c.id)
                        except Exception:
                            turn_count = 0
                    sessions.append({
                        "id": c.id,
                        "userId": c.user_id,
                        "title": c.title or "",
                        "status": c.status,
                        "channel": "web",
                        "turnCount": turn_count,
                        "createdAt": c.created_at.isoformat(),
                        "updatedAt": c.updated_at.isoformat(),
                    })
                return _ok({"sessions": sessions, "total": len(sessions)})

            elif method == "sessions.turns":
                from rune.api import conversation_wiring

                manager = conversation_wiring.get_conv_manager()
                conv = None
                if manager is not None:
                    conv = await manager._store.load(params.get("sessionId", ""))
                if conv is None:
                    return _err("not_found", "Session not found")
                return _ok({
                    "turns": [
                        {
                            "role": t.role,
                            "content": t.content,
                            "timestamp": t.timestamp.isoformat(),
                        }
                        for t in conv.turns
                        if t.role in ("user", "assistant")
                    ],
                    "run": _run_snapshots.latest(conv.id),
                })

            elif method == "sessions.events":
                # Legacy placeholder (superseded by sessions.turns)
                params.get("sessionId", "")
                return _ok({"events": [], "runs": []})

            # workspace — a directory pinned per conversation; the agent runs
            # there and the app's file/diff views read from it.
            elif method == "workspace.get":
                from rune.api import conversation_wiring

                ws = await conversation_wiring.get_workspace(
                    params.get("sessionId", ""),
                )
                return _ok({"path": ws or ""})

            elif method == "workspace.set":
                from rune.api import conversation_wiring

                try:
                    resolved = await conversation_wiring.set_workspace(
                        params.get("sessionId", ""), params.get("path", ""),
                    )
                except ValueError as exc:
                    return _err("invalid_path", str(exc))
                return _ok({"path": resolved})

            elif method == "workspace.recents":
                from rune.api import conversation_wiring

                recents = await conversation_wiring.recent_workspaces()
                default_ws = str(Path.cwd())
                if default_ws not in recents:
                    recents.append(default_ws)
                return _ok({"paths": recents})

            elif method == "workspace.listdirs":
                # Subdirectories under `dir` (default home), for the folder
                # picker's type-ahead. Local single-user daemon: listing the
                # user's own filesystem to choose a project folder is expected.
                raw = params.get("dir", "") or "~"
                _skip_dirs = {"__pycache__", "node_modules", ".git"}

                def _scan(raw_dir: str) -> tuple[str, str, list[str]]:
                    # ALL filesystem work (is_dir/resolve/scandir) runs here,
                    # off the event loop, so a hung mount (`/Volumes/stale-nfs`)
                    # can't freeze the daemon. Early-break at the cap bounds a
                    # 100k-entry dir.
                    base = Path(raw_dir).expanduser()
                    if not base.is_dir():
                        base = base.parent if base.parent.is_dir() else Path.home()
                    base = base.resolve()
                    names: list[str] = []
                    with os.scandir(base) as it:
                        for de in it:
                            name = de.name
                            if name.startswith(".") or name in _skip_dirs:
                                continue
                            try:
                                # Follow symlinks so symlinked project dirs
                                # (~/dev, /tmp on macOS) still show — safe here
                                # because we're off the event loop.
                                if de.is_dir():
                                    names.append(name)
                            except OSError:
                                continue
                            if len(names) >= 1000:
                                break
                    names.sort(key=str.lower)
                    parent = str(base.parent) if base.parent != base else ""
                    return str(base), parent, names[:500]

                try:
                    d, parent, entries = await asyncio.to_thread(_scan, raw)
                except (OSError, ValueError) as exc:
                    return _err("read_failed", str(exc)[:150])
                return _ok({"dir": d, "parent": parent, "entries": entries})

            elif method == "workspace.diff":
                from rune.api import command_actions, conversation_wiring

                ws = await conversation_wiring.get_workspace(
                    params.get("sessionId", ""),
                ) or str(Path.cwd())
                ctx = command_actions.ActionContext(
                    broadcast=_broadcast, workspace=Path(ws),
                )
                text = await command_actions.execute_action(
                    "toggle_git_diff", ctx,
                )
                return _ok({"diff": text})

            elif method == "escalation.status":
                # Powers the trust card's "retry on a stronger model" ladder:
                # is a stronger model configured, which one, and does using it
                # send data off the machine (cloud) vs stay local.
                from rune.config import get_config as _gc

                _lcfg = _gc().llm
                _prov = _lcfg.escalation_provider or ""
                _model = _lcfg.escalation_model or ""
                if _prov and not _model:
                    with suppress(Exception):
                        from rune.llm.client import get_llm_client
                        from rune.types import ModelTier, Provider

                        _model = get_llm_client().resolve_model(
                            ModelTier.BEST, Provider(_prov),
                        )
                # When unconfigured, suggest the strongest INSTALLED local model
                # that clears the current model's tier — a single-jump local
                # candidate the user can accept in one click (no auto-run, no
                # multi-rung ladder; see escalation-ladder-research).
                _suggestion = ""
                if not _prov:
                    with suppress(Exception):
                        from rune.agent.advisor.tiers import (
                            suggest_local_escalation,
                        )

                        _suggestion = await suggest_local_escalation(
                            _lcfg.active_provider or "ollama",
                            _lcfg.active_model or "",
                        ) or ""
                return _ok({
                    "enabled": bool(_prov),
                    "provider": _prov,
                    "model": _model,
                    # ollama is the only local provider; everything else leaves
                    # the machine.
                    "isCloud": bool(_prov) and _prov != "ollama",
                    # Local single-jump candidate when nothing is configured.
                    "suggestion": _suggestion,
                })

            elif method == "escalation.set":
                # Accept a suggested (or user-chosen) escalation model for this
                # session — the "click to use this local model" path. Sets the
                # in-memory config that resolve/escalate read; the user can
                # still change it in Settings.
                from rune.config import get_config as _gc

                _prov = str(params.get("provider", "")).strip()
                _model = str(params.get("model", "")).strip()
                if not _prov:
                    return _err("invalid", "provider required")
                _lcfg = _gc().llm
                _lcfg.escalation_provider = _prov
                _lcfg.escalation_model = _model or None
                return _ok({"provider": _prov, "model": _model})

            elif method == "terminal.status":
                from rune.api import terminal as term

                return _ok({"enabled": term.is_enabled()})

            elif method == "terminal.token":
                from rune.api import conversation_wiring
                from rune.api import terminal as term

                if not term.is_enabled():
                    return _err(
                        "disabled",
                        "Terminal is off. Enable with RUNE_TERMINAL_ENABLED=1.",
                    )
                ws = await conversation_wiring.get_workspace(
                    params.get("sessionId", ""),
                ) or str(Path.cwd())
                return _ok({"token": term.mint_token(ws), "workspace": ws})

            elif method == "files.read":
                from rune.api import conversation_wiring

                ws = await conversation_wiring.get_workspace(
                    params.get("sessionId", ""),
                ) or str(Path.cwd())
                rel = params.get("path", "")

                def _read(ws_dir: str, rel_path: str) -> tuple[str, str]:
                    # (code, payload) — all blocking fs work off the event loop.
                    try:
                        root = Path(ws_dir).resolve()
                        target = (root / rel_path).resolve()
                    except ValueError:
                        # e.g. embedded null byte in the path.
                        return ("forbidden", "Invalid path")
                    # Jail: resolved target must sit inside the resolved
                    # workspace (defeats .., absolute paths, and symlinks —
                    # resolve() follows links before the prefix check).
                    if target != root and not str(target).startswith(
                        str(root) + os.sep
                    ):
                        return ("forbidden", "Path escapes the workspace")
                    if not target.is_file():
                        return ("not_found", f"No such file: {rel_path}")
                    if target.stat().st_size > 512_000:
                        return ("too_large", "File exceeds 500KB view limit")
                    try:
                        return ("ok", target.read_text(encoding="utf-8", errors="replace"))
                    except OSError as exc:
                        return ("read_failed", str(exc)[:150])

                code, payload = await asyncio.to_thread(_read, ws, rel)
                if code != "ok":
                    return _err(code, payload)
                return _ok({"path": rel, "content": payload})

            # skills
            elif method == "skills.list":
                from rune.api.handlers.skills import list_skills
                result = await list_skills(scope=params.get("scope"))
                return _ok(result)

            elif method == "skills.get":
                from rune.api.handlers.skills import get_skill
                result = await get_skill(skill_name=params.get("name", ""))
                return _ok(result)

            elif method == "skills.create":
                from rune.api.handlers.skills import SkillCreateRequest, create_skill
                result = await create_skill(SkillCreateRequest(
                    name=params.get("name") or "",
                    description=params.get("description", "") or "",
                    body=params.get("body", "") or "",
                    scope=params.get("scope") if params.get("scope") in ("user", "project") else "user",
                ))
                return _ok(result)

            elif method == "skills.update":
                from rune.api.handlers.skills import SkillUpdateRequest, update_skill
                result = await update_skill(
                    params.get("name") or "",
                    SkillUpdateRequest(
                        description=params.get("description"),
                        body=params.get("body"),
                    ),
                )
                return _ok(result)

            elif method == "skills.delete":
                from rune.api.handlers.skills import delete_skill
                result = await delete_skill(params.get("name") or "")
                return _ok(result)

            # -- env --
            elif method == "env.list":
                from rune.api.handlers.env import list_env
                result = await list_env(scope=params.get("scope"))
                return _ok(result)

            elif method == "env.set":
                from rune.api.handlers.env import EnvSetRequest, set_env
                result = await set_env(
                    key=params["key"],
                    req=EnvSetRequest(
                        value=params["value"],
                        scope=params.get("scope", "project"),
                    ),
                )
                return _ok(result)

            elif method == "env.unset":
                from rune.api.handlers.env import delete_env
                result = await delete_env(
                    key=params["key"],
                    scope=params.get("scope", "project"),
                )
                return _ok(result)

            # config
            elif method == "config.get":
                from rune.api.handlers.config import get_config_endpoint
                result = await get_config_endpoint()
                return _ok(result)

            elif method == "config.patch":
                from rune.api.handlers.config import ConfigPatchRequest, patch_config
                result = await patch_config(ConfigPatchRequest(**params))
                return _ok(result)

            # cron
            elif method == "cron.list":
                from rune.api.handlers.cron import list_cron_jobs
                result = await list_cron_jobs()
                return _ok(result)

            elif method == "cron.create":
                from rune.api.handlers.cron import CronCreateRequest, create_cron_job
                result = await create_cron_job(CronCreateRequest(**params))
                return _ok(result)

            elif method == "cron.update":
                from rune.api.handlers.cron import CronUpdateRequest, update_cron_job
                job_id = params.pop("id", params.pop("jobId", ""))
                result = await update_cron_job(job_id, CronUpdateRequest(**params))
                return _ok(result)

            elif method == "cron.delete":
                from rune.api.handlers.cron import delete_cron_job
                job_id = params.get("id", params.get("jobId", ""))
                result = await delete_cron_job(job_id)
                return _ok(result)

            # health
            elif method == "health":
                from rune.api.handlers.health import health
                result = await health()
                return _ok(result)

            # channels
            elif method == "channels.list":
                from rune.api.handlers.channels import list_channels
                result = await list_channels()
                return _ok(result)

            elif method == "channels.restart":
                # Stop and start the adapter for real, so the Settings button
                # reports what actually happened.
                from rune.channels.registry import get_channel_registry

                _cname = str(params.get("name", "")).strip()
                _adapter = get_channel_registry().get(_cname) if _cname else None
                if _adapter is None:
                    return _err("not_found", f"No such channel: {_cname or '(unnamed)'}")
                try:
                    with contextlib.suppress(Exception):
                        await _adapter.stop()
                    await _adapter.start()
                except Exception as exc:
                    log.warning("channel_restart_failed", name=_cname, error=str(exc))
                    return _err("restart_failed", f"{_cname}: {exc}")
                return _ok({"restarted": True, "name": _cname})

            # mcp
            elif method == "mcp.list":
                from rune.api.handlers.mcp import list_mcp_servers
                result = await list_mcp_servers()
                return _ok(result.model_dump())

            elif method == "mcp.add":
                from rune.api.handlers.mcp import MCPServerRequest, add_mcp_server
                result = await add_mcp_server(MCPServerRequest(**params))
                return _ok(result.model_dump())

            elif method == "mcp.update":
                from rune.api.handlers.mcp import MCPServerRequest, update_mcp_server
                original_name = params.pop("originalName", params.get("name", ""))
                result = await update_mcp_server(original_name, MCPServerRequest(**params))
                return _ok(result.model_dump())

            elif method == "mcp.delete":
                from rune.api.handlers.mcp import delete_mcp_server
                result = await delete_mcp_server(params["name"])
                return _ok(result)

            elif method == "mcp.test":
                from rune.api.handlers.mcp import test_mcp_server
                result = await test_mcp_server(params["name"])
                return _ok(result.model_dump())

            elif method == "commands.list":
                from rune.api.command_actions import WEB_UNSUPPORTED_COMMANDS
                from rune.slash_commands import COMMANDS
                return _ok([
                    {
                        "name": c.name,
                        "description": c.description,
                        "usage": c.usage or "",
                        "aliases": c.aliases,
                    }
                    for c in COMMANDS.values()
                    # Commands whose action only the TUI implements would answer
                    # "not available here"; the palette should not offer them.
                    if not c.hidden and c.name.lstrip("/") not in WEB_UNSUPPORTED_COMMANDS
                ])

            elif method == "runs.active":
                # Which runs are still going. A client whose event stream
                # dropped mid-run has no other way to learn the run ended, and
                # would sit on "running" until the page was reloaded.
                return _ok({
                    "runIds": sorted(set(_active_loops) | set(_active_tasks)),
                })

            elif method == "models.list":
                from rune.llm.client import prime_ollama_installed
                from rune.llm.models import known_models

                # Fill the local-model cache off the loop first; known_models
                # only reads it, so probing here would stall every other
                # request (SSE heartbeats, a run's text_delta) on localhost.
                await prime_ollama_installed()
                providers: dict[str, list[str]] = {}
                for prov, model in known_models():
                    providers.setdefault(prov, []).append(model)
                return _ok(providers)

            elif method == "model.set":
                from rune.llm.model_selection import (
                    ActiveModelSelection,
                    persist_active_model_selection,
                )

                _prov = str(params.get("provider", "")).strip()
                _model = str(params.get("model", "")).strip()
                if not _prov or not _model:
                    return _err("invalid", "provider and model required")
                # An unknown provider would be written to config.yaml and then
                # blow up Provider() on the next start, with the bad value on disk.
                from rune.types import Provider as _Provider
                try:
                    _Provider(_prov)
                except ValueError:
                    _known = ", ".join(sorted(p.value for p in _Provider))
                    return _err("invalid", f"unknown provider '{_prov}'; known: {_known}")
                try:
                    persist_active_model_selection(
                        ActiveModelSelection(provider=_Provider(_prov), model=_model), update_default=True,
                    )
                except OSError as exc:
                    return _err("storage", str(exc))
                return _ok({"provider": _prov, "model": _model})

            elif method == "reasoning.set":
                from rune.api.handlers.config import set_reasoning_effort
                try:
                    return _ok(await set_reasoning_effort(
                        params.get("effort", ""), provider=params.get("provider", ""), model=params.get("model", ""),
                    ))
                except ValueError as exc:
                    return _err("invalid", str(exc))
                except OSError as exc:
                    return _err("storage", str(exc))

            # Markdown file editor (HEARTBEAT.md, MEMORY.md, learned.md, user-profile.md)
            elif method == "markdown.list":
                from rune.utils.paths import rune_home
                _rh = rune_home()
                _files = {
                    "heartbeat": {"path": str(_rh / "HEARTBEAT.md"), "label": "Heartbeat", "description": "Periodic monitoring checklist"},
                    "memory": {"path": str(_rh / "memory" / "MEMORY.md"), "label": "Memory", "description": "Your knowledge — edit freely"},
                    "learned": {"path": str(_rh / "memory" / "learned.md"), "label": "Learned", "description": "Auto-extracted facts & rules"},
                    "profile": {"path": str(_rh / "memory" / "user-profile.md"), "label": "Profile", "description": "Your preferences"},
                }
                result = []
                for key, info in _files.items():
                    p = Path(info["path"])
                    result.append({
                        "key": key,
                        "label": info["label"],
                        "description": info["description"],
                        "exists": p.exists(),
                        "size": p.stat().st_size if p.exists() else 0,
                    })
                return _ok(result)

            elif method == "markdown.read":
                _key = params.get("key", "")
                _content = _read_markdown_file(_key)
                if _content is None:
                    return _err("NOT_FOUND", f"File not found: {_key}")
                return _ok({"key": _key, "content": _content})

            elif method == "markdown.write":
                _key = params.get("key", "")
                _content = params.get("content", "")
                _ok_write = _write_markdown_file(_key, _content)
                if not _ok_write:
                    return _err("WRITE_FAILED", f"Cannot write: {_key}")
                return _ok({"key": _key, "saved": True})

            else:
                return _err("METHOD_NOT_FOUND", f"Unknown method: {method}")

        except HTTPException as exc:
            return _err("HTTP_ERROR", exc.detail)
        except Exception as exc:
            log.error("rpc_dispatch_error", method=method, error=str(exc))
            return _err("INTERNAL_ERROR", f"Internal error: {type(exc).__name__}")

    # Markdown file helpers
    def _resolve_markdown_path(key: str) -> Path | None:
        from rune.utils.paths import rune_home
        _rh = rune_home()
        _map = {
            "heartbeat": _rh / "HEARTBEAT.md",
            "memory": _rh / "memory" / "MEMORY.md",
            "learned": _rh / "memory" / "learned.md",
            "profile": _rh / "memory" / "user-profile.md",
        }
        return _map.get(key)

    def _read_markdown_file(key: str) -> str | None:
        path = _resolve_markdown_path(key)
        if path is None or not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_markdown_file(key: str, content: str) -> bool:
        path = _resolve_markdown_path(key)
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return True
        except OSError:
            return False

    # Static file serving for Web UI (SPA)

    _web_static_dir = os.environ.get("RUNE_WEB_STATIC_DIR", "")
    if _web_static_dir:
        _static_path = Path(_web_static_dir)
        if _static_path.is_dir() and (_static_path / "index.html").is_file():
            from starlette.responses import FileResponse

            _index_html = _static_path / "index.html"

            # Catch-all for SPA routing - must come after all API routes
            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str) -> Any:
                # Try to serve static file first
                file = _static_path / full_path
                if full_path and file.is_file():
                    return FileResponse(file)
                # Fallback to index.html for SPA client-side routing
                return FileResponse(_index_html)

            log.info("web_static_mounted", path=str(_static_path))

    return app
