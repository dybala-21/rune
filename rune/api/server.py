"""FastAPI server for RUNE.

Ported from src/api/server.ts - REST + SSE + WebSocket API with token-based
auth, CORS, streaming agent execution, and session management.

Supports three real-time protocols:
- SSE (GET /api/v1/events) - Server→Client event stream
- NDJSON (POST /api/v1/agent/execute) - Streaming execution
- WebSocket (/ws) - Bidirectional real-time communication
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)


# SSE Client Manager


class SseClientManager:
    """Manages connected SSE clients with heartbeat and broadcasting."""

    def __init__(self) -> None:
        self._clients: dict[str, asyncio.Queue[str]] = {}
        self._event_counter = 0

    def add_client(self, client_id: str) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
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
        for queue in self._clients.values():
            try:
                queue.put_nowait(formatted)
            except asyncio.QueueFull:
                pass  # Drop events for slow clients

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
        for cid, ws in self._clients.items():
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
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "FastAPI and uvicorn are required for the API server. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    from rune.api.auth import TokenAuthDependency

    # Active tasks - declared early so the lifespan can reference them.
    _active_tasks: dict[str, asyncio.Task[Any]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[arg-type]
        log.info("api_server_started")
        yield
        for task in list(_active_tasks.values()):
            task.cancel()
        if _active_tasks:
            await asyncio.gather(
                *list(_active_tasks.values()), return_exceptions=True
            )
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

    class ApprovalRequestModel(BaseModel):
        id: str
        decision: str
        user_guidance: str | None = None

    class QuestionRequestModel(BaseModel):
        id: str
        answer: str
        selected_index: int | None = None

    # State

    _start_time = time.monotonic()
    _active_loops: dict[str, Any] = {}  # run_id -> NativeAgentLoop
    _sse_manager = SseClientManager()
    _ws_manager = WsClientManager()
    _pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}
    _pending_questions: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # Broadcast helper

    async def _broadcast(event: str, data: dict[str, Any]) -> None:
        """Broadcast to both SSE and WebSocket clients."""
        _sse_manager.broadcast(event, data)
        await _ws_manager.broadcast(event, data)

    # Agent execution helpers (inside create_app for closure access)

    async def _run_agent_for_client(
        goal: str,
        run_id: str,
        client_id: str | None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """Run the agent loop, broadcasting events to SSE/WS clients.

        Parameters
        ----------
        goal:
            The user's goal / prompt text.
        run_id:
            Unique identifier for this execution run.
        client_id:
            Optional client id to direct messages to.  When *None*,
            events are broadcast to all connected clients.
        attachments:
            Optional list of attachment dicts (name, mimeType, data).
        """
        try:
            from rune.agent.agent_context import (
                PostProcessInput,
                PrepareContextOptions,
                post_process_agent_result,
                prepare_agent_context,
            )
            from rune.agent.loop import NativeAgentLoop

            # 1. Prepare agent context
            agent_ctx = await prepare_agent_context(
                PrepareContextOptions(
                    goal=goal,
                    channel="web",
                    attachments=attachments or [],
                )
            )

            loop = NativeAgentLoop()
            _active_loops[run_id] = loop

            # 2. Wire approval callback (SSE/WS ↔ future)
            async def _web_approval_callback(command: str, risk_level: str) -> bool:
                approval_id = f"approval:{run_id}:{int(time.monotonic() * 1000)}"
                approval_future: asyncio.Future[dict[str, Any]] = (
                    asyncio.get_running_loop().create_future()
                )
                _pending_approvals[approval_id] = approval_future
                await _broadcast(
                    "approval_request",
                    {
                        "id": approval_id,
                        "command": command,
                        "riskLevel": risk_level,
                        "runId": run_id,
                    },
                )
                try:
                    result = await asyncio.wait_for(approval_future, timeout=120.0)
                    return result.get("decision") == "approve"
                except TimeoutError:
                    return False
                finally:
                    _pending_approvals.pop(approval_id, None)

            loop.set_approval_callback(_web_approval_callback)

            # 3. Wire ask_user callback (SSE/WS ↔ future)
            async def _web_ask_user_callback(
                question: str, options: list[str] | None = None
            ) -> str:
                question_id = f"question:{run_id}:{int(time.monotonic() * 1000)}"
                question_future: asyncio.Future[dict[str, Any]] = (
                    asyncio.get_running_loop().create_future()
                )
                _pending_questions[question_id] = question_future
                await _broadcast(
                    "question",
                    {
                        "id": question_id,
                        "question": question,
                        "options": options or [],
                        "runId": run_id,
                    },
                )
                try:
                    result = await asyncio.wait_for(question_future, timeout=300.0)
                    return result.get("answer", "")
                except TimeoutError:
                    return ""
                finally:
                    _pending_questions.pop(question_id, None)

            loop.set_ask_user_callback(_web_ask_user_callback)

            collected: list[str] = []

            # -- wire event callbacks ------------------------------------

            _run_start_time = time.monotonic()

            async def _on_step(step: int) -> None:
                await _broadcast(
                    "step_start",
                    {"stepNumber": step, "tokens": 0, "runId": run_id},
                )

            async def _on_text(delta: str) -> None:
                collected.append(delta)
                await _broadcast(
                    "text_delta",
                    {"text": "".join(collected), "runId": run_id},
                )

            async def _on_tool(info: dict[str, Any]) -> None:
                await _broadcast(
                    "tool_call",
                    {
                        "toolName": info.get("name", ""),
                        "args": info.get("params", {}),
                        "runId": run_id,
                    },
                )

            async def _on_tool_result(info: dict[str, Any]) -> None:
                await _broadcast(
                    "tool_result",
                    {
                        "toolName": info.get("name", ""),
                        "result": "",
                        "success": info.get("success", True),
                        "runId": run_id,
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
                {"runId": run_id, "goal": agent_ctx.goal},
            )

            # 4. Run with context
            trace = await loop.run(
                agent_ctx.goal,
                context={"workspace_root": agent_ctx.workspace_root},
            )
            answer = "".join(collected)
            duration_ms = int((time.monotonic() - _run_start_time) * 1000)

            # 5. Post-process (memory persistence)
            try:
                await post_process_agent_result(PostProcessInput(
                    context=agent_ctx,
                    success=trace.reason == "completed",
                    answer=answer,
                    duration_ms=duration_ms,
                ))
            except Exception as exc:
                log.warning("web_post_process_failed", error=str(exc)[:100])

            await _broadcast(
                "agent_complete",
                {
                    "runId": run_id,
                    "success": trace.reason == "completed",
                    "answer": answer,
                    "durationMs": duration_ms,
                },
            )

            return answer

        except Exception as exc:
            log.error("agent_execution_error", run_id=run_id, error=str(exc))
            await _broadcast(
                "agent_error",
                {"runId": run_id, "error": f"Agent execution failed: {type(exc).__name__}"},
            )
            return f"error: {type(exc).__name__}"
        finally:
            _active_loops.pop(run_id, None)

    async def _ndjson_execution(
        goal: str, run_id: str
    ) -> AsyncGenerator[str]:
        """NDJSON streaming generator for agent execution."""
        try:
            from rune.agent.agent_context import (
                PostProcessInput,
                PrepareContextOptions,
                post_process_agent_result,
                prepare_agent_context,
            )
            from rune.agent.loop import NativeAgentLoop

            # 1. Prepare agent context
            agent_ctx = await prepare_agent_context(
                PrepareContextOptions(goal=goal, channel="web")
            )

            yield (
                json_encode(
                    {
                        "event": "agent_start",
                        "data": {"runId": run_id, "goal": agent_ctx.goal},
                    }
                )
                + "\n"
            )

            loop = NativeAgentLoop()
            _active_loops[run_id] = loop
            collected: list[str] = []

            # Use a queue so event callbacks can feed the generator
            event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            # 2. NDJSON is unidirectional (server→client) - no way to receive
            #    approval/question responses. Use auto-approve + autonomous mode.
            async def _ndjson_approval_cb(command: str, risk_level: str) -> bool:
                await event_queue.put(
                    {
                        "event": "approval_request",
                        "data": {
                            "id": f"ndjson:{run_id}",
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
                question: str, options: list[str] | None = None
            ) -> str:
                await event_queue.put(
                    {
                        "event": "question",
                        "data": {
                            "id": f"ndjson:{run_id}",
                            "question": question,
                            "options": options or [],
                            "runId": run_id,
                            "autonomous": True,
                        },
                    }
                )
                return ""

            loop.set_ask_user_callback(_ndjson_ask_user_cb)

            async def _on_step(step: int) -> None:
                await event_queue.put(
                    {
                        "event": "step_start",
                        "data": {"stepNumber": step, "tokens": 0, "runId": run_id},
                    }
                )

            async def _on_text(delta: str) -> None:
                collected.append(delta)
                await event_queue.put(
                    {
                        "event": "text_delta",
                        "data": {"text": "".join(collected), "runId": run_id},
                    }
                )

            async def _on_tool(info: dict[str, Any]) -> None:
                await event_queue.put(
                    {
                        "event": "tool_call",
                        "data": {
                            "toolName": info.get("name", ""),
                            "args": info.get("params", {}),
                            "runId": run_id,
                        },
                    }
                )

            async def _on_tool_result(info: dict[str, Any]) -> None:
                await event_queue.put(
                    {
                        "event": "tool_result",
                        "data": {
                            "toolName": info.get("name", ""),
                            "result": "",
                            "success": info.get("success", True),
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
                    context={"workspace_root": agent_ctx.workspace_root},
                )
            )

            while not run_task.done():
                try:
                    evt = await asyncio.wait_for(
                        event_queue.get(), timeout=2.0
                    )
                    yield json_encode(evt) + "\n"
                except TimeoutError:
                    yield (
                        json_encode({"event": "heartbeat", "data": {}}) + "\n"
                    )

            # Drain any remaining events
            while not event_queue.empty():
                evt = event_queue.get_nowait()
                yield json_encode(evt) + "\n"

            trace = run_task.result()
            answer = "".join(collected)
            duration_ms = int((time.monotonic() - _run_start_time) * 1000)

            # Post-process (memory persistence)
            try:
                await post_process_agent_result(PostProcessInput(
                    context=agent_ctx,
                    success=trace.reason == "completed",
                    answer=answer,
                    duration_ms=duration_ms,
                ))
            except Exception as exc:
                log.warning("ndjson_post_process_failed", error=str(exc)[:100])

            yield (
                json_encode(
                    {
                        "event": "agent_complete",
                        "data": {
                            "runId": run_id,
                            "success": trace.reason == "completed",
                            "answer": answer,
                            "durationMs": duration_ms,
                        },
                    }
                )
                + "\n"
            )

        except Exception as exc:
            log.error("agent_stream_error", run_id=run_id, error=str(exc))
            yield (
                json_encode(
                    {
                        "event": "agent_error",
                        "data": {"runId": run_id, "error": f"Agent execution failed: {type(exc).__name__}"},
                    }
                )
                + "\n"
            )
        finally:
            _active_loops.pop(run_id, None)

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
                        ws_attachments = msg.get("attachments") or None
                        task = asyncio.create_task(
                            _run_agent_for_client(
                                goal=text,
                                run_id=run_id,
                                client_id=client_id,
                                attachments=ws_attachments,
                            )
                        )
                        _active_tasks[run_id] = task
                        task.add_done_callback(lambda _t, _rid=run_id: _active_tasks.pop(_rid, None))
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
                    agent_loop = _active_loops.get(run_id)
                    if agent_loop:
                        await agent_loop.cancel()

                elif msg_type == "approval":
                    aid = msg.get("id", "")
                    future = _pending_approvals.get(aid)
                    if future and not future.done():
                        future.set_result(
                            {
                                "decision": msg.get("decision", "deny"),
                                "userGuidance": msg.get(
                                    "userGuidance", ""
                                ),
                            }
                        )

                elif msg_type == "question":
                    qid = msg.get("id", "")
                    future = _pending_questions.get(qid)
                    if future and not future.done():
                        future.set_result(
                            {
                                "answer": msg.get("answer", ""),
                                "selectedIndex": msg.get("selectedIndex"),
                            }
                        )

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

        if req.stream:
            # NDJSON streaming response
            return StreamingResponse(
                _ndjson_execution(req.goal, run_id),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: run and broadcast events via SSE/WS
        task = asyncio.create_task(
            _run_agent_for_client(
                goal=req.goal, run_id=run_id, client_id=None
            )
        )
        _active_tasks[run_id] = task

        try:
            result = await task
        finally:
            _active_tasks.pop(run_id, None)

        return ExecuteResponse(
            request_id=run_id,
            status="completed",
            result=result,
        )

    # Legacy REST endpoints

    @app.post("/api/message", dependencies=[Depends(auth)])
    async def api_message(req: MessageRequest) -> dict[str, Any]:
        # Intercept slash commands — execute without agent
        if req.text.startswith("/"):
            from rune.ui.commands import COMMANDS, parse_slash_command
            parsed = parse_slash_command(req.text)
            if parsed:
                cmd_name, args = parsed
                cmd = COMMANDS.get(cmd_name)
                if cmd:
                    try:
                        result = await cmd.handler(args)
                        # __ACTION__ results are TUI-specific; return as info
                        output = result or f"{cmd_name} executed."
                        if isinstance(output, str) and output.startswith("__ACTION__:"):
                            output = f"{cmd_name}: {output.replace('__ACTION__:', '')}"
                    except Exception as exc:
                        output = f"Error: {exc}"
                    await _broadcast("command_result", {"command": cmd_name, "output": output})
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
        task = asyncio.create_task(
            _run_agent_for_client(
                goal=req.text, run_id=run_id, client_id=None,
                attachments=raw_attachments or None,
            )
        )
        _active_tasks[run_id] = task
        task.add_done_callback(lambda _t, _rid=run_id: _active_tasks.pop(_rid, None))
        return {"ok": True}

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
            agent_loop = _active_loops.get(run_id)
            if agent_loop:
                await agent_loop.cancel()
                return {"ok": True}
            task = _active_tasks.get(run_id)
            if task:
                task.cancel()
                return {"ok": True}

        # No specific runId - abort the most recent active loop
        if _active_loops:
            last_run_id = list(_active_loops.keys())[-1]
            await _active_loops[last_run_id].cancel()
            return {"ok": True}
        if _active_tasks:
            last_task_id = list(_active_tasks.keys())[-1]
            _active_tasks[last_task_id].cancel()
            return {"ok": True}

        return {"ok": True}

    @app.post("/api/approval", dependencies=[Depends(auth)])
    async def api_approval(req: ApprovalRequestModel) -> dict[str, Any]:
        future = _pending_approvals.get(req.id)
        if future and not future.done():
            future.set_result(
                {
                    "decision": req.decision,
                    "userGuidance": req.user_guidance or "",
                }
            )
        return {"ok": True}

    @app.post("/api/question", dependencies=[Depends(auth)])
    async def api_question(req: QuestionRequestModel) -> dict[str, Any]:
        future = _pending_questions.get(req.id)
        if future and not future.done():
            future.set_result(
                {
                    "answer": req.answer,
                    "selectedIndex": req.selected_index,
                }
            )
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
            # sessions
            if method == "sessions.list":
                from rune.api.handlers.sessions import list_sessions
                result = await list_sessions(
                    status=params.get("status"),
                    limit=params.get("limit", 20),
                    offset=params.get("offset", 0),
                )
                return _ok(result)

            elif method == "sessions.events":
                # Return event log for a session (placeholder)
                params.get("sessionId", "")
                return _ok({"events": [], "runs": []})

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
                return _ok({"name": params.get("name", ""), "filePath": ""})

            elif method == "skills.update":
                return _ok({"updated": True})

            elif method == "skills.delete":
                return _ok({"deleted": True})

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
                # Placeholder - channel restart not yet implemented
                return _ok({"restarted": True, "name": params.get("name", "")})

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
                from rune.ui.commands import COMMANDS
                return _ok([
                    {
                        "name": c.name,
                        "description": c.description,
                        "usage": c.usage or "",
                        "aliases": c.aliases,
                    }
                    for c in COMMANDS.values()
                    if not c.hidden
                ])

            elif method == "models.list":
                from rune.ui.app import _KNOWN_MODELS
                providers: dict[str, list[str]] = {}
                for prov, model in _KNOWN_MODELS:
                    providers.setdefault(prov, []).append(model)
                return _ok(providers)

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
