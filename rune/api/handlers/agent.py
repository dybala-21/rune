"""Submit, inspect and cancel background agent runs."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.agent.run_control import RunControl, control_scope
from rune.agent.run_outcome import run_outcome
from rune.api.auth import TokenAuthDependency
from rune.api.run_tracker import RunResult, RunTracker
from rune.safety.approval_context import approval_required
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])
auth = TokenAuthDependency()

MAX_CONCURRENT_API_RUNS = 3

# Shared tracker - in production this would be injected via app state.
_tracker = RunTracker()
_background_tasks: set[asyncio.Task[Any]] = set()


def get_tracker() -> RunTracker:
    return _tracker


# Request / Response models


class AgentRunRequest(BaseModel):
    goal: str
    session_id: str | None = Field(None, alias="sessionId")
    cwd: str | None = None
    sender_name: str | None = Field(None, alias="senderName")

    model_config = ConfigDict(populate_by_name=True)


class AgentRunResponse(BaseModel):
    run_id: str = Field(alias="runId")
    session_id: str = Field(alias="sessionId")

    model_config = ConfigDict(populate_by_name=True)


class AgentStatusResponse(BaseModel):
    run_id: str = Field(alias="runId")
    status: str
    answer: str | None = None
    error: str | None = None
    success: bool | None = None
    outcome: dict[str, Any] | None = None
    elapsed_ms: int | None = Field(None, alias="elapsedMs")

    model_config = ConfigDict(populate_by_name=True)


class AgentCancelResponse(BaseModel):
    run_id: str = Field(alias="runId")
    cancelled: bool


# Routes


@router.post("/run", response_model=AgentRunResponse, dependencies=[Depends(auth)])
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """Start a background run; poll its status or subscribe to SSE for progress."""
    tracker = get_tracker()

    if tracker.get_active_count() >= MAX_CONCURRENT_API_RUNS:
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent agent runs. Try again later.",
        )

    run_id = uuid.uuid4().hex
    session_id = req.session_id or uuid.uuid4().hex

    tracker.create(run_id, client_id="api", session_id=session_id, goal=req.goal)

    # Pass the resolved session ID so the first turn is saved under the ID returned to the client.
    task = asyncio.create_task(_execute_agent(tracker, run_id, req, session_id), name=f"api-run:{run_id}")
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return AgentRunResponse(runId=run_id, sessionId=session_id)


@router.get("/status", response_model=AgentStatusResponse, dependencies=[Depends(auth)])
async def agent_status(run_id: str = "") -> AgentStatusResponse:
    """Get the current status of an agent run."""
    if not run_id:
        raise HTTPException(status_code=400, detail="runId is required")

    tracker = get_tracker()
    run = tracker.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    import time

    elapsed_ms = int(
        ((run.completed_at or time.time()) - run.started_at) * 1000
    )

    return AgentStatusResponse(
        runId=run.run_id,
        status=run.status,
        answer=run.result.answer if run.result else None,
        error=run.error,
        success=run.result.success if run.result else None,
        outcome=run.result.outcome if run.result else None,
        elapsedMs=elapsed_ms,
    )


@router.post("/cancel", dependencies=[Depends(auth)])
async def agent_cancel(run_id: str = "") -> AgentCancelResponse:
    """Cancel a running agent execution."""
    if not run_id:
        raise HTTPException(status_code=400, detail="runId is required")

    tracker = get_tracker()
    success = tracker.abort(run_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found or already completed: {run_id}",
        )

    for task in _background_tasks:
        if task.get_name() == f"api-run:{run_id}":
            task.cancel()

    return AgentCancelResponse(runId=run_id, cancelled=True)


# Background execution


async def _execute_agent(
    tracker: RunTracker, run_id: str, req: AgentRunRequest, session_id: str = "",
) -> None:
    """Execute the agent in the background, updating the tracker."""
    tracker.mark_running(run_id)
    control = RunControl(run_id)
    try:
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
            resolve_assistant_answer,
        )
        from rune.agent.loop import NativeAgentLoop

        # Load conversation manager for multi-turn context
        conv_manager = None
        session_id = session_id or req.session_id or ""
        if session_id:
            try:

                from rune.conversation.manager import ConversationManager
                from rune.conversation.store import ConversationStore
                from rune.utils.paths import conversations_db_path

                conv_store = ConversationStore(conversations_db_path())
                conv_manager = ConversationManager(conv_store)

                # Try to find or create the conversation for this session
                existing = await conv_store.load(session_id)
                if existing is not None:
                    conv_manager._active[session_id] = existing
                else:
                    from rune.conversation.types import Conversation
                    conv = Conversation(id=session_id, user_id="api")
                    conv_manager._active[session_id] = conv
            except Exception as exc:
                log.debug("api_conv_manager_init_failed", error=str(exc)[:100])

        # Record user turn
        if conv_manager and session_id:
            with contextlib.suppress(Exception):
                conv_manager.add_turn(session_id, "user", req.goal)

        # 1. Prepare agent context
        agent_ctx = await prepare_agent_context(
            PrepareContextOptions(
                goal=req.goal,
                channel="api",
                cwd=req.cwd or "",
                sender_name=req.sender_name or "",
                conversation_id=session_id,
            ),
            conversation_manager=conv_manager,
        )

        loop = NativeAgentLoop()

        blocked: list[str] = []

        async def _api_approval_cb(command: str, reason: str) -> bool:
            blocked.append(command)
            return False

        loop.set_approval_callback(_api_approval_cb)

        from rune.capabilities.ask_user import AskUserParams, UserResponse, user_response

        async def _api_ask_user_cb(
            params: AskUserParams,
        ) -> UserResponse:
            log.info("api_ask_user_autonomous", run_id=run_id, question=params.question[:100])
            return user_response(params, "")

        loop.set_ask_user_callback(_api_ask_user_cb)

        # 3. Collect streamed text for answer quality
        collected: list[str] = []

        async def _collect_text(delta: str) -> None:
            collected.append(delta)

        loop.on("text_delta", _collect_text)

        # 4. Run with context + conversation history
        context_dict: dict[str, Any] = {}
        if agent_ctx.workspace_root:
            context_dict["workspace_root"] = agent_ctx.workspace_root

        with control_scope(control), approval_required():
            trace = await loop.run(
                agent_ctx.goal,
                context=context_dict if context_dict else None,
                message_history=agent_ctx.messages if agent_ctx.messages else None,
            )
        outcome = run_outcome(trace)
        needs_approval = bool(blocked) and not outcome.verified
        success = outcome.success and not needs_approval
        result_outcome = outcome.payload()
        if needs_approval:
            result_outcome.update(completionStatus="incomplete", reason="approval_required",
                                  blockedTools=sorted(set(blocked)))
        # Prefer the loop's final answer; retain a nonempty fallback so the next turn has context.
        answer = resolve_assistant_answer(
            getattr(loop, "_last_answer_text", ""), "".join(collected),
        ) or (trace.reason or "completed")

        # Record assistant turn
        if conv_manager and session_id and answer:
            try:
                conv_manager.add_turn(
                    session_id, "assistant", answer,
                    goal_type=getattr(loop, "_last_goal_type", ""),
                )
                await conv_manager._store.save(conv_manager._active[session_id])
            except Exception as exc:
                log.debug("api_conv_save_failed", error=str(exc)[:100])

        # 5. Post-process (memory persistence)
        try:
            await post_process_agent_result(PostProcessInput(
                classification_hint=getattr(loop, "_last_goal_type", "") or None,
                verification=getattr(trace, "verification", None),
                reason=getattr(trace, "reason", ""),
                mech_check=getattr(trace, "mech_check", ""),
                evidence_gate=getattr(trace, "evidence_gate", None),
                context=agent_ctx,
                success=success,
                answer=answer,
            ))
        except Exception as exc:
            log.warning("api_post_process_failed", run_id=run_id, error=str(exc)[:100])

        tracker.mark_completed(
            run_id,
            RunResult(
                success=success,
                answer=answer,
                outcome=result_outcome,
            ),
        )
    except asyncio.CancelledError:
        tracker.mark_aborted(run_id)
    except Exception as exc:
        tracker.mark_failed(run_id, str(exc))
        log.error("agent_run_failed", run_id=run_id, error=str(exc))
    finally:
        control.stop()
