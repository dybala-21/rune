"""Main daemon server for RUNE.

Ported from src/daemon/daemon.ts - Unix domain socket server that
accepts commands, manages task execution, and orchestrates the agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rune.daemon.types import (
    CommandType,
    DaemonCommand,
    DaemonResponse,
    DaemonStatus,
)
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_DEFAULT_SOCKET_PATH = rune_home() / "daemon.sock"


@dataclass(slots=True)
class _ActiveTask:
    """Tracks a running task inside the daemon."""

    request_id: str
    goal: str
    sender_id: str
    task: asyncio.Task[Any]
    started_at: float = field(default_factory=time.monotonic)


class DaemonServer:
    """Unix domain socket server that manages RUNE agent tasks."""

    def __init__(
        self,
        socket_path: Path | str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._socket_path = Path(socket_path) if socket_path else _DEFAULT_SOCKET_PATH
        self._config = config or {}
        self._server: asyncio.Server | None = None
        self._start_time: float = 0.0
        self._active_tasks: dict[str, _ActiveTask] = {}
        self._task_queue: asyncio.Queue[DaemonCommand] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self._queue_worker: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._server is not None and self._server.is_serving()

    @property
    def uptime_seconds(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    async def start(self) -> None:
        """Create and start the Unix domain socket server."""
        # Remove stale socket file
        if self._socket_path.exists():
            self._socket_path.unlink()

        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )
        # Restrict socket permissions to owner only (prevent other local
        # users from sending commands to the daemon).
        with contextlib.suppress(OSError):
            self._socket_path.chmod(0o600)
        self._start_time = time.monotonic()
        self._shutdown_event.clear()

        # Start queue worker
        self._queue_worker = asyncio.create_task(self._process_queue())

        log.info("daemon_started", socket=str(self._socket_path))

    async def stop(self) -> None:
        """Gracefully shut down the daemon."""
        log.info("daemon_stopping")
        self._shutdown_event.set()

        # Cancel active tasks
        active_snapshot = list(self._active_tasks.values())
        for active in active_snapshot:
            active.task.cancel()
        if active_snapshot:
            await asyncio.gather(
                *(t.task for t in active_snapshot),
                return_exceptions=True,
            )
        self._active_tasks.clear()

        # Stop queue worker
        if self._queue_worker is not None:
            self._queue_worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._queue_worker
            self._queue_worker = None

        # Stop server
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Clean up socket file
        if self._socket_path.exists():
            self._socket_path.unlink()

        self._start_time = 0.0
        log.info("daemon_stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        peer = writer.get_extra_info("peername") or "unknown"
        log.debug("daemon_connection_opened", peer=peer)

        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break

                try:
                    data = json_decode(line.decode("utf-8"))
                    command = DaemonCommand(
                        type=CommandType(data["type"]),
                        payload=data.get("payload", {}),
                        request_id=data.get("request_id", ""),
                    )
                    response = await self._process_command(command)
                except (ValueError, KeyError) as exc:
                    response = DaemonResponse(
                        request_id="",
                        success=False,
                        error=f"Invalid command: {exc}",
                    )

                response_bytes = json_encode({
                    "request_id": response.request_id,
                    "success": response.success,
                    "data": response.data,
                    "error": response.error,
                }).encode("utf-8") + b"\n"

                writer.write(response_bytes)
                await writer.drain()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("daemon_connection_error", error=str(exc))
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            log.debug("daemon_connection_closed", peer=peer)

    async def _process_command(self, command: DaemonCommand) -> DaemonResponse:
        """Route a command to the appropriate handler."""
        match command.type:
            case CommandType.STATUS:
                return self._handle_status(command)
            case CommandType.EXECUTE:
                return await self._handle_execute(command)
            case CommandType.CANCEL:
                return self._handle_cancel(command)
            case CommandType.SHUTDOWN:
                return await self._handle_shutdown(command)
            case _:
                return DaemonResponse(
                    request_id=command.request_id,
                    success=False,
                    error=f"Unknown command type: {command.type}",
                )

    def _handle_status(self, command: DaemonCommand) -> DaemonResponse:
        """Return current daemon status."""
        status = DaemonStatus(
            running=self.running,
            uptime_seconds=self.uptime_seconds,
            active_tasks=len(self._active_tasks),
            queued_tasks=self._task_queue.qsize(),
            channels=[],  # Populated by gateway if connected
        )
        return DaemonResponse(
            request_id=command.request_id,
            success=True,
            data={
                "running": status.running,
                "uptime_seconds": status.uptime_seconds,
                "active_tasks": status.active_tasks,
                "queued_tasks": status.queued_tasks,
                "channels": status.channels,
            },
        )

    async def _handle_execute(self, command: DaemonCommand) -> DaemonResponse:
        """Start executing a goal."""
        goal = command.payload.get("goal", "")
        sender_id = command.payload.get("sender_id", "")

        if not goal:
            return DaemonResponse(
                request_id=command.request_id,
                success=False,
                error="No goal provided",
            )

        task = asyncio.create_task(
            self._execute_task(goal, sender_id, command.request_id)
        )
        self._active_tasks[command.request_id] = _ActiveTask(
            request_id=command.request_id,
            goal=goal,
            sender_id=sender_id,
            task=task,
        )

        log.info(
            "task_started",
            request_id=command.request_id,
            goal=goal[:100],
        )
        return DaemonResponse(
            request_id=command.request_id,
            success=True,
            data={"status": "started"},
        )

    def _handle_cancel(self, command: DaemonCommand) -> DaemonResponse:
        """Cancel a running task."""
        target_id = command.payload.get("request_id", "")
        active = self._active_tasks.get(target_id)

        if active is None:
            return DaemonResponse(
                request_id=command.request_id,
                success=False,
                error=f"Task {target_id} not found",
            )

        active.task.cancel()
        log.info("task_cancelled", request_id=target_id)
        return DaemonResponse(
            request_id=command.request_id,
            success=True,
            data={"cancelled": target_id},
        )

    async def _handle_shutdown(self, command: DaemonCommand) -> DaemonResponse:
        """Initiate graceful shutdown."""
        log.info("shutdown_requested")
        # Schedule stop in the background so we can still send the response
        asyncio.create_task(self.stop())
        return DaemonResponse(
            request_id=command.request_id,
            success=True,
            data={"status": "shutting_down"},
        )

    async def _execute_task(
        self, goal: str, sender_id: str, request_id: str
    ) -> None:
        """Execute a single agent task with full context preparation.

        Steps:
        1. Prepare agent context (workspace, identity, conversation)
        2. Build memory context for the goal
        3. Create and run the agent loop with context injection
        4. Post-process results (memory persistence)
        5. Report progress if clients are connected
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
            context_opts = PrepareContextOptions(
                goal=goal,
                channel="daemon",
                sender_id=sender_id,
            )
            agent_ctx = await prepare_agent_context(context_opts)

            # 2. Build memory context (best-effort)
            memory_context = ""
            try:
                from rune.memory.manager import get_memory_manager
                manager = get_memory_manager()
                memory_context = await manager.build_memory_context(goal)
            except Exception as exc:
                log.debug("memory_context_build_failed", error=str(exc)[:100])

            # 3. Run agent loop with context + callbacks
            loop = NativeAgentLoop()

            # Daemon is non-interactive - auto-approve safe/low/medium only.
            # High and critical risk commands are denied because no human is
            # present to review them.
            async def _daemon_approval_cb(command: str, risk_level: str) -> bool:
                if risk_level in ("high", "critical"):
                    log.warning(
                        "daemon_approval_denied",
                        command=command[:100],
                        risk_level=risk_level,
                        reason="non-interactive daemon cannot approve high-risk commands",
                    )
                    return False
                log.info(
                    "daemon_auto_approve",
                    command=command[:100],
                    risk_level=risk_level,
                )
                return True

            loop.set_approval_callback(_daemon_approval_cb)

            # Daemon ask_user: return empty (agent proceeds autonomously)
            async def _daemon_ask_user_cb(
                question: str, options: list[str] | None = None
            ) -> str:
                log.info("daemon_ask_user_autonomous", question=question[:100])
                return ""

            loop.set_ask_user_callback(_daemon_ask_user_cb)

            # Collect streamed text for answer quality
            collected: list[str] = []

            async def _collect_text(delta: str) -> None:
                collected.append(delta)

            loop.on("text_delta", _collect_text)

            context_dict: dict[str, Any] = {}
            if agent_ctx.workspace_root:
                context_dict["cwd"] = agent_ctx.workspace_root
            if memory_context:
                context_dict["memory_context"] = memory_context
            if agent_ctx.conversation_id:
                context_dict["conversation_id"] = agent_ctx.conversation_id
            if sender_id:
                context_dict["sender_id"] = sender_id

            trace = await loop.run(goal, context=context_dict if context_dict else None)

            # 4. Post-process results (memory persistence)
            try:
                answer = "".join(collected) if collected else (trace.reason or "")
                await post_process_agent_result(PostProcessInput(
                    context=agent_ctx,
                    success=trace.reason == "completed",
                    answer=answer,
                    duration_ms=0.0,
                ))
            except Exception as exc:
                log.warning("post_process_failed", error=str(exc)[:100])

            # 5. Emit task completion event to proactive engine
            try:
                from rune.proactive.engine import get_proactive_engine
                engine = get_proactive_engine()
                engine.emit_task_completed(goal, {
                    "success": trace.reason != "error",
                    "steps": trace.final_step,
                    "request_id": request_id,
                })
            except Exception:
                pass

            log.info(
                "task_completed",
                request_id=request_id,
                reason=trace.reason,
                steps=trace.final_step,
            )
        except asyncio.CancelledError:
            log.info("task_cancelled_during_execution", request_id=request_id)
        except Exception as exc:
            log.error(
                "task_execution_failed",
                request_id=request_id,
                error=str(exc),
            )
            # Emit task failure event
            try:
                from rune.proactive.engine import get_proactive_engine
                engine = get_proactive_engine()
                engine.emit_task_failed(goal, str(exc))
            except Exception:
                pass
        finally:
            self._active_tasks.pop(request_id, None)

    async def _process_queue(self) -> None:
        """Background worker that processes queued commands."""
        while not self._shutdown_event.is_set():
            try:
                command = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0
                )
                await self._process_command(command)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("queue_worker_error", error=str(exc))
