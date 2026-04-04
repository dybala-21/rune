"""AgentLoopController for RUNE TUI.

Replaces the TS useAgentLoop hook. Single bridge between NativeAgentLoop
events and RuneApp/Renderer UI updates. Wires ALL loop events:

  step, text_delta, tool_call, tool_result,
  status_change, completed, error, goal_classified

Also handles blocking interactions (approval, question, credential).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any

from rune.ui.approval_selection import ApprovalDecision
from rune.ui.controllers.delayed_commit import DelayedCommitController
from rune.ui.controllers.file_tracker import FileTracker
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.agent.loop import NativeAgentLoop
    from rune.ui.app import RuneApp
    from rune.ui.renderer import Renderer

log = get_logger(__name__)


# Tool category mapping (TS RUNE capability to display category)

_TOOL_CAT_MAP: dict[str, str] = {
    "file_read": "read", "file_list": "read", "file_search": "read",
    "project_map": "read", "code_search": "read", "code_symbols": "read",
    "file_write": "write", "file_edit": "write", "file_delete": "write",
    "bash": "exec", "shell": "exec", "exec": "exec", "bash_execute": "exec",
    "browser_navigate": "browse", "browser_click": "browse",
    "web_search": "web", "web_fetch": "web",
    "think": "think",
    "ask_user": "interact",
    "memory_search": "memory", "memory_store": "memory",
    "delegate": "delegate", "delegate_task": "delegate",
    "delegate_orchestrate": "delegate",
}


def _tool_category(name: str) -> str:
    """Map tool name to TS RUNE display category."""
    return _TOOL_CAT_MAP.get(name, "default")


class AgentLoopController:
    """Orchestrates the agent loop and emits UI events to RuneApp.

    This controller is the SINGLE bridge between the agent execution
    back-end and the Rich + prompt_toolkit front-end, mirroring TS useAgentLoop.
    """

    def __init__(self, app: RuneApp, loop: NativeAgentLoop) -> None:
        self._app = app
        self._loop = loop
        self._renderer: Renderer = app.renderer
        self._task: asyncio.Task[None] | None = None
        self._cancelled = False
        self._start_time: float = 0.0
        self._step_count: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._tool_count: int = 0
        self._failed_tool_count: int = 0
        self._last_tool_target: str = ""
        self._files_modified: list[str] = []
        self._streaming_buffer: list[str] = []

        # File tracker for undo/snapshot support
        self._file_tracker: FileTracker | None = None

        # Delayed commit controller (TS pattern: buffer text, flush on tool/complete)
        self._delayed = DelayedCommitController(
            on_flush=self._flush_text,
            delay_ms=500,
        )

        # --- Conversation manager for multi-turn context ---
        self._conv_manager = None
        self._conversation_id = ""
        try:
            import os
            import tempfile

            from rune.conversation.manager import ConversationManager
            from rune.conversation.store import ConversationStore
            db_path = os.path.join(tempfile.gettempdir(), "rune_tui_conversations.db")
            conv_store = ConversationStore(db_path)
            self._conv_manager = ConversationManager(conv_store)
            conv = self._conv_manager.start_conversation(user_id="tui:local")
            self._conversation_id = conv.id
        except Exception as exc:
            log.debug("tui_conv_manager_init_failed", error=str(exc)[:200])

        # Wire all events from the agent loop
        self._wire_events()

        # Proactive suggestion state
        self._proactive_task: asyncio.Task[None] | None = None
        self._proactive_started = False
        self._proactive_queue: list[Any] = []  # pending suggestions to display
        self._proactive_shown_titles: set[str] = set()  # dedup within session
        self._proactive_session_count = 0  # total shown this session
        self._proactive_last_shown: float = 0.0  # monotonic timestamp
        self._proactive_cooldown_until: float = 0.0  # monotonic timestamp
        self._last_user_input_time: float = time.monotonic()
        self._pending_suggestion_context: str = ""  # injected into next goal
        self._pending_suggestion_source: str = ""  # source type of pending suggestion

    # Proactive suggestion system
    _PROACTIVE_SESSION_MAX = 5       # max suggestions per session
    _PROACTIVE_MIN_INTERVAL = 600.0  # 10 minutes between suggestions
    _PROACTIVE_TICK_INTERVAL = 60.0  # evaluation cycle
    _PROACTIVE_COOLDOWN_SECS = 1800.0  # 30 min cooldown after 2 dismissals
    _PROACTIVE_IDLE_THRESHOLD = 180.0  # 3 min idle before "stuck?" suggestion
    _PROACTIVE_POST_TASK_DELAY = 3.0   # seconds after task completion

    def _ensure_proactive_started(self) -> None:
        """Start proactive + cron loop (idempotent).

        Called eagerly from TUI app startup and also on first user
        interaction as a safety net.  Cron jobs need this loop to run
        even before the user sends any message.
        """
        if self._proactive_started:
            return
        self._proactive_started = True
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._proactive_task = loop.create_task(self._proactive_loop())
        except RuntimeError:
            pass

    async def _proactive_loop(self) -> None:
        """Background loop: proactive evaluation + cron execution every 60 seconds."""
        await asyncio.sleep(15)  # settle time

        while True:
            try:
                await self._proactive_evaluate()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("proactive_tick_failed", error=str(exc)[:200])

            # Background consolidation (LLM-based episode extraction)
            try:
                from rune.memory.consolidation import consolidate_recent
                await consolidate_recent(limit=2)
            except Exception:
                pass

            # Cron runs independently of proactive suppress conditions
            try:
                await self._check_cron_jobs()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("cron_tick_failed", error=str(exc)[:200])

            # Flush any queued results (cron or proactive) to TUI
            with contextlib.suppress(Exception):
                self._flush_proactive_queue()

            await asyncio.sleep(self._PROACTIVE_TICK_INTERVAL)

    async def _proactive_evaluate(self) -> None:
        """Run one proactive evaluation and queue suggestions."""
        # Suppress while agent is running
        if self._task is not None and not self._task.done():
            return

        # Suppress if session cap reached
        if self._proactive_session_count >= self._PROACTIVE_SESSION_MAX:
            return

        # Suppress if in cooldown (after consecutive dismissals)
        now = time.monotonic()
        if now < self._proactive_cooldown_until:
            return

        # Suppress if min interval not elapsed
        if now - self._proactive_last_shown < self._PROACTIVE_MIN_INTERVAL:
            return

        try:
            from dataclasses import asdict

            from rune.proactive.context import ContextGatherer
            from rune.proactive.engine import get_proactive_engine
            from rune.proactive.formatter import format_suggestion

            engine = get_proactive_engine()
            gatherer = ContextGatherer()
            awareness = await gatherer.gather()
            ctx = asdict(awareness)

            # Enrich with prediction + action data
            try:
                from rune.proactive.prediction.engine import get_prediction_engine
                pred = get_prediction_engine()
                recent_actions = getattr(pred, '_recent_actions', [])
                if recent_actions:
                    ctx["recent_actions"] = recent_actions[-20:]
                    ctx["error_count"] = sum(
                        1 for a in recent_actions[-20:]
                        if not a.get("success", True)
                    )
                    history = pred.behavior_predictor._history
                    if len(history) >= 2:
                        repeated = 0
                        last = history[-1]
                        for t in reversed(history[:-1]):
                            if t == last:
                                repeated += 1
                            else:
                                break
                        ctx["repeated_commands"] = repeated
            except Exception:
                pass

            # Add idle detection
            idle_secs = now - self._last_user_input_time
            if idle_secs >= self._PROACTIVE_IDLE_THRESHOLD:
                ctx["idle_seconds"] = idle_secs

            suggestions = await engine.evaluate(ctx)

            for s in suggestions:
                title_key = s.title.lower().strip()
                if title_key in self._proactive_shown_titles:
                    continue  # already shown this session
                fmt = format_suggestion(s)
                self._proactive_queue.append(fmt)

        except Exception as exc:
            log.debug("proactive_evaluate_failed", error=str(exc)[:200])

    async def _check_cron_jobs(self) -> None:
        """Check and execute cron jobs that match the current minute."""
        try:
            from datetime import datetime

            from rune.capabilities.cron import get_active_cron_jobs
            from rune.daemon.heartbeat import HeartbeatScheduler
            from rune.proactive.formatter import FormattedSuggestion

            now = datetime.now()
            active_jobs = get_active_cron_jobs()
            if active_jobs:
                log.debug("cron_check", count=len(active_jobs), minute=now.minute)
            for job in active_jobs:
                # Skip if already ran this minute (compare in UTC)
                if job.last_run_at:
                    try:
                        from datetime import datetime as _dt
                        last = _dt.fromisoformat(job.last_run_at.replace("Z", "+00:00"))
                        now_utc = _dt.now(UTC)
                        if last.minute == now_utc.minute and last.hour == now_utc.hour:
                            continue
                    except Exception:
                        pass

                if HeartbeatScheduler._matches_cron(job.schedule, now):
                    # Mark as running BEFORE execution to prevent duplicate triggers
                    try:
                        from rune.capabilities.cron import _get_store
                        _get_store().record_cron_run(job.id)
                    except Exception:
                        pass

                    async def _run_and_queue(j=job):
                        try:
                            log.info("cron_goal_executing", name=j.name, goal=j.goal[:80])
                            result = await self._execute_cron_goal(j)
                            log.info("cron_goal_result", name=j.name, has_result=bool(result))
                            if result:
                                fmt = FormattedSuggestion(
                                    body=f"🔔 [{j.name}]\n{result}",
                                    confidence=1.0,
                                    source="cron",
                                    intensity="intervene",
                                )
                                self._proactive_queue.append(fmt)
                                log.info("cron_result_queued", name=j.name, queue_size=len(self._proactive_queue))
                                self._flush_proactive_queue()
                        except Exception as exc:
                            log.warning("cron_run_failed", name=j.name, error=str(exc))

                    asyncio.create_task(_run_and_queue())
        except Exception as exc:
            log.debug("cron_check_failed", error=str(exc)[:200])

    async def _execute_cron_goal(self, job) -> str | None:
        """Execute a cron job and return the output text."""

        if job.goal:
            try:
                from rune.agent.loop import NativeAgentLoop
                from rune.types import AgentConfig

                cfg = AgentConfig(max_iterations=30, timeout_seconds=120)
                agent_loop = NativeAgentLoop(config=cfg)

                # Capture text output via event listener
                collected: list[str] = []
                agent_loop.on("text_delta", lambda delta: collected.append(delta))

                result = await asyncio.wait_for(agent_loop.run(job.goal), timeout=120)

                # Prefer collected text, fall back to trace reason
                text = "".join(collected).strip()
                if not text:
                    text = getattr(result, "answer", None) or getattr(result, "reason", str(result))
                return text
            except TimeoutError:
                return f"⏱ Timed out: {job.goal[:80]}"
            except Exception as exc:
                return f"❌ Failed: {str(exc)[:200]}"
        elif job.command:
            from rune.capabilities.bash import BashParams, bash_execute
            result = await bash_execute(BashParams(command=job.command))
            return result.output if result.success else f"❌ {result.error}"
        return None

    def _flush_proactive_queue(self) -> None:
        """Display queued proactive suggestions and store as pending context.

        Only shows ONE suggestion at a time (most recent/highest priority).
        The suggestion text is stored in _pending_suggestion_context so it
        can be injected into the next user goal for natural-language handling.
        """
        if not self._proactive_queue:
            return

        # Don't print proactive suggestions if user's agent is running,
        # BUT always allow cron results through.
        has_cron = any(getattr(f, "source", "") == "cron" for f in self._proactive_queue)
        if not has_cron and self._task is not None and not self._task.done():
            return

        now = time.monotonic()

        # Find the first valid suggestion
        shown_fmt = None
        remaining: list[Any] = []

        for fmt in self._proactive_queue:
            if shown_fmt is not None:
                remaining.append(fmt)
                continue

            is_cron = getattr(fmt, "source", "") == "cron"

            # Cron results bypass session cap and cooldown
            if not is_cron:
                if self._proactive_session_count >= self._PROACTIVE_SESSION_MAX:
                    break
                if now < self._proactive_cooldown_until:
                    remaining.append(fmt)
                    continue

            title_key = fmt.body.lower().strip()[:50]
            if not is_cron and title_key in self._proactive_shown_titles:
                continue

            shown_fmt = fmt

        self._proactive_queue = remaining

        if shown_fmt is not None:
            self._renderer.print_proactive(
                body=shown_fmt.body,
                intensity=shown_fmt.intensity,
            )
            title_key = shown_fmt.body.lower().strip()[:50]
            self._proactive_shown_titles.add(title_key)
            self._proactive_session_count += 1
            self._proactive_last_shown = now

            # Store as pending context for the next user input
            self._pending_suggestion_context = shown_fmt.raw_description or shown_fmt.body
            self._pending_suggestion_source = getattr(shown_fmt, "source", "")

            log.debug("proactive_shown",
                      session_total=self._proactive_session_count)

    def consume_pending_suggestion(self, user_input: str = "") -> str:
        """Return and clear the pending suggestion context.

        Called by start() to inject into the agent goal.
        If the user's response looks like a dismissal AND the suggestion
        came from commitment_tracking, resolve it in the DB so it
        doesn't keep reappearing.
        """
        ctx = self._pending_suggestion_context
        source = self._pending_suggestion_source
        self._pending_suggestion_context = ""
        self._pending_suggestion_source = ""

        # If user dismisses a commitment suggestion, resolve it
        if ctx and source == "commitment_tracking":
            lower = user_input.lower().strip()
            _DISMISS_WORDS = {"n", "no", "아니", "괜찮아", "됐어", "나중에", "패스", "skip"}
            if lower in _DISMISS_WORDS or not lower:
                try:
                    from rune.memory.manager import get_memory_manager
                    store = getattr(get_memory_manager(), "store", None)
                    if store and hasattr(store, "get_open_commitments"):
                        # Find and resolve the commitment that matches
                        for c in store.get_open_commitments(limit=5):
                            if c["text"][:50] in ctx:
                                store.resolve_commitment(c["id"])
                                log.debug("commitment_resolved_by_dismiss",
                                          id=c["id"], text=c["text"][:60])
                                break
                except Exception:
                    pass
                return ""  # Don't inject dismissed suggestion into goal

        return ctx

    # Event wiring - registers handlers on NativeAgentLoop
    def _wire_events(self) -> None:
        """Register handlers for ALL events emitted by NativeAgentLoop."""
        self._loop.on("step", self._on_step)
        self._loop.on("text_delta", self._on_text_delta)
        self._loop.on("tool_call", self._on_tool_call)
        self._loop.on("tool_result", self._on_tool_result)
        self._loop.on("status_change", self._on_status_change)
        self._loop.on("completed", self._on_completed)
        self._loop.on("error", self._on_error)
        self._loop.on("goal_classified", self._on_goal_classified)
        self._loop.on("step_tokens", self._on_step_tokens)

    def _unwire_events(self) -> None:
        """Remove all event handlers."""
        self._loop.off("step", self._on_step)
        self._loop.off("text_delta", self._on_text_delta)
        self._loop.off("tool_call", self._on_tool_call)
        self._loop.off("tool_result", self._on_tool_result)
        self._loop.off("status_change", self._on_status_change)
        self._loop.off("completed", self._on_completed)
        self._loop.off("error", self._on_error)
        self._loop.off("goal_classified", self._on_goal_classified)
        self._loop.off("step_tokens", self._on_step_tokens)

    # Orchestrator event bridge
    def wire_orchestrator(self, orchestrator: Any) -> None:
        """Wire orchestrator events to TUI display.

        Call this when an :class:`Orchestrator` instance is created so
        that multi-agent progress is visible in the terminal.
        """
        orchestrator.on("plan_ready", self._on_orch_plan_ready)
        orchestrator.on("progress", self._on_orch_progress)
        orchestrator.on("subtask_retry", self._on_orch_retry)
        orchestrator.on("completed", self._on_orch_completed)

    async def _on_orch_plan_ready(self, plan: Any) -> None:
        task_count = len(plan.tasks) if hasattr(plan, "tasks") else 0
        desc = getattr(plan, "description", "")
        self._renderer.print_orchestration_started(task_count, desc)

    async def _on_orch_progress(
        self, completed: int, total: int, task_id: str, success: bool,
        description: str = "", role: str = "",
    ) -> None:
        self._renderer.print_orchestration_task_progress(
            task_id, completed, total, success=success,
            description=description, role=role,
        )

    async def _on_orch_retry(
        self, task_id: str, failure_type: str, attempt: int, error: str,
    ) -> None:
        self._renderer.print_orchestration_task_retry(
            task_id, failure_type, attempt, error,
        )

    async def _on_orch_completed(self, result: Any) -> None:
        success = getattr(result, "success", False)
        duration_ms = getattr(result, "duration_ms", 0.0)
        results = getattr(result, "results", [])
        ok = sum(1 for r in results if getattr(r, "success", False))
        fail = len(results) - ok
        self._renderer.print_orchestration_completed(
            success=success, duration_ms=duration_ms,
            completed_count=ok, failed_count=fail,
        )


    # Public API
    async def start(self, goal: str) -> None:
        """Start the agent loop for a user goal.

        Runs in a background task so the UI remains responsive.
        """
        if self._task is not None and not self._task.done():
            log.warning("agent_loop_already_running")
            return

        # Track user input time for idle detection
        self._last_user_input_time = time.monotonic()

        # Lazily start proactive loop on first interaction
        self._ensure_proactive_started()

        # If there's a pending proactive suggestion, inject it as context
        # so the agent can interpret the user's response naturally.
        # e.g., suggestion: "커밋 안 한 게 있어요. 커밋할까요?"
        #        user: "어 해줘"
        #        -> goal becomes: "[RUNE suggested: 커밋 안 한 게 있어요...] 어 해줘"
        pending = self.consume_pending_suggestion(user_input=goal)
        if pending:
            goal = f"[RUNE이 방금 제안함: \"{pending}\"] {goal}"

        self._cancelled = False
        self._start_time = time.monotonic()
        self._step_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_count = 0
        self._failed_tool_count = 0
        self._files_modified = []
        self._streaming_buffer.clear()

        self._app.update_status(status="thinking", current_step=0)
        log.info("agent_loop_start", goal=goal[:120])

        self._task = asyncio.create_task(self._run_loop(goal))

    async def cancel(self) -> None:
        """Request cancellation of the running loop."""
        self._cancelled = True
        # Cancel via the loop's own mechanism
        await self._loop.cancel()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._app.update_status(status="idle")
        self._finish_streaming()
        log.info("agent_loop_cancelled")

    # Event handlers (called by NativeAgentLoop via EventEmitter)
    async def _on_step(self, step: int) -> None:
        """Handle step start - update step counter and status bar."""
        self._step_count = step
        self._app.update_status(
            status="thinking",
            current_step=step,
        )
        log.debug("agent_step", step=step)

    async def _on_step_tokens(
        self, step: int, step_tokens: int, used: int, total: int,
    ) -> None:
        """Update the status bar with current token usage."""
        self._renderer.update_status(tokens_used=used, token_budget=total)

    async def _on_text_delta(self, delta: str) -> None:
        """Handle streaming text delta - update streaming display."""
        self._streaming_buffer.append(delta)
        self._renderer.update_streaming("".join(self._streaming_buffer))

    async def _on_tool_call(self, info: dict[str, Any]) -> None:
        """Handle tool call start - show tool activity."""
        name = info.get("name", "tool")
        params = info.get("params", {})
        category = _tool_category(name)
        self._tool_count += 1

        # Flush any buffered streaming text before showing tool call
        self._delayed.handle_tool_call()
        self._finish_streaming()

        self._app.update_status(status="acting", status_text=name)

        # Extract target info (filename, command, etc.) for display
        from rune.ui.message_format import _extract_tool_target
        target = _extract_tool_target(name, params)

        # Print tool call to scrollback
        self._renderer.print_tool_call(name, category, target=target)
        self._last_tool_target = target  # Save for result display

        # Take file snapshot before write operations (for undo)
        if self._file_tracker and name in ("file_write", "file_edit"):
            file_path = params.get("path") or params.get("file_path", "")
            if file_path:
                from pathlib import Path
                with contextlib.suppress(Exception):
                    self._file_tracker._take_snapshot(Path(file_path))

        log.debug("tool_call_start", tool=name)

    async def _on_tool_result(self, info: dict[str, Any]) -> None:
        """Handle tool result - update tool status, track files."""
        name = info.get("name", "tool")
        success = info.get("success", True)
        duration_ms = info.get("duration_ms", 0.0)
        category = _tool_category(name)

        self._app.update_status(status="observing")
        if not success:
            self._failed_tool_count += 1

        # Track file modifications
        if name in ("file_write", "file_edit", "file_delete") and success:
            self._files_modified.append(name)

        # Reuse target from the matching tool_call
        target = getattr(self, "_last_tool_target", "")

        # Print tool result to scrollback
        self._renderer.print_tool_result(
            name, category, success=success, duration_ms=duration_ms, target=target,
        )

        if not success:
            log.debug("tool_call_failed", tool=name)

    async def _on_status_change(self, status: Any) -> None:
        """Handle agent status transition (thinking/acting/idle)."""
        status_str = str(status).lower()

        status_map = {
            "thinking": "thinking",
            "acting": "acting",
            "observing": "observing",
            "reflecting": "reflecting",
            "idle": "idle",
            "completed": "idle",
            "failed": "idle",
        }
        display = status_map.get(status_str, status_str)
        self._app.update_status(status=display, status_text=display.capitalize())
        log.debug("status_change", status=display)

    async def _on_completed(self, trace: Any) -> None:
        """Handle agent loop completion - show TS-style inline summary."""
        elapsed_ms = (time.monotonic() - self._start_time) * 1000

        # Flush any remaining streamed text
        self._delayed.handle_complete()
        self._finish_streaming()

        self._app.update_status(status="idle", current_step=self._step_count)

        # Extract data from trace
        total_tokens = getattr(trace, "total_tokens_used", 0) or 0
        success = getattr(trace, "success", True)
        reason = getattr(trace, "reason", "")
        evidence_score = getattr(trace, "evidence_score", 0.0)

        # Update token display
        if total_tokens > 0:
            self._app.update_status(tokens_used=total_tokens)

        # Build TS RUNE-style completion summary
        summary = self._build_completion_summary(
            success=success,
            reason=reason,
            steps=self._step_count,
            tool_count=self._tool_count,
            failed_tools=self._failed_tool_count,
            total_tokens=total_tokens,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            elapsed_ms=elapsed_ms,
            files_modified=self._files_modified,
            evidence_score=evidence_score,
            trace=trace,
        )
        self._renderer.print_completion_summary(summary)

        # Ring terminal bell (TS RUNE behavior)
        self._app._ring_bell()

        log.info(
            "agent_loop_complete",
            steps=self._step_count,
            duration_ms=elapsed_ms,
            reason=reason,
        )

        # Post-task proactive: flush after a short delay
        # (gives user a moment to read the result before suggestion appears)
        async def _post_task_flush() -> None:
            await asyncio.sleep(self._PROACTIVE_POST_TASK_DELAY)
            self._flush_proactive_queue()

        with contextlib.suppress(RuntimeError):
            asyncio.create_task(_post_task_flush())

    async def _on_error(self, exc: Any) -> None:
        """Handle agent loop error - show error message."""
        self._finish_streaming()
        self._app.update_status(status="idle")

        error_msg = str(exc) if exc else "Unknown error"
        self._renderer.print_system_message(f"Error: {error_msg}")
        self._app.push_toast(f"Agent error: {error_msg}", toast_type="error")

        log.error("agent_loop_error", error=error_msg)

    async def _on_goal_classified(self, classification: Any) -> None:
        """Handle goal classification - update phase tracker."""
        goal_type = getattr(classification, "goal_type", "")
        confidence = getattr(classification, "confidence", 0.0)

        phase_map = {
            "chat": "explore",
            "research": "explore",
            "web": "explore",
            "code_modify": "implement",
            "execution": "implement",
            "full": "implement",
        }
        phase = phase_map.get(goal_type, "explore")
        self._app.update_phase(phase)

        tier = getattr(classification, "tier", 1)
        log.info("goal_classified_ui", type=goal_type, confidence=confidence, tier=tier)

    # Blocking interaction handlers (called by agent loop)

    async def _async_input(self, prompt: str = "> ") -> str:
        """Run input() in a thread so the asyncio event loop isn't blocked."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))

    def _pause_live(self) -> None:
        """Pause Rich Live and flush streaming before interactive prompt."""
        self._finish_streaming()

    async def request_approval(self, command: str, risk: str = "medium") -> ApprovalDecision:
        """Show approval prompt and return decision.

        Uses resolve_approval_from_input() for consistent parsing,
        runs input in executor to avoid blocking the event loop.
        """
        from rune.ui.approval_selection import (
            DEFAULT_APPROVAL_SELECTION_INDEX,
            ApprovalInputNeedsPrompt,
            resolve_approval_from_input,
        )

        log.info("approval_request", command=command, risk=risk)

        # Pause live display so prompt doesn't collide with spinner
        self._pause_live()

        # Show approval card
        self._app.show_pending_input("approval", title="Approval required", headline=command)

        risk_color = {"low": "#98C379", "medium": "#E5C07B", "high": "#E06C75", "critical": "#E06C75"}.get(risk, "#E5C07B")
        self._app.console.print()
        self._app.console.print("  [bold #D4A017]Approval required[/bold #D4A017]")
        self._app.console.print(f"  [#CCCCCC]{command}[/#CCCCCC]")
        self._app.console.print(f"  [{risk_color}]Risk: {risk}[/{risk_color}]")
        self._app.console.print()
        self._app.console.print("  [#888888]  (y) approve once   (a) approve always   (n) deny[/#888888]")

        try:
            response = await self._async_input("  > ")
            response = response.strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        finally:
            self._app.hide_pending_input()

        result = resolve_approval_from_input(response, DEFAULT_APPROVAL_SELECTION_INDEX)

        if result is None:
            # Invalid input -> default deny
            self._app.console.print("  [dim]Invalid input — denied.[/dim]")
            return ApprovalDecision.DENY

        if isinstance(result, ApprovalInputNeedsPrompt):
            # "Deny with instructions" - collect instructions
            self._app.console.print("  [dim]Enter instructions for the agent:[/dim]")
            try:
                instructions = await self._async_input("  > ")
            except (EOFError, KeyboardInterrupt):
                instructions = ""
            if instructions.strip():
                self._app.console.print(f"  [dim]Denied with: {instructions.strip()[:80]}[/dim]")
            return ApprovalDecision.DENY

        decision = result.decision
        labels = {
            ApprovalDecision.APPROVE_ONCE: "[#98C379]Approved (once)[/#98C379]",
            ApprovalDecision.APPROVE_ALWAYS: "[#98C379]Approved (always)[/#98C379]",
            ApprovalDecision.DENY: "[#E06C75]Denied[/#E06C75]",
        }
        self._app.console.print(f"  {labels.get(decision, '[dim]Denied[/dim]')}")
        return decision

    async def request_question(
        self,
        question: str,
        *,
        options: list[str] | None = None,
        urgency: str = "clarify",
    ) -> str:
        """Show question prompt with proper option rendering.

        Uses render_question_options() for formatted display,
        runs input in executor to avoid blocking the event loop.
        """
        from rune.ui.question_selection import (
            QuestionOption,
            is_custom_question_selection,
            render_question_options,
        )

        log.info("question_request", question=question[:80])

        # Pause live display
        self._pause_live()

        # Show pending focus
        self._app.show_pending_input("question", title="Agent question", headline=question)

        urgency_label = {"blocking": "Blocking", "clarifying": "Clarifying", "confirming": "Confirming"}.get(urgency, "")

        self._app.console.print()
        self._app.console.print(f"  [bold #D4A017]Agent question[/bold #D4A017]  [dim]{urgency_label}[/dim]")
        self._app.console.print(f"  [#CCCCCC]{question}[/#CCCCCC]")

        # Render options using the proper module
        q_options: list[QuestionOption] | None = None
        if options:
            q_options = [QuestionOption(label=opt) for opt in options]
            rendered = render_question_options(q_options, selected_index=0)
            self._app.console.print()
            for line in rendered:
                self._app.console.print(f"  [#AAAAAA]{line}[/#AAAAAA]")

        try:
            response = await self._async_input("  > ")
            response = response.strip()
        except (EOFError, KeyboardInterrupt):
            response = ""
        finally:
            self._app.hide_pending_input()

        # Parse numeric response to option text
        if options and response.isdigit():
            idx = int(response) - 1
            if q_options and is_custom_question_selection(idx, q_options):
                # Custom response selected - ask for free text
                self._app.console.print("  [dim]Type your answer:[/dim]")
                try:
                    response = await self._async_input("  > ")
                    response = response.strip()
                except (EOFError, KeyboardInterrupt):
                    response = ""
            elif 0 <= idx < len(options):
                response = options[idx]

        return response

    async def request_credential(
        self,
        label: str,
        *,
        help_url: str = "",
    ) -> str | None:
        """Show credential prompt with masked input."""
        import getpass

        log.info("credential_request", label=label)

        self._pause_live()
        self._app.show_pending_input("credential", title="Credential required", headline=label)

        self._app.console.print()
        self._app.console.print("  [bold #D4A017]Credential required[/bold #D4A017]")
        self._app.console.print(f"  [#CCCCCC]{label}[/#CCCCCC]")
        if help_url:
            self._app.console.print(f"  [dim]Help: {help_url}[/dim]")

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: getpass.getpass("  > "))
        except (EOFError, KeyboardInterrupt):
            response = ""
        finally:
            self._app.hide_pending_input()

        return response or None

    # Internal
    def _flush_text(self, text: str) -> None:
        """Commit buffered text to scrollback as markdown."""
        if text.strip():
            self._renderer.print_assistant_response(text)
            self._app._last_response_text = text

    def _finish_streaming(self) -> None:
        """Finish any active streaming display.

        Only commits text to scrollback if the streaming buffer is
        non-empty.  Calling this on an already-finished stream is a
        no-op, preventing empty response boxes from appearing.
        """
        if not self._streaming_buffer:
            return
        self._renderer.finish_streaming()
        raw = self._renderer.get_streaming_text()
        if raw:
            self._app._last_response_text = raw
        self._streaming_buffer.clear()

    async def _run_loop(self, goal: str) -> None:
        """Run the actual agent loop - delegates to NativeAgentLoop.run()."""
        try:
            # Record user turn for multi-turn context
            if self._conv_manager and self._conversation_id:
                with contextlib.suppress(Exception):
                    self._conv_manager.add_turn(self._conversation_id, "user", goal)

            # Session context: sanitize goal, resolve workspace, etc.
            ctx = None
            effective_goal = goal
            run_context: dict[str, Any] | None = None
            try:
                from rune.agent.agent_context import (
                    PrepareContextOptions,
                    prepare_agent_context,
                )
                ctx = await prepare_agent_context(
                    PrepareContextOptions(
                        goal=goal,
                        channel="tui",
                        conversation_id=self._conversation_id,
                    ),
                    conversation_manager=self._conv_manager,
                )
                effective_goal = ctx.goal
                run_context = {"workspace_root": ctx.workspace_root}
                # Build memory context for self-improving (same as server/gateway)
                try:
                    from rune.memory.manager import get_memory_manager
                    mgr = get_memory_manager()
                    mem_ctx = await mgr.build_memory_context(goal)
                    if mem_ctx:
                        run_context["memory_context"] = mem_ctx
                except Exception:
                    pass  # Memory context is best-effort
            except Exception as exc:
                log.debug("prepare_context_skipped", error=str(exc)[:200])

            # Pass conversation history to the agent loop
            message_history = None
            if ctx is not None and ctx.messages:
                message_history = ctx.messages

            trace = await self._loop.run(
                effective_goal,
                context=run_context,
                message_history=message_history,
            )

            # If streaming was active but completed event didn't fire cleanly,
            # ensure we finish streaming
            self._finish_streaming()

            # If no streamed text was shown, display the final text
            if not self._streaming_buffer:
                final_text = getattr(trace, "final_text", None)
                if final_text:
                    self._renderer.print_assistant_response(final_text)
                    self._app._last_response_text = final_text
                elif not getattr(trace, "success", True):
                    error = getattr(trace, "error", "") or getattr(trace, "reason", "")
                    if error:
                        self._renderer.print_system_message(f"Completed with issues: {error}")

            # Record assistant turn for multi-turn context
            answer = self._app._last_response_text or ""
            if self._conv_manager and self._conversation_id and answer:
                with contextlib.suppress(Exception):
                    self._conv_manager.add_turn(
                        self._conversation_id, "assistant", answer,
                        goal_type=getattr(self._loop, "_last_goal_type", ""),
                    )

            # Post-process: save assistant turn + episodic memory
            if ctx is not None:
                try:
                    from rune.agent.agent_context import (
                        PostProcessInput,
                        post_process_agent_result,
                    )
                    await post_process_agent_result(PostProcessInput(
                        context=ctx,
                        success=getattr(trace, "success", True),
                        answer=answer,
                    ))
                except Exception as exc:
                    log.debug("post_process_skipped", error=str(exc)[:200])

        except asyncio.CancelledError:
            log.info("agent_loop_task_cancelled")
            self._finish_streaming()
            raise
        except Exception as exc:
            log.exception("agent_loop_run_error")
            self._finish_streaming()
            self._renderer.print_system_message(f"Agent loop error: {exc}")
            self._app.update_status(status="idle")

    # Completion summary (TS RUNE format)
    @staticmethod
    def _fmt_tokens(n: int) -> str:
        """Format token count: 483000 -> '483k', 1900 -> '1.9k', 50 -> '50'."""
        if n >= 100_000:
            return f"{n // 1000}k"
        if n >= 1_000:
            v = n / 1000
            return f"{v:.1f}k" if v < 100 else f"{int(v)}k"
        return str(n)

    @staticmethod
    def _fmt_elapsed(ms: float) -> str:
        """Format elapsed: 189000 -> '3m 9s', 23000 -> '23s'."""
        secs = int(ms / 1000)
        if secs < 60:
            return f"{secs}s"
        mins = secs // 60
        remaining = secs % 60
        if mins < 60:
            return f"{mins}m {remaining}s"
        hours = mins // 60
        remaining_m = mins % 60
        return f"{hours}h {remaining_m}m"

    @staticmethod
    def _fmt_cost(model: str, input_tokens: int, output_tokens: int) -> str:
        """Estimate cost from token counts."""
        prices: dict[str, tuple[float, float]] = {
            "gpt-5.4-pro": (10.0, 40.0),
            "gpt-5.4": (2.50, 10.0),
            "gpt-5-mini": (0.30, 1.20),
            "gpt-5-nano": (0.10, 0.40),
            "gpt-5.3-codex": (2.50, 10.0),
            "gpt-4o": (2.50, 10.0),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4.1": (2.00, 8.00),
            "o4-mini": (1.10, 4.40),
            "o3-mini": (1.10, 4.40),
            "claude-sonnet": (3.0, 15.0),
            "claude-haiku": (0.25, 1.25),
            "claude-opus": (15.0, 75.0),
        }
        in_price, out_price = 2.50, 10.0
        model_lower = model.lower()
        for key, (ip, op) in prices.items():
            if key in model_lower:
                in_price, out_price = ip, op
                break
        cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
        if cost < 0.001:
            return "<$0.001"
        return f"~${cost:.2f}"

    def _build_completion_summary(
        self,
        *,
        success: bool,
        reason: str,
        steps: int,
        tool_count: int,
        failed_tools: int,
        total_tokens: int,
        input_tokens: int,
        output_tokens: int,
        elapsed_ms: float,
        files_modified: list[str],
        evidence_score: float,
        trace: Any,
    ) -> str:
        """Build TS RUNE-style completion summary text.

        Format:
          ✓ I wrapped up after 23 tool calls and 5 file updates.
          steps 9 - tools 23 - tokens 483k - cost ~$0.79 - time 3m 9s
          input 386k - output 1.9k
          evidence △ PARTIAL (13/14 requirements)
              reads:19  exec:3
        """
        sep = " \u2014 "  # em dash
        lines: list[str] = []

        # Line 1: Narrative
        file_count = len(set(files_modified))
        if success:
            icon = "[bold #98C379]\u2713[/bold #98C379]"
            verb = "I wrapped up"
        else:
            icon = "[bold #E06C75]\u2717[/bold #E06C75]"
            verb = "I hit a stopping point"

        parts = []
        if tool_count > 0:
            parts.append(f"{tool_count} tool call{'s' if tool_count != 1 else ''}")
            if failed_tools > 0:
                parts[-1] += f", with {failed_tools} failure{'s' if failed_tools != 1 else ''}"
        if file_count > 0:
            parts.append(f"{file_count} file update{'s' if file_count != 1 else ''}")

        narrative = f"{icon} {verb} after {' and '.join(parts)}." if parts else f"{icon} {verb}."
        lines.append(narrative)

        # Line 2: Metrics
        metrics: list[str] = []
        metrics.append(f"[bold]steps {steps}[/bold]")

        tool_str = f"tools {tool_count}"
        if failed_tools > 0:
            tool_str += f" [#E06C75]({failed_tools} failed)[/#E06C75]"
        metrics.append(tool_str)

        if total_tokens > 0:
            metrics.append(f"tokens {self._fmt_tokens(total_tokens)}")
        elif input_tokens > 0 or output_tokens > 0:
            metrics.append(f"tokens {self._fmt_tokens(input_tokens + output_tokens)}")

        model = self._app._model or "gpt-5.4"
        if input_tokens > 0 or output_tokens > 0:
            cost_str = self._fmt_cost(model, input_tokens, output_tokens)
            metrics.append(f"cost {cost_str}")
        elif total_tokens > 0:
            est_in = int(total_tokens * 0.65)
            est_out = total_tokens - est_in
            cost_str = self._fmt_cost(model, est_in, est_out)
            metrics.append(f"cost {cost_str}")

        metrics.append(f"time {self._fmt_elapsed(elapsed_ms)}")

        lines.append(f"[#56B6C2]{sep.join(metrics)}[/#56B6C2]")

        # Line 3: Token breakdown
        if input_tokens > 0 or output_tokens > 0:
            lines.append(
                f"[#555555]input {self._fmt_tokens(input_tokens)}"
                f"{sep}output {self._fmt_tokens(output_tokens)}[/#555555]"
            )

        # Line 4: Evidence
        outcome = getattr(trace, "outcome", None)
        evidence = getattr(trace, "evidence", None)
        if outcome:
            outcome_str = str(outcome).upper()
            if outcome == "verified":
                icon_e = "[bold #98C379]\u2713[/bold #98C379]"
                color = "#98C379"
            elif outcome == "partial":
                icon_e = "[bold #E5C07B]\u25B3[/bold #E5C07B]"
                color = "#E5C07B"
            else:
                icon_e = "[bold #E06C75]\u2717[/bold #E06C75]"
                color = "#E06C75"

            req_parts = ""
            missing = getattr(trace, "missing_requirement_ids", [])
            reqs = getattr(trace, "requirements", [])
            if reqs:
                total_reqs = len(reqs)
                met = total_reqs - len(missing)
                req_parts = f" ({met}/{total_reqs} requirements)"

            lines.append(f"evidence {icon_e} [{color}]{outcome_str}[/{color}]{req_parts}")

            if evidence:
                ev_parts: list[str] = []
                reads = getattr(evidence, "reads", 0)
                writes = getattr(evidence, "writes", 0)
                executions = getattr(evidence, "executions", 0)
                verifications = getattr(evidence, "verifications", 0)
                if reads:
                    ev_parts.append(f"reads:{reads}")
                if writes:
                    ev_parts.append(f"writes:{writes}")
                if executions:
                    ev_parts.append(f"exec:{executions}")
                if verifications:
                    ev_parts.append(f"verify:{verifications}")
                if ev_parts:
                    lines.append(f"[#555555]    {'  '.join(ev_parts)}[/#555555]")

        return "\n".join(lines)

    def set_file_tracker(self, tracker: FileTracker) -> None:
        """Attach a FileTracker for undo/snapshot support."""
        self._file_tracker = tracker
        self._app.set_file_tracker(tracker)
