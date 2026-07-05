"""Server-side executors for slash-command ``__ACTION__:*`` markers.

Used by the web app (results flow to the client over SSE: ``command_result``
and ``goal_iteration``) and by the REPL (console broadcast, foreground mode).
Rendering-side actions (retry/copy/export/…) are handled by each surface
before reaching this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

Broadcast = Callable[[str, dict[str, Any]], Awaitable[None]]
RunAgent = Callable[..., Awaitable[str]]


@dataclass
class ActionContext:
    """Everything an action needs from the server, injected per call."""

    broadcast: Broadcast
    workspace: Path
    session_id: str | None = None
    # Runs a full agent turn on the live conversation; accepts (goal,
    # agent_config) so /escalate can override the model per run.
    run_agent: RunAgent | None = None
    active_run_count: Callable[[], int] = lambda: 0
    started_at: float = 0.0
    # Await long actions (/goal, /escalate) inline instead of spawning a
    # background task. SSE surfaces want the request back immediately; the
    # REPL blocks until done.
    foreground: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ── module state (single-user local daemon) ─────────────────────────────

_goal_task: asyncio.Task[None] | None = None
_file_tracker: Any | None = None
_background_tasks: set[asyncio.Task[Any]] = set()


def _reset_for_tests() -> None:
    global _goal_task, _file_tracker
    _goal_task = None
    _file_tracker = None


async def _get_file_tracker(workspace: Path) -> Any | None:
    """Lazily start the workspace file tracker backing /files and /undo."""
    global _file_tracker
    if _file_tracker is not None:
        return _file_tracker
    try:
        from rune.agent.file_tracker import FileTracker

        tracker = FileTracker(workspace=str(workspace))
        await tracker.start()
        _file_tracker = tracker
    except Exception as exc:
        log.debug("file_tracker_start_failed", error=str(exc)[:100])
    return _file_tracker


def _spawn(coro: Awaitable[None]) -> asyncio.Task[None]:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


# ── direct commands (no __ACTION__ marker; TUI handles these in-app) ────


async def handle_direct_command(cmd_name: str, args: str) -> str | None:
    """Commands the TUI intercepts before the registry: /help, /learning,
    /advisor, /model. Returns output text, or None if not a direct command."""
    if cmd_name == "/help":
        from rune.slash_commands import COMMANDS

        lines = ["Available commands:"]
        for cmd in COMMANDS.values():
            if cmd.hidden:
                continue
            usage = cmd.usage or cmd.name
            lines.append(f"  {usage} — {cmd.description}")
        return "\n".join(lines)

    if cmd_name == "/learning":
        from rune.memory.consolidation import (
            is_consolidation_enabled,
            set_learning_enabled,
        )

        sub = args.strip().lower()
        if sub in ("on", "off"):
            set_learning_enabled(sub == "on")
            return f"Learning {sub}."
        state = "on" if is_consolidation_enabled() else "off"
        return f"Learning is {state}. Use /learning on|off to change."

    if cmd_name == "/advisor":
        from rune.agent.advisor.runtime_toggle import (
            is_advisor_enabled,
            set_advisor_enabled,
        )

        sub = args.strip().lower()
        if sub in ("on", "off"):
            set_advisor_enabled(sub == "on")
            return f"Advisor {sub}."
        state = "on" if is_advisor_enabled() else "off"
        return f"Advisor is {state}. Use /advisor on|off to change."

    if cmd_name == "/model":
        return _handle_model_command(args)

    return None


def _handle_model_command(args: str) -> str:
    from rune.config import get_config

    cfg = get_config()
    spec = args.strip()
    if not spec:
        return (
            f"Current model: {cfg.llm.active_provider}:{cfg.llm.active_model}\n"
            "Usage: /model <provider:model> or /model <model>"
        )
    if ":" in spec:
        provider, model = spec.split(":", 1)
    else:
        provider, model = cfg.llm.active_provider or "", spec
    cfg.llm.active_provider = provider or cfg.llm.active_provider
    cfg.llm.active_model = model
    return f"Switched to {cfg.llm.active_provider}:{model} (new runs use this model)."


# ── action dispatch ──────────────────────────────────────────────────────


async def execute_action(action: str, ctx: ActionContext) -> str:
    """Execute an ``__ACTION__:`` payload server-side; returns output text.

    Long-running actions (/goal, /escalate) start background work and return
    an acknowledgement immediately; progress/results arrive over SSE.
    """
    try:
        if action.startswith("goal_loop:"):
            return await _action_goal_loop(action.split(":", 1)[1], ctx)
        if action.startswith("escalate"):
            task = action.split(":", 1)[1] if ":" in action else ""
            return await _action_escalate(task, ctx)
        if action == "toggle_git_diff":
            return await _action_git_diff(ctx)
        if action == "toggle_files":
            return await _action_files(ctx)
        if action == "undo":
            return await _action_undo(ctx)
        if action.startswith("memory:"):
            return await _action_memory(action.split(":", 1)[1], ctx)
        if action.startswith("search:"):
            return await _action_search(action.split(":", 1)[1], ctx)
        if action == "show_status":
            return _action_status(ctx)
        if action == "show_config":
            return _action_config()
        if action in ("show_sessions", "show_session"):
            return await _action_sessions(ctx)
        if action.startswith("load:") or action == "interactive_load":
            conv_id = action.split(":", 1)[1] if ":" in action else ""
            return await _action_load(conv_id, ctx)
        if action.startswith("save"):
            return (
                "Web conversations persist automatically"
                + (f" (session {ctx.session_id})." if ctx.session_id else ".")
            )
        # Rendering-side actions belong to the calling surface.
        return (
            f"'{action.split(':', 1)[0]}' is handled by the app/TUI interface "
            "and is not available here."
        )
    except Exception as exc:
        log.warning("command_action_failed", action=action[:40], error=str(exc)[:200])
        return f"Command failed: {type(exc).__name__}: {str(exc)[:200]}"


# ── /goal ────────────────────────────────────────────────────────────────


async def _action_goal_loop(request: str, ctx: ActionContext) -> str:
    global _goal_task
    from rune.config import get_config

    cfg = get_config().goal_loop
    if not cfg.enabled:
        return "/goal is disabled (config goal_loop.enabled=false)."
    if _goal_task is not None and not _goal_task.done():
        return "A goal loop is already running; wait for it to finish."

    from rune.agent.goal_spec import crystallize_goal

    cr = await crystallize_goal(request)
    if cr.ambiguous:
        qs = "\n".join(f"  - {q}" for q in cr.clarifications)
        return (
            "Goal is too vague to verify automatically. Refine and re-run /goal:\n"
            + qs
        )
    if (
        cr.spec.acceptance_criteria
        and not cr.spec.validation_commands
        and not cfg.adversarial_review
    ):
        return (
            "Goal has acceptance criteria but no way to verify them "
            "(no validation commands, adversarial review off). Refine /goal."
        )

    if ctx.foreground:
        await _run_goal_loop(cr.spec, ctx)
        return ""  # result already delivered via ctx.broadcast
    _goal_task = _spawn(_run_goal_loop(cr.spec, ctx))
    return f"Goal loop started: {cr.spec.goal[:120]} (progress streams below)"


async def _run_goal_loop(spec: Any, ctx: ActionContext) -> None:
    try:
        from rune.agent.goal_loop import GoalLoop, GoalLoopConfig
        from rune.agent.goal_review import (
            make_adversarial_review_fn,
            make_ssc_critique_fn,
        )
        from rune.agent.goal_runtime import GoalRuntime
        from rune.agent.goal_validate import make_validate_fn
        from rune.agent.loop import NativeAgentLoop
        from rune.config import get_config
        from rune.types import AgentConfig

        cfg = get_config().goal_loop
        workspace = ctx.workspace / ".rune" / "goal"
        workspace.mkdir(parents=True, exist_ok=True)

        runtime = GoalRuntime(
            NativeAgentLoop(
                config=AgentConfig(token_budget_override=cfg.inner_token_budget)
            ),
            channel="web",
        )

        main_loop = asyncio.get_running_loop()

        def _on_iteration(it: Any) -> None:
            payload = {
                "n": it.n,
                "verdict": str(getattr(it.verdict, "value", it.verdict)),
                "reason": it.reason,
                "evidence": it.evidence,
                "tokens": it.tokens,
            }
            main_loop.create_task(ctx.broadcast("goal_iteration", payload))

        kwargs: dict[str, Any] = {}
        if cfg.adversarial_review:
            kwargs["review_fn"] = make_adversarial_review_fn()
        if cfg.ssc_interval > 0:
            kwargs["critique_fn"] = make_ssc_critique_fn()
        if cfg.escalate_on_stuck and get_config().llm.escalation_provider:
            kwargs["escalate_fn"] = runtime.escalate_run_fn

        gl = GoalLoop(
            config=GoalLoopConfig(
                max_iterations=cfg.max_iterations,
                max_total_tokens=cfg.max_total_tokens,
                stagnation_window=cfg.stagnation_window,
                evidence_threshold=cfg.evidence_threshold,
                adversarial_review=cfg.adversarial_review,
                ssc_interval=cfg.ssc_interval,
            ),
            workspace=workspace,
            run_fn=runtime.run_fn,
            validate_fn=make_validate_fn(
                cwd=str(ctx.workspace),
                timeout_s=cfg.validation_timeout_seconds,
            ),
            persist_fn=runtime.persist_fn,
            answer_of=runtime.answer_of,
            on_iteration=_on_iteration,
            **kwargs,
        )

        res = await gl.run(spec)
        verdict = "verified" if res.success else f"not verified ({res.stop_cause})"
        out = f"/goal finished: {verdict} after {len(res.iterations)} iteration(s)."
        if res.final_answer:
            out += f"\n\n{res.final_answer}"
        if not res.success:
            with contextlib.suppress(Exception):
                from rune.agent.escalation import goal_escalation_hint

                hint = goal_escalation_hint(res.stop_cause)
                if hint:
                    out += f"\n{hint}"
        await ctx.broadcast("command_result", {"command": "/goal", "output": out})
    except Exception as exc:
        log.error("web_goal_loop_failed", error=str(exc)[:200])
        await ctx.broadcast(
            "command_result",
            {"command": "/goal", "output": f"/goal failed: {type(exc).__name__}"},
        )


# ── /escalate ────────────────────────────────────────────────────────────


async def _action_escalate(task: str, ctx: ActionContext) -> str:
    from rune.config import get_config

    cfg = get_config()
    provider = cfg.llm.escalation_provider
    if not provider:
        return (
            "No escalation model configured. Set llm.escalationProvider "
            "(and optionally llm.escalationModel) in config."
        )
    model = cfg.llm.escalation_model
    if not model:
        try:
            from rune.llm.client import get_llm_client
            from rune.llm.types import ModelTier, Provider

            model = get_llm_client().resolve_model(ModelTier.BEST, Provider(provider))
        except Exception:
            return f"Could not resolve a model for provider '{provider}'."

    goal = task.strip()
    if not goal:
        goal = await _last_user_turn(ctx) or ""
    if not goal:
        return "Nothing to escalate: no task given and no prior message found."

    if ctx.run_agent is None:
        return "Escalation is unavailable on this endpoint."

    from rune.types import AgentConfig

    agent_config = AgentConfig(provider=provider, model=model)
    agent_config._overridden = True

    if ctx.foreground:
        try:
            await ctx.run_agent(goal, agent_config=agent_config)  # type: ignore[misc]
        except Exception as exc:
            return f"Escalation run failed: {type(exc).__name__}: {str(exc)[:150]}"
        return f"Escalation run finished ({provider}:{model})."

    async def _run() -> None:
        with contextlib.suppress(Exception):
            await ctx.run_agent(goal, agent_config=agent_config)  # type: ignore[misc]

    _spawn(_run())
    return f"Escalating to {provider}:{model} — the run streams below."


async def _last_user_turn(ctx: ActionContext) -> str | None:
    if not ctx.session_id:
        return None
    try:
        from rune.api import conversation_wiring

        manager = conversation_wiring.get_conv_manager()
        if manager is None:
            return None
        conv = manager._active.get(ctx.session_id)
        if conv is None:
            conv = await manager._store.load(ctx.session_id)
        if conv is None:
            return None
        for turn in reversed(conv.turns):
            if turn.role == "user":
                return turn.content
    except Exception as exc:
        log.debug("last_user_turn_failed", error=str(exc)[:100])
    return None


# ── workspace actions ────────────────────────────────────────────────────


async def _action_git_diff(ctx: ActionContext) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git", "diff",
        cwd=str(ctx.workspace),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    except TimeoutError:
        proc.kill()
        return "git diff timed out."
    if proc.returncode != 0:
        err = stderr.decode(errors="replace")
        if "not a git repository" in err.lower():
            return "This workspace is not a git repository — no diff to show."
        return f"git diff failed: {err[:200]}"
    text = stdout.decode(errors="replace")
    if not text.strip():
        return "No uncommitted changes."
    lines = text.splitlines()
    if len(lines) > 400:
        text = "\n".join(lines[:400]) + f"\n… ({len(lines) - 400} more lines)"
    return f"```diff\n{text}\n```"


async def _action_files(ctx: ActionContext) -> str:
    tracker = await _get_file_tracker(ctx.workspace)
    if tracker is None:
        return "File tracking is unavailable."
    changed = tracker.get_changed_files()
    if not changed:
        return "No file changes tracked this session."
    return "Files changed this session:\n" + "\n".join(f"  {f}" for f in changed)


async def _action_undo(ctx: ActionContext) -> str:
    tracker = await _get_file_tracker(ctx.workspace)
    if tracker is None:
        return "Undo is unavailable (file tracking not running)."
    changed = tracker.get_changed_files()
    if not changed:
        return "Nothing to undo."
    last = changed[-1]
    snapshot = tracker._snapshots.get(last)
    if snapshot is None:
        return f"No snapshot for {last}; cannot undo."
    try:
        (Path(tracker._workspace) / last).write_text(snapshot)
        tracker._changed_files.discard(last)
        tracker._snapshots.pop(last, None)
        return f"Reverted {last} to its pre-change content."
    except Exception as exc:
        return f"Undo failed for {last}: {str(exc)[:150]}"


# ── memory / search / info ───────────────────────────────────────────────


async def _action_memory(sub: str, ctx: ActionContext) -> str:
    from rune.memory.project_memory import (
        find_project_memory_file,
        read_project_memory_head,
    )

    if sub == "show":
        head = read_project_memory_head(ctx.workspace)
        return head or "No project memory found."
    if sub.startswith("add"):
        text = sub[3:].lstrip(":").strip()
        if not text:
            return "Usage: /memory add <text>"
        path = find_project_memory_file(ctx.workspace)
        if path is None:
            # read_project_memory_head only finds MEMORY.md, so create it
            # under that name.
            path = ctx.workspace / "MEMORY.md"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{text}\n")
        return f"Added to project memory ({path.name})."
    if sub == "clear":
        path = find_project_memory_file(ctx.workspace)
        if path is None:
            return "No project memory file to clear."
        path.write_text("")
        return "Project memory cleared."
    return "Usage: /memory [show|add <text>|clear]"


async def _action_search(query: str, ctx: ActionContext) -> str:
    if not ctx.session_id:
        return "No live conversation to search."
    try:
        from rune.api import conversation_wiring

        manager = conversation_wiring.get_conv_manager()
        conv = None
        if manager is not None:
            conv = manager._active.get(ctx.session_id)
            if conv is None:
                conv = await manager._store.load(ctx.session_id)
        if conv is None or not conv.turns:
            return "No conversation history to search."
        q = query.lower()
        hits = [
            f"  [{i + 1}] {t.role}: {t.content[:120]}"
            for i, t in enumerate(conv.turns)
            if q in t.content.lower()
        ]
        if not hits:
            return f"No matches for '{query}'."
        return f"Matches for '{query}':\n" + "\n".join(hits[-20:])
    except Exception as exc:
        return f"Search failed: {str(exc)[:150]}"


def _action_status(ctx: ActionContext) -> str:
    uptime = int(time.monotonic() - ctx.started_at) if ctx.started_at else 0
    running = ctx.active_run_count()
    state = f"{running} run(s) in flight" if running else "idle"
    return (
        f"Agent status: {state}\n"
        f"Server uptime: {uptime // 3600}h {(uptime % 3600) // 60}m\n"
        + (f"Live session: {ctx.session_id}" if ctx.session_id else "")
    ).strip()


def _action_config() -> str:
    from rune.config import get_config

    cfg = get_config()

    def _mask(key: str | None) -> str:
        if not key:
            return "(not set)"
        return key[:6] + "…" + key[-4:] if len(key) > 12 else "***"

    return (
        f"Model: {cfg.llm.active_provider}:{cfg.llm.active_model}\n"
        f"Escalation: {cfg.llm.escalation_provider or '(none)'}"
        f":{cfg.llm.escalation_model or '(auto)'}\n"
        f"OpenAI key: {_mask(cfg.openai_api_key)}\n"
        f"Anthropic key: {_mask(cfg.anthropic_api_key)}"
    )


async def _action_sessions(ctx: ActionContext) -> str:
    try:
        from rune.api import conversation_wiring

        convs = await conversation_wiring.list_web_conversations(limit=15)
        if not convs:
            return "No saved conversations."
        lines = ["Recent conversations (use /load <id> to continue one):"]
        for c in convs:
            marker = " (current)" if c.id == ctx.session_id else ""
            title = (c.title or "(untitled)")[:60]
            lines.append(f"  {c.id}  {title}{marker}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Could not list conversations: {str(exc)[:150]}"


async def _action_load(conv_id: str, ctx: ActionContext) -> str:
    if not conv_id:
        return "Usage: /load <conversation-id> (see /sessions)"
    try:
        from rune.api import conversation_wiring

        manager = conversation_wiring.get_conv_manager()
        if manager is None:
            return "Conversation store unavailable."
        conv = await manager._store.load(conv_id)
        if conv is None:
            return f"Conversation not found: {conv_id}"
        turns = [
            {"role": t.role, "content": t.content}
            for t in conv.turns
            if t.role in ("user", "assistant")
        ][-40:]
        await ctx.broadcast(
            "command_result",
            {
                "command": "/load",
                "output": f"Loaded conversation {conv_id} ({len(conv.turns)} turns).",
                "data": {"action": "load_session", "sessionId": conv_id, "turns": turns},
            },
        )
        return ""  # broadcast already carries the structured result
    except Exception as exc:
        return f"Load failed: {str(exc)[:150]}"
