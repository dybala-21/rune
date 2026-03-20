"""Agent Runner for RUNE CLI.

Ported from src/cli/agent-runner.ts -- single-run agent execution wrapper
that sets up the agentic loop, handles tool-call events, approval prompts,
and prints an execution summary when finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# Types
# ============================================================================


@dataclass
class AgentResult:
    """Outcome of a single agent run."""

    success: bool = False
    iterations: int = 0
    error: str | None = None
    answer: str = ""
    duration_sec: float = 0.0


@dataclass
class NativeEventHandlers:
    """Callback hooks for agent execution events."""

    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None
    on_tool_result: Callable[[str, str, bool], None] | None = None
    on_step_finish: Callable[[dict[str, Any]], None] | None = None
    on_complete: Callable[[str], None] | None = None
    on_error: Callable[[str], None] | None = None


# ============================================================================
# Default Handlers (CLI output)
# ============================================================================


def _default_on_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    from rune.utils.fast_serde import json_encode

    args_preview = json_encode(args)[:100]
    print(f"\n  Action: {tool_name}")
    print(f"   Params: {args_preview}")


def _default_on_tool_result(tool_name: str, result: str, success: bool) -> None:
    if not success:
        print(f"\n  FAIL: {result}")
    else:
        print(f"\n  Result: {result[:200]}")


def _default_on_step_finish(step: dict[str, Any]) -> None:
    text = step.get("text", "")
    step_number = step.get("step_number", "?")
    if text:
        print(f"\n  Thinking: {text[:200]}...")
    print(f"\n--- Step {step_number} ---")


def _default_on_complete(answer: str) -> None:
    separator = "-" * 50
    print(f"\n{separator}")
    print("  Final Response:")
    print(f"\n{answer}")


def _default_on_error(error: str) -> None:
    separator = "-" * 50
    print(f"\n{separator}")
    print(f"  Task Failed: {error}")


def _make_default_handlers() -> NativeEventHandlers:
    return NativeEventHandlers(
        on_tool_call=_default_on_tool_call,
        on_tool_result=_default_on_tool_result,
        on_step_finish=_default_on_step_finish,
        on_complete=_default_on_complete,
        on_error=_default_on_error,
    )


# ============================================================================
# Approval Prompt (stdin)
# ============================================================================


def _prompt_approval(command: str, reason: str | None = None) -> str:
    """Prompt the user for approval on stdin.

    Returns one of ``"approve_once"``, ``"approve_always"``, or ``"deny"``.
    """
    print(f"\n  Approval required for: {command}")
    if reason:
        print(f"   Reason: {reason}")

    try:
        answer = input("Approve? ([y] once / [a] always / [n] deny): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "deny"

    if answer in ("a", "always"):
        return "approve_always"
    if answer in ("y", "yes"):
        return "approve_once"
    return "deny"


# ============================================================================
# Runner
# ============================================================================


async def run_agent(
    goal: str,
    *,
    handlers: NativeEventHandlers | None = None,
    max_steps: int = 200,
    timeout_ms: int = 1_800_000,
) -> AgentResult:
    """Execute the RUNE agent loop for a single *goal*.

    This is the main entry point called by the CLI ``exec`` subcommand.
    It creates the agent model, sets up event handlers, prepares the
    context, runs the loop, and returns a summary result.

    The actual agent and model creation are imported lazily so that this
    module can be loaded without heavy dependencies being present.
    """
    if handlers is None:
        handlers = _make_default_handlers()

    print("\n  RUNE Agent Starting...\n")
    print(f"  Goal: {goal}\n")
    print("-" * 50)

    start_time = time.monotonic()

    try:
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )
        from rune.agent.loop import NativeAgentLoop
        from rune.types import AgentConfig
    except ImportError as exc:
        log.error("agent_import_failed: %s", exc)
        return AgentResult(
            success=False,
            error=f"Agent subsystem not available: {exc}",
            duration_sec=time.monotonic() - start_time,
        )

    try:
        agent_config = AgentConfig()
        loop = NativeAgentLoop(config=agent_config)

        # Wire approval callback (adapt sync str->bool to async)
        async def _approval_adapter(command: str, reason: str) -> bool:
            return _prompt_approval(command, reason) in ("approve_once", "approve_always")
        loop.set_approval_callback(_approval_adapter)

        # Wire event handlers
        if handlers:
            if handlers.on_tool_call:
                h = handlers.on_tool_call
                async def _tc(info: dict) -> None:
                    h(info.get("name", ""), info.get("params", {}))
                loop.on("tool_call", _tc)
            if handlers.on_complete:
                h_complete = handlers.on_complete
                async def _done(text: str) -> None:
                    h_complete(text)
                loop.on("complete", _done)

        ctx = await prepare_agent_context(PrepareContextOptions(
            goal=goal, channel="cli",
        ))

        trace = await loop.run(
            ctx.goal,
            context={"workspace_root": ctx.workspace_root},
        )

        output_text = getattr(trace, "final_answer", "") or ""
        success = trace.reason == "completed"

        try:
            await post_process_agent_result(PostProcessInput(
                context=ctx,
                success=success,
                answer=output_text,
            ))
        except Exception:
            pass  # best-effort

        result: dict[str, Any] = {
            "success": success,
            "iterations": trace.final_step,
            "answer": output_text,
            "error": None if success else trace.reason,
        }

        duration = time.monotonic() - start_time

        print("\n" + "=" * 50)
        print("  Execution Summary:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Iterations: {result.get('iterations', 0)}")
        print(f"   Duration: {duration:.2f}s")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        print("=" * 50 + "\n")

        return AgentResult(
            success=result.get("success", False),
            iterations=result.get("iterations", 0),
            error=result.get("error"),
            answer=result.get("answer", ""),
            duration_sec=duration,
        )

    except KeyboardInterrupt:
        duration = time.monotonic() - start_time
        print("\n\n  Agent interrupted by user.")
        return AgentResult(
            success=False,
            error="Interrupted by user",
            duration_sec=duration,
        )
    except Exception as exc:
        duration = time.monotonic() - start_time
        log.error("agent_execution_failed: %s", exc)
        print(f"\nUnexpected error: {exc}")
        return AgentResult(
            success=False,
            error=str(exc),
            duration_sec=duration,
        )
