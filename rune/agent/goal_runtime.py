"""Runtime bridge between :class:`~rune.agent.goal_loop.GoalLoop` and the rune
agent/memory stack.

``post_process_agent_result`` persists an episode with
``utility = 1 if success else -1`` and later injects negative ones as
warnings. This wrapper calls it once, at the terminal outcome only, never per
iteration, so a failing ``/goal`` loop does not record many negative episodes.
It is a wrapper and changes no core code.

Each iteration runs with a fresh context (``message_history=None``, no
``resume_session_id``); continuity comes from the workspace state files owned
by ``GoalLoop`` (SPEC, fix_plan, progress), not chat history.

Collaborators are injected so this can be unit-tested without the agent or
memory stack.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


class _LoopLike:
    """Structural: NativeAgentLoop subset used here."""

    def on(self, event: str, handler: Callable[..., Any]) -> None: ...
    def off(self, event: str, handler: Callable[..., Any]) -> None: ...
    async def run(self, goal: str, **kwargs: Any) -> Any: ...


_EXCLUDE_DIRS = {".git", ".rune", "node_modules", "vendor", "__pycache__", ".venv"}
_EXCLUDE_SUFFIX = (".sum", ".lock")


def _is_texty(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\x00" not in fh.read(1024)
    except OSError:
        return False


def _collect_artifact(
    root: str,
    started_at: float,
    tracked: set[str],
    *,
    max_total: int,
    per_file: int,
    max_files: int,
) -> str:
    """Bounded snapshot of changed source: the files the agent wrote via the
    file tools, plus any source under *root* modified since *started_at*
    (covers files written via bash that tool tracking misses). Best-effort."""
    rootp = Path(root) if root else Path.cwd()
    cands: list[Path] = []
    for fp in sorted(tracked):  # tracked (tool-written) files first
        p = Path(fp)
        cands.append(p if p.is_absolute() else rootp / p)
    try:
        for dirpath, dirnames, filenames in os.walk(rootp):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
            for name in filenames:
                p = Path(dirpath) / name
                try:
                    if p.stat().st_mtime + 1e-6 >= started_at:
                        cands.append(p)
                except OSError:
                    continue
    except OSError:
        pass

    # (rel, lines, bytes, content_included) for every changed source file, so
    # the reviewer still knows what exists when content is capped.
    manifest: list[tuple[str, int, int, bool]] = []
    bodies: list[str] = []
    total = 0
    seen: set[str] = set()
    for p in cands:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.name.endswith(_EXCLUDE_SUFFIX) or any(
            part in _EXCLUDE_DIRS for part in p.parts
        ):
            continue
        try:
            if not p.is_file() or p.stat().st_size > 256 * 1024:
                continue
        except OSError:
            continue
        if not _is_texty(p):
            continue
        try:
            raw = p.read_text("utf-8", "replace")
        except OSError:
            continue
        try:
            rel = str(p.relative_to(rootp))
        except ValueError:
            rel = p.name
        included = len(bodies) < max_files and total < max_total
        manifest.append(
            (rel, len(raw.splitlines()), len(raw.encode("utf-8", "replace")), included)
        )
        if not included:
            continue
        text = raw
        if len(text) > per_file:
            text = (
                text[: per_file * 3 // 4]
                + f"\n... ({len(text)} chars, elided) ...\n"
                + text[-per_file // 4 :]
            )
        block = f"=== {rel} ===\n{text}\n"
        bodies.append(block)
        total += len(block)

    if not manifest:
        return ""

    shown = sum(1 for m in manifest if m[3])
    man_cap = 60
    head = [
        f"CHANGED FILES MANIFEST - {len(manifest)} changed source file(s), "
        f"{shown} with content shown below:"
    ]
    for rel, nlines, nbytes, inc in manifest[:man_cap]:
        head.append(
            f"- {rel} ({nlines} lines, {nbytes} B) "
            f"[{'shown' if inc else 'omitted: cap'}]"
        )
    if len(manifest) > man_cap:
        head.append(f"  (+{len(manifest) - man_cap} more)")
    return "\n".join(head) + "\n\n" + "".join(bodies)[:max_total]


class GoalRuntime:
    def __init__(
        self,
        loop: _LoopLike,
        *,
        channel: str = "tui",
        conversation_id: str = "",
        prepare: Callable[..., Awaitable[Any]] | None = None,
        post_process: Callable[..., Awaitable[None]] | None = None,
        make_opts: Callable[..., Any] | None = None,
        make_post_input: Callable[..., Any] | None = None,
    ) -> None:
        self._loop = loop
        self._channel = channel
        self._conversation_id = conversation_id
        self._prepare = prepare
        self._post = post_process
        self._make_opts = make_opts
        self._make_post_input = make_post_input
        self._last_ctx: Any | None = None
        self._last_answer: str = ""
        self._last_files: set[str] = set()
        self._run_started_at: float = 0.0

    # -- lazy real-stack wiring (skipped entirely when injected) -----------

    def _resolve(self) -> None:
        if self._prepare and self._post and self._make_opts and self._make_post_input:
            return
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )

        self._prepare = self._prepare or prepare_agent_context
        self._post = self._post or post_process_agent_result
        self._make_opts = self._make_opts or PrepareContextOptions
        self._make_post_input = self._make_post_input or PostProcessInput

    # -- GoalLoop injection points ----------------------------------------

    async def run_fn(self, prompt: str, iteration: int) -> Any:
        """Run one fresh-context attempt. Does not persist episodic memory."""
        self._resolve()
        assert self._prepare and self._make_opts  # for type-checkers

        ctx = await self._prepare(
            self._make_opts(
                goal=prompt,
                channel=self._channel,
                conversation_id=self._conversation_id,
            )
        )

        collected: list[str] = []

        def _collect(delta: str) -> None:
            collected.append(delta)

        self._loop.on("text_delta", _collect)
        self._run_started_at = time.time()
        try:
            trace = await self._loop.run(
                getattr(ctx, "goal", prompt),
                context={"workspace_root": getattr(ctx, "workspace_root", "")},
                message_history=None,  # fresh context each attempt
            )
        finally:
            self._loop.off("text_delta", _collect)
            # Snapshot files the agent wrote this iteration before the next
            # run()'s _reset_run_state() clears the loop's tracking set.
            self._last_files = set(
                getattr(self._loop, "_files_written", set()) or set()
            )

        self._last_ctx = ctx
        self._last_answer = (
            "".join(collected).strip()
            or getattr(trace, "final_text", "")
            or self._last_answer
        )
        return trace

    def answer_of(self, _trace: Any) -> str:
        return self._last_answer

    def make_artifact_fn(
        self, *, max_total: int = 16384, per_file: int = 4096, max_files: int = 40
    ) -> Callable[[], Awaitable[str]]:
        """GoalLoop ``artifact_fn``: a bounded changed-source snapshot so the
        reviewer can detect a passing but empty or no-assertion test."""

        async def _artifact() -> str:
            root = getattr(self._last_ctx, "workspace_root", "") or os.getcwd()
            try:
                return _collect_artifact(
                    root,
                    self._run_started_at,
                    set(self._last_files),
                    max_total=max_total,
                    per_file=per_file,
                    max_files=max_files,
                )
            except Exception as exc:  # context only; do not block on failure
                log.debug("goal_runtime_artifact_failed", error=str(exc)[:200])
                return ""

        return _artifact

    async def persist_fn(self, success: bool, answer: str) -> None:
        """Called once by GoalLoop at the terminal outcome only."""
        if self._last_ctx is None:
            return  # nothing ran (e.g. cancelled before first iteration)
        self._resolve()
        assert self._post and self._make_post_input
        try:
            await self._post(
                self._make_post_input(
                    context=self._last_ctx,
                    success=success,
                    answer=answer or self._last_answer,
                )
            )
        except Exception as exc:  # best-effort; do not crash the terminal step
            log.debug("goal_runtime_persist_failed", error=str(exc)[:200])
