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

from rune.agent.obs_cap import head_tail
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
    exclude_names: frozenset[str] = frozenset(),
) -> str:
    """Bounded snapshot of the project source for the reviewer. Always
    includes the project's source (not only files changed this iteration), so
    the reviewer keeps seeing the real code after it stabilises. Files changed
    this iteration (tool-written or modified since *started_at*) are shown
    first; the rest follow most-recent-first (a language/scale-agnostic
    relevance proxy). ``exclude_names`` drops loop-managed state files
    (SPEC/plan/progress/feedback) by basename so they never pose as source.
    Best-effort; bounded by the caps regardless of repo size."""
    rootp = Path(root) if root else Path.cwd()

    def _key(p: Path) -> str:
        try:
            return str(p.resolve())
        except OSError:
            return str(p)

    def _eligible(p: Path) -> bool:
        if p.name in exclude_names or p.name.endswith(_EXCLUDE_SUFFIX):
            return False
        if any(part in _EXCLUDE_DIRS for part in p.parts):
            return False
        try:
            if not p.is_file() or p.stat().st_size > 256 * 1024:
                return False
        except OSError:
            return False
        return _is_texty(p)

    changed: list[Path] = []
    changed_keys: set[str] = set()
    for fp in sorted(tracked):  # tool-written files (this iteration)
        p = Path(fp)
        p = p if p.is_absolute() else rootp / p
        k = _key(p)
        if k not in changed_keys and _eligible(p):
            changed_keys.add(k)
            changed.append(p)

    baseline: list[tuple[float, Path]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(rootp):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
            for name in filenames:
                p = Path(dirpath) / name
                if not _eligible(p):
                    continue
                try:
                    mt = p.stat().st_mtime
                except OSError:
                    continue
                k = _key(p)
                if mt + 1e-6 >= started_at:  # changed this iteration
                    if k not in changed_keys:
                        changed_keys.add(k)
                        changed.append(p)
                else:
                    baseline.append((mt, p))
    except OSError:
        pass

    baseline.sort(key=lambda t: t[0], reverse=True)  # most-recent first
    ordered: list[tuple[Path, bool]] = [(p, True) for p in changed]
    seen = set(changed_keys)
    for _mt, p in baseline:
        k = _key(p)
        if k not in seen:
            seen.add(k)
            ordered.append((p, False))

    # (rel, lines, bytes, content_shown, is_changed) for every file, so the
    # reviewer knows the full inventory even when content is capped.
    manifest: list[tuple[str, int, int, bool, bool]] = []
    bodies: list[str] = []
    total = 0
    for p, is_changed in ordered:
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
            (
                rel,
                len(raw.splitlines()),
                len(raw.encode("utf-8", "replace")),
                included,
                is_changed,
            )
        )
        if not included:
            continue
        bodies.append(f"=== {rel} ===\n{head_tail(raw, per_file)}\n")
        total += len(bodies[-1])

    if not manifest:
        return ""

    shown = sum(1 for m in manifest if m[3])
    man_cap = 60
    head = [
        f"SOURCE MANIFEST - {len(manifest)} file(s), {shown} with content shown "
        f"below ([changed] = modified this iteration):"
    ]
    for rel, nlines, nbytes, inc, ch in manifest[:man_cap]:
        head.append(
            f"- {rel} ({nlines} lines, {nbytes} B) "
            f"[{'changed' if ch else 'baseline'}] "
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
        self,
        *,
        max_total: int = 16384,
        per_file: int = 4096,
        max_files: int = 40,
        exclude_names: frozenset[str] = frozenset(),
    ) -> Callable[[], Awaitable[str]]:
        """GoalLoop ``artifact_fn``: a bounded project-source snapshot so the
        reviewer can judge whether the tests genuinely meet the spec.
        ``exclude_names`` drops loop-managed state files by basename."""

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
                    exclude_names=exclude_names,
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
