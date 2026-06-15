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

import contextlib
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


_EXCLUDE_DIRS = {
    ".git", ".rune", "node_modules", "vendor", "__pycache__", ".venv",
    # Build / tooling output: regenerated every run, never hand-written
    # source. Walking it floods the reviewer artifact with hundreds of
    # cache files whose fresh mtimes masquerade as "changed this
    # iteration" and starve the real source - a fully correct project
    # then fails review on omission alone.
    "target", "build", "dist", "out", "bin", "obj", "coverage", "htmlcov",
    ".next", ".nuxt", ".svelte-kit", ".parcel-cache", ".turbo",
    ".gradle", ".mvn", ".tox", ".nox", ".terraform", ".dart_tool",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".eggs",
    "Pods", "DerivedData", ".idea", ".vscode",
}
_EXCLUDE_SUFFIX = (".sum", ".lock")
# Cache Directory Tagging Standard: a directory holding this file is a
# tool cache (Cargo target/, etc.). Prune it by content marker so no
# per-tool directory-name maintenance is needed.
_CACHE_TAG = "CACHEDIR.TAG"
# Project-definition files. Acceptance criteria very often hinge on
# package name / edition / declared dependencies, so these are forced
# past the byte/file caps and listed first - the reviewer can always
# verify that class of criteria.
_ALWAYS_INCLUDE = frozenset({
    "Cargo.toml", "go.mod", "package.json", "pyproject.toml",
    "setup.py", "setup.cfg", "requirements.txt", "Makefile",
    "CMakeLists.txt", "pom.xml", "build.gradle", "build.gradle.kts",
    "build.sbt", "tsconfig.json", "Gemfile", "composer.json",
    "pubspec.yaml",
})


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
    per_file_baseline: int | None = None,
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

    seen: set[str] = set()
    tracked_changed: list[Path] = []
    for fp in sorted(tracked):  # agent-written this iteration (reliable)
        p = Path(fp)
        p = p if p.is_absolute() else rootp / p
        k = _key(p)
        if k not in seen and _eligible(p):
            seen.add(k)
            tracked_changed.append(p)

    # mtime-"changed" is unreliable: build tooling rewrites cache files
    # with fresh mtimes, so it ranks strictly below agent-written files.
    mtime_changed: list[tuple[float, Path]] = []
    baseline: list[tuple[float, Path]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(rootp):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
            if _CACHE_TAG in filenames:  # tool cache dir: prune entirely
                dirnames[:] = []
                continue
            for name in filenames:
                p = Path(dirpath) / name
                if not _eligible(p):
                    continue
                try:
                    mt = p.stat().st_mtime
                except OSError:
                    continue
                k = _key(p)
                if k in seen:
                    continue
                seen.add(k)
                if mt + 1e-6 >= started_at:  # changed this iteration
                    mtime_changed.append((mt, p))
                else:
                    baseline.append((mt, p))
    except OSError:
        pass

    mtime_changed.sort(key=lambda t: t[0], reverse=True)  # recent first
    baseline.sort(key=lambda t: t[0], reverse=True)
    ordered: list[tuple[Path, bool]] = (
        [(p, True) for p in tracked_changed]
        + [(p, True) for _mt, p in mtime_changed]
        + [(p, False) for _mt, p in baseline]
    )
    # Project-definition files first and never capped out, so criteria
    # about package name / edition / dependencies stay verifiable. Sort
    # is stable, so ordering within each group is preserved.
    ordered.sort(key=lambda t: t[0].name not in _ALWAYS_INCLUDE)

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
        forced = p.name in _ALWAYS_INCLUDE
        # The reviewer most needs this iteration's work (and the
        # project-definition files) whole; head_tail returns the text
        # untouched when its limit >= the file size, so a generous limit
        # for changed/forced files shows them entirely instead of eliding
        # their middle. Baseline context keeps the tighter limit so a big
        # stable tree cannot blow the max_total ceiling.
        lim = (
            per_file
            if (is_changed or forced) or per_file_baseline is None
            else per_file_baseline
        )
        included = forced or (len(bodies) < max_files and total < max_total)
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
        bodies.append(f"=== {rel} ===\n{head_tail(raw, lim)}\n")
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

    async def escalate_run_fn(self, prompt: str, iteration: int) -> Any:
        """Like :meth:`run_fn` but for the one final stuck-escalation attempt:
        point this runtime's loop at the configured escalation profile, run, then
        restore. Switch is in-memory and undone in ``finally`` so the loop returns
        to the local default even on failure. Falls back to a normal run when no
        escalation profile is configured."""
        from rune.config import get_config

        llm = get_config().llm
        provider = llm.escalation_provider
        if not provider:
            return await self.run_fn(prompt, iteration)
        model = llm.escalation_model
        if not model:
            from rune.llm.client import get_llm_client
            from rune.types import ModelTier, Provider
            try:
                model = get_llm_client().resolve_model(ModelTier.BEST, Provider(provider))
            except ValueError:
                return await self.run_fn(prompt, iteration)

        prev_active = (llm.active_provider, llm.active_model)
        loop_cfg = getattr(self._loop, "_config", None)
        prev_loop = (
            getattr(loop_cfg, "provider", None),
            getattr(loop_cfg, "model", None),
            getattr(loop_cfg, "_overridden", None),
        ) if loop_cfg is not None else None

        log.info("goal_escalate_attempt", provider=provider, model=model)
        llm.active_provider, llm.active_model = provider, model
        if loop_cfg is not None:
            loop_cfg.provider, loop_cfg.model, loop_cfg._overridden = provider, model, True
            with contextlib.suppress(Exception):
                from rune.agent.failover import build_profiles_from_config
                if hasattr(self._loop, "_failover"):
                    self._loop._failover._profiles = build_profiles_from_config()
        try:
            return await self.run_fn(prompt, iteration)
        finally:
            llm.active_provider, llm.active_model = prev_active
            if loop_cfg is not None and prev_loop is not None:
                loop_cfg.provider, loop_cfg.model, loop_cfg._overridden = prev_loop
                with contextlib.suppress(Exception):
                    from rune.agent.failover import build_profiles_from_config
                    if hasattr(self._loop, "_failover"):
                        self._loop._failover._profiles = build_profiles_from_config()

    def answer_of(self, _trace: Any) -> str:
        return self._last_answer

    def make_artifact_fn(
        self,
        *,
        max_total: int = 65536,
        per_file: int = 24576,
        per_file_baseline: int = 6144,
        max_files: int = 40,
        exclude_names: frozenset[str] = frozenset(),
    ) -> Callable[[], Awaitable[str]]:
        """GoalLoop ``artifact_fn``: a bounded project-source snapshot so the
        reviewer can judge whether the tests genuinely meet the spec. The
        budgets are sized so a small project's changed/definition files are
        shown WHOLE (an elided middle made the reviewer fail-close a correct
        project); ``per_file_baseline`` keeps unchanged context tighter so a
        large stable tree still cannot exceed ``max_total``. ``exclude_names``
        drops loop-managed state files by basename."""

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
                    per_file_baseline=per_file_baseline,
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
