"""Run a wave of isolated subagents in parallel, then merge their changes.

Each worker runs in its own worktree as a subprocess (separate process = its own
cwd, so they really run concurrently), all branching from the same base; the
change-sets are merged atomically afterwards. cmd_builder is injectable so the
whole pipeline is testable with a fake worker (no LLM). Dependency-aware
scheduling across waves is in wave_orchestrator.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field

from rune.agent import worktree
from rune.agent.merge import AUTO_3WAY, MergeResult, merge_changesets
from rune.agent.worktree import ChangeSet, IsolatedWorkspace
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class WorkerSpec:
    worker_id: str
    goal: str
    provider: str | None = None
    model: str | None = None
    max_iterations: int | None = None
    context: dict | None = None


@dataclass(slots=True)
class WorkerOutcome:
    worker_id: str
    result: dict = field(default_factory=dict)   # JSON contract from worker_proc
    changeset: ChangeSet | None = None
    ok: bool = False


@dataclass(slots=True)
class ParallelMergeResult:
    merge: MergeResult
    workers: list[WorkerOutcome] = field(default_factory=list)


def _default_cmd(spec_path: str, result_path: str,
                 ws: IsolatedWorkspace) -> list[str]:
    return [sys.executable, "-m", "rune.agent.worker_proc",
            "--spec", spec_path, "--result", result_path]


async def run_isolated_worker(
    repo: str, spec: WorkerSpec, *,
    isolation: str = "auto",
    timeout_seconds: float = 600.0,
    cmd_builder=_default_cmd,
) -> tuple[WorkerOutcome, IsolatedWorkspace]:
    """Create an isolated workspace, run the worker subprocess in it, collect
    its change-set. Caller merges + cleans up (so changesets can be merged as a
    group). Returns (outcome, workspace)."""
    ws = await worktree.create(repo, spec.worker_id, isolation=isolation)
    outcome = WorkerOutcome(worker_id=spec.worker_id)
    tmpdir = tempfile.mkdtemp(prefix="rune-wspec-")
    try:
        spec_path = os.path.join(tmpdir, "spec.json")
        result_path = os.path.join(tmpdir, "result.json")
        with open(spec_path, "w", encoding="utf-8") as fh:
            json.dump({
                "root": ws.path, "goal": spec.goal, "provider": spec.provider,
                "model": spec.model, "max_iterations": spec.max_iterations,
                "context": spec.context,
            }, fh)

        env = {**os.environ,
               "RUNE_ISOLATION_ROOT": ws.path, "RUNE_WORKER": "1"}
        cmd = cmd_builder(spec_path, result_path, ws)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=ws.path, env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
        except TimeoutError:
            log.warning("isolated_worker_timeout", worker=spec.worker_id)
            try:
                proc.kill()
            except Exception:
                pass
            return outcome, ws  # failed worker → empty changeset, discarded
        except Exception as exc:
            log.debug("isolated_worker_spawn_failed",
                      worker=spec.worker_id, error=str(exc)[:120])
            return outcome, ws

        if os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as fh:
                    outcome.result = json.load(fh)
            except Exception:
                pass
        outcome.ok = bool(outcome.result.get("ok"))
        # Only collect a successful worker's changes (discard partial/failed).
        if outcome.ok:
            outcome.changeset = worktree.collect(ws, spec.worker_id)
        return outcome, ws
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


async def run_wave_and_merge(
    repo: str, specs: list[WorkerSpec], *,
    isolation: str = "auto",
    policy: str = AUTO_3WAY,
    timeout_seconds: float = 600.0,
    cmd_builder=_default_cmd,
) -> ParallelMergeResult:
    """Run a wave of independent workers in parallel, then merge atomically.

    All workers branch from the same base. Their change-sets are merged
    all-or-nothing (§merge). Worktrees are always cleaned up.
    """
    results = await asyncio.gather(*[
        run_isolated_worker(repo, s, isolation=isolation,
                            timeout_seconds=timeout_seconds,
                            cmd_builder=cmd_builder)
        for s in specs
    ], return_exceptions=True)

    outcomes: list[WorkerOutcome] = []
    workspaces: list[IsolatedWorkspace] = []
    changesets: list[ChangeSet] = []
    for r in results:
        if isinstance(r, BaseException):
            log.debug("worker_task_errored", error=str(r)[:120])
            continue
        outcome, ws = r
        outcomes.append(outcome)
        workspaces.append(ws)
        if outcome.changeset is not None:
            changesets.append(outcome.changeset)

    try:
        merged = merge_changesets(repo, changesets, policy=policy)
    finally:
        for ws in workspaces:
            await worktree.cleanup(ws)

    return ParallelMergeResult(merge=merged, workers=outcomes)
