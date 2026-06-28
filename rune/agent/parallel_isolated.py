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
from rune.agent.acceptance import Acceptance, evaluate_local
from rune.agent.merge import AUTO_3WAY, MergeResult, merge_changesets
from rune.agent.worktree import ChangeSet, IsolatedWorkspace
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Tail lines of a worker's captured stdout/stderr to surface (I8 observability —
# never DEVNULL a worker's logs; a silent worker failure is undiagnosable).
_LOG_TAIL_LINES = 40


@dataclass(slots=True)
class WorkerSpec:
    worker_id: str
    goal: str
    provider: str | None = None
    model: str | None = None
    max_iterations: int | None = None
    context: dict | None = None
    # Tier-1 deterministic acceptance (None -> conservative default: must change
    # at least one file). docs/design/worktree-subagent-verification.md §4-A.
    acceptance: Acceptance | None = None


@dataclass(slots=True)
class WorkerOutcome:
    worker_id: str
    result: dict = field(default_factory=dict)   # JSON contract from worker_proc
    changeset: ChangeSet | None = None
    ok: bool = False
    acceptance_reason: str = ""                   # why the I6 gate accepted/rejected
    log_tail: str = ""                            # last lines of captured worker output (I8)
    escalation_step: int = 0                      # 0=first try; >0 = which ladder rung won/failed (I7)


@dataclass(slots=True)
class Escalation:
    """Stronger-model target for the escalation ladder (I7, step 2). The caller
    resolves ModelTier.BEST -> a concrete model before passing this in (the
    worker subprocess takes a literal model string, not a tier)."""
    provider: str | None = None
    model: str | None = None


def resolve_escalation(provider: str | None = None,
                       model: str | None = None) -> Escalation | None:
    """Build an Escalation from config, resolving ModelTier.BEST when *model* is
    unset (worker_proc takes a literal model, never a tier). Returns None when no
    escalation provider is configured (ladder then ends at the feedback retry)."""
    if not provider:
        return None
    if model:
        return Escalation(provider=provider, model=model)
    try:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier, Provider
        resolved = str(get_llm_client().resolve_model(ModelTier.BEST, Provider(provider)))
        return Escalation(provider=provider, model=resolved)
    except Exception as exc:  # resolution is best-effort; worker_proc can default
        log.warning("escalation_resolve_failed", provider=provider, error=str(exc)[:120])
        return Escalation(provider=provider, model=None)


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
        # I8: capture worker stdout+stderr to a per-worker log (not DEVNULL) so a
        # silent failure is diagnosable. Surfaced as log_tail below.
        log_path = os.path.join(tmpdir, "worker.log")
        try:
            with open(log_path, "wb") as log_fh:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=ws.path, env=env,
                    stdout=log_fh, stderr=asyncio.subprocess.STDOUT,
                )
                await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
        except TimeoutError:
            log.warning("isolated_worker_timeout", worker=spec.worker_id)
            try:
                proc.kill()
            except Exception:
                pass
            outcome.acceptance_reason = "worker timed out"
            outcome.log_tail = _read_tail(log_path)
            return outcome, ws  # failed worker → empty changeset, discarded
        except Exception as exc:
            log.debug("isolated_worker_spawn_failed",
                      worker=spec.worker_id, error=str(exc)[:120])
            outcome.acceptance_reason = f"spawn failed: {str(exc)[:120]}"
            outcome.log_tail = _read_tail(log_path)
            return outcome, ws

        outcome.log_tail = _read_tail(log_path)
        if os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as fh:
                    outcome.result = json.load(fh)
            except Exception:
                pass

        # The worker crashed mid-run (no/failed result) → discard its partial
        # worktree; never merge half-written state.
        if not outcome.result.get("ok"):
            outcome.acceptance_reason = (
                "worker errored: " + str(outcome.result.get("error", ""))[:120]
                if outcome.result else "worker produced no result")
            return outcome, ws

        # I6: the worker CLAIMS done — verify deterministically against its own
        # change-set (do NOT trust the self-reported ok). Collect first, then
        # gate; reject => discard the change-set so it is not merged.
        changeset = worktree.collect(ws, spec.worker_id)
        verdict = evaluate_local(
            spec.acceptance,
            changeset.paths,
            str(outcome.result.get("answer", "")),
        )
        outcome.ok = verdict.ok
        outcome.acceptance_reason = verdict.reason or "accepted"
        outcome.changeset = changeset if verdict.ok else None
        if not verdict.ok:
            log.warning("worker_rejected", worker=spec.worker_id,
                        reason=verdict.reason,
                        self_reported_ok=bool(outcome.result.get("ok")))
        return outcome, ws
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _read_tail(path: str, lines: int = _LOG_TAIL_LINES) -> str:
    """Last *lines* of a captured worker log (best-effort, for diagnostics)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:]).strip()
    except OSError:
        return ""


# --- escalation ladder (I7) --------------------------------------------------

_RETRY_DIRECTIVE = (
    "\n\n[RUNE-ESCALATION step {step}] Your PREVIOUS attempt was REJECTED: "
    "{reason}. Do NOT repeat it. You MUST create the required file(s) ON DISK "
    "now — actually write them, do not merely describe them — then stop."
)


def _escalate_spec(spec: WorkerSpec, outcome: WorkerOutcome, step: int,
                   escalation: Escalation | None) -> WorkerSpec | None:
    """Build the next attempt's spec, or None if the ladder is exhausted.

    Step 1 = feedback-conditioned retry on the SAME model (recovers *recoverable*
    no-ops — model capable but lazy/confused). Step >=2 = escalate to a stronger
    model (the categorically-different rung; capability failures need this). NOT
    blind best-of-N: the goal always carries the concrete rejection reason.
    """
    from dataclasses import replace
    reason = outcome.acceptance_reason or "no acceptable output"
    goal = spec.goal + _RETRY_DIRECTIVE.format(step=step, reason=reason)
    if step == 1:
        return replace(spec, goal=goal)  # same model, stronger directive
    if not escalation or not escalation.provider:
        return None  # no stronger model configured → ladder ends, fail-closed
    return replace(spec, goal=goal, provider=escalation.provider,
                   model=escalation.model)


async def _run_worker_with_escalation(
    repo: str, spec: WorkerSpec, *, isolation: str, timeout_seconds: float,
    cmd_builder, max_escalation_steps: int, escalation: Escalation | None,
) -> tuple[WorkerOutcome, IsolatedWorkspace, list[IsolatedWorkspace]]:
    """Run a worker; on rejection, walk the escalation ladder against the SAME
    pre-wave base (no merge happens until after the ladder). Returns the winning
    (outcome, workspace) plus the rejected attempts' workspaces to clean up."""
    outcome, ws = await run_isolated_worker(
        repo, spec, isolation=isolation, timeout_seconds=timeout_seconds,
        cmd_builder=cmd_builder)
    losers: list[IsolatedWorkspace] = []
    step = 0
    while not outcome.ok and step < max_escalation_steps:
        step += 1
        new_spec = _escalate_spec(spec, outcome, step, escalation)
        if new_spec is None:
            break
        losers.append(ws)  # discard the rejected attempt's worktree
        outcome, ws = await run_isolated_worker(
            repo, new_spec, isolation=isolation, timeout_seconds=timeout_seconds,
            cmd_builder=cmd_builder)
        outcome.escalation_step = step
        log.info("worker_escalated", worker=spec.worker_id, step=step,
                 ok=outcome.ok, reason=outcome.acceptance_reason)
    return outcome, ws, losers


async def run_wave_and_merge(
    repo: str, specs: list[WorkerSpec], *,
    isolation: str = "auto",
    policy: str = AUTO_3WAY,
    timeout_seconds: float = 600.0,
    cmd_builder=_default_cmd,
    post_merge_check: str | None = None,
    max_escalation_steps: int = 0,
    escalation: Escalation | None = None,
) -> ParallelMergeResult:
    """Run independent workers in parallel (same base) and merge atomically.

    - Escalation (I7): if *max_escalation_steps* > 0, a Tier-1-rejected worker is
      retried against the same pre-wave base — feedback directive, then a stronger
      *escalation* model — before the merge. Bounded, fail-closed, not best-of-N.
    - Tier-2 acceptance (I6): *post_merge_check* runs as a deterministic shell
      check on the MERGED tree (compile/test); non-pass rolls back (I4). Callers
      apply it to the final wave only (intermediate trees may not build).
    """
    results = await asyncio.gather(*[
        _run_worker_with_escalation(
            repo, s, isolation=isolation, timeout_seconds=timeout_seconds,
            cmd_builder=cmd_builder, max_escalation_steps=max_escalation_steps,
            escalation=escalation)
        for s in specs
    ], return_exceptions=True)

    outcomes: list[WorkerOutcome] = []
    workspaces: list[IsolatedWorkspace] = []
    changesets: list[ChangeSet] = []
    for r in results:
        if isinstance(r, BaseException):
            log.debug("worker_task_errored", error=str(r)[:120])
            continue
        outcome, ws, losers = r
        outcomes.append(outcome)
        workspaces.append(ws)
        workspaces.extend(losers)  # rejected attempts' worktrees also cleaned up
        if outcome.changeset is not None:
            changesets.append(outcome.changeset)

    try:
        merged = merge_changesets(repo, changesets, policy=policy)
        # Tier-2 deterministic gate: verify the *merged* tree (compile/test).
        # fail-closed — anything but a clean pass rolls the merge back (I4).
        if merged.ok and post_merge_check:
            from rune.agent.evidence_gate import run_evidence_check
            from rune.agent.merge import rollback
            state, output = await run_evidence_check(post_merge_check, repo)
            if state != "pass":
                rollback(repo, merged.snapshot)
                merged.ok = False
                merged.reason = (f"post-merge check {state} (rc!=0): "
                                 + output[:300].replace("\n", " "))
                log.warning("post_merge_check_failed", state=state,
                            reason=merged.reason[:160])
    finally:
        for ws in workspaces:
            await worktree.cleanup(ws)

    return ParallelMergeResult(merge=merged, workers=outcomes)
