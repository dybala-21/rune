"""Run tasks in dependency waves, each wave isolated + merged before the next.

Tasks are scheduled in Kahn levels: a wave is every remaining task whose
dependencies are done. Each wave runs as parallel isolated workers and is merged
before the next wave's worktrees are created — so a dependent task sees its
prerequisites' actual file changes, not just their text output. A wave whose
merge conflicts stops the run (the main tree is left untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rune.agent.acceptance import Acceptance
from rune.agent.merge import AUTO_3WAY
from rune.agent.parallel_isolated import (
    Escalation,
    WorkerOutcome,
    WorkerSpec,
    _default_cmd,
    run_wave_and_merge,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class WaveTask:
    id: str
    goal: str
    dependencies: list[str] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    max_iterations: int | None = None
    acceptance: Acceptance | None = None   # Tier-1 deterministic gate (I6); None=default floor


@dataclass(slots=True)
class WaveResult:
    ok: bool
    waves_run: int = 0
    outcomes: dict[str, WorkerOutcome] = field(default_factory=dict)
    failed_wave: int = -1
    reason: str = ""


@dataclass(slots=True)
class VerificationStats:
    """§8 instrumentation. ``self_report_mismatch`` = how often a worker's
    self-reported ok disagreed with the deterministic gate (self-judgment is
    unreliable); ``by_step`` = recoveries per escalation rung (escalation-vs-
    resampling data)."""
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    self_report_mismatch: int = 0           # self-reported ok=True but deterministically rejected
    escalation_recoveries: int = 0          # accepted only after >=1 escalation step
    by_step: dict[int, int] = field(default_factory=dict)  # winning escalation_step -> #accepted


def summarize_verification(
    outcomes: dict[str, WorkerOutcome] | list[WorkerOutcome],
) -> VerificationStats:
    """Deterministic verification metrics over a run's worker outcomes."""
    vals = list(outcomes.values()) if isinstance(outcomes, dict) else list(outcomes)
    st = VerificationStats(total=len(vals))
    for o in vals:
        if o.ok:
            st.accepted += 1
            if o.escalation_step > 0:
                st.escalation_recoveries += 1
            st.by_step[o.escalation_step] = st.by_step.get(o.escalation_step, 0) + 1
        else:
            st.rejected += 1
            if bool(o.result.get("ok")):
                st.self_report_mismatch += 1
    return st


def _compute_waves(tasks: list[WaveTask]) -> list[list[WaveTask]] | None:
    """Kahn levels; returns None on cycle/unknown dependency."""
    ids = {t.id for t in tasks}
    for t in tasks:
        if any(d not in ids for d in t.dependencies):
            return None  # dependency on unknown task
    done: set[str] = set()
    remaining = list(tasks)
    waves: list[list[WaveTask]] = []
    while remaining:
        ready = [t for t in remaining if all(d in done for d in t.dependencies)]
        if not ready:
            return None  # cycle
        waves.append(ready)
        done |= {t.id for t in ready}
        remaining = [t for t in remaining if t.id not in done]
    return waves


async def execute_waves(
    repo: str, tasks: list[WaveTask], *,
    isolation: str = "auto",
    policy: str = AUTO_3WAY,
    timeout_seconds: float = 600.0,
    cmd_builder=_default_cmd,
    post_merge_check: str | None = None,
    max_escalation_steps: int = 0,
    escalation: Escalation | None = None,
) -> WaveResult:
    """Run *tasks* in dependency waves with isolation + atomic per-wave merge.

    - Escalation (I7): *max_escalation_steps*/*escalation* apply to every wave.
    - Tier-2 acceptance (I6): *post_merge_check* (compile/test) is applied to the
      FINAL wave only — intermediate waves are partial and may not build. On
      non-pass the final merge rolls back (I4), fail-closed.
    """
    waves = _compute_waves(tasks)
    if waves is None:
        return WaveResult(ok=False, reason="dependency cycle or unknown dependency")

    res = WaveResult(ok=True)
    completed_outputs: dict[str, str] = {}
    for i, wave in enumerate(waves):
        specs = []
        for t in wave:
            # prerequisites' text output as context; their file changes are
            # already in this wave's base from the prior merge.
            ctx = {d: completed_outputs.get(d, "") for d in t.dependencies} or None
            specs.append(WorkerSpec(
                worker_id=t.id, goal=t.goal, provider=t.provider,
                model=t.model, max_iterations=t.max_iterations,
                context={"dependencies": ctx} if ctx else None,
                acceptance=t.acceptance,
            ))
        # Tier-2 compile/test gate runs only after the FINAL wave (the fully
        # assembled tree); intermediate waves are partial and may not build.
        is_final = i == len(waves) - 1
        wave_res = await run_wave_and_merge(
            repo, specs, isolation=isolation, policy=policy,
            timeout_seconds=timeout_seconds, cmd_builder=cmd_builder,
            post_merge_check=post_merge_check if is_final else None,
            max_escalation_steps=max_escalation_steps, escalation=escalation)
        res.waves_run += 1
        for o in wave_res.workers:
            res.outcomes[o.worker_id] = o
            if o.ok:
                completed_outputs[o.worker_id] = o.result.get("answer", "")
        if not wave_res.merge.ok:
            res.ok = False
            res.failed_wave = i
            res.reason = wave_res.merge.reason or "wave merge failed"
            log.warning("wave_merge_failed", wave=i, reason=res.reason)
            break  # fail-closed: stop; main tree is unchanged for this wave
    return res
