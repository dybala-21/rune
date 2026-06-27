"""Run tasks in dependency waves, each wave isolated + merged before the next.

Tasks are scheduled in Kahn levels: a wave is every remaining task whose
dependencies are done. Each wave runs as parallel isolated workers and is merged
before the next wave's worktrees are created — so a dependent task sees its
prerequisites' actual file changes, not just their text output. A wave whose
merge conflicts stops the run (the main tree is left untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rune.agent.merge import AUTO_3WAY
from rune.agent.parallel_isolated import (
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


@dataclass(slots=True)
class WaveResult:
    ok: bool
    waves_run: int = 0
    outcomes: dict[str, WorkerOutcome] = field(default_factory=dict)
    failed_wave: int = -1
    reason: str = ""


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
) -> WaveResult:
    """Run *tasks* in dependency waves with isolation + atomic per-wave merge."""
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
            ))
        wave_res = await run_wave_and_merge(
            repo, specs, isolation=isolation, policy=policy,
            timeout_seconds=timeout_seconds, cmd_builder=cmd_builder)
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
