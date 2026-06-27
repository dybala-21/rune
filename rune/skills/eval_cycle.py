"""Background skill-evaluation cycle (T1-1).

Drives gated skill learning automatically: finds distilled skills awaiting
judgement that have a replay corpus, runs the paired replay + Bayesian decision,
and persists the resulting lifecycle state to disk. Intended to be invoked
periodically by the daemon (see ``rune/daemon/main.py``), bounded to a few
skills per cycle since each evaluation runs real agent loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rune.skills.lifecycle import SkillState, get_state
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.skills.evaluation import EvalConfig, EvalReport
    from rune.skills.replay import ReplayConfig, TaskRunner

log = get_logger(__name__)

_EVALUABLE = (SkillState.CANDIDATE, SkillState.SHADOW)


async def run_skill_eval_cycle(
    *,
    store: object,
    registry: object,
    runner: TaskRunner,
    cfg: EvalConfig | None = None,
    replay_cfg: ReplayConfig | None = None,
    max_skills: int = 1,
) -> list[EvalReport]:
    """Evaluate up to *max_skills* candidate/shadow skills that have a corpus.

    For each: run paired replay + decide + transition, then persist the new
    state to disk so it survives a restart. Returns the reports produced.
    Best-effort per skill — one failure does not abort the cycle.
    """
    from rune.skills.replay import evaluate_skill_via_replay

    reports: list[EvalReport] = []
    processed = 0
    for skill in registry.list():
        if processed >= max_skills:
            break
        if get_state(skill) not in _EVALUABLE:
            continue
        # Only skills with reproducible tasks can be measured.
        try:
            corpus = store.get_replay_tasks_for_skill(skill.name, 1)
        except Exception:
            corpus = []
        if not corpus:
            continue

        processed += 1
        try:
            report = await evaluate_skill_via_replay(
                skill, store=store, registry=registry, runner=runner,
                cfg=cfg, replay_cfg=replay_cfg,
            )
        except Exception as exc:
            log.debug("skill_eval_cycle_skill_failed",
                      skill=skill.name, error=str(exc)[:120])
            continue
        if report is None:
            continue
        if report.new_state != report.old_state:
            from rune.skills.persistence import persist_skill_state
            persist_skill_state(skill)
        reports.append(report)
        log.info("skill_eval_cycle_decision", skill=skill.name,
                 action=report.decision.action,
                 transition=f"{report.old_state}->{report.new_state}")

    return reports
