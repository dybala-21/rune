"""Gated Skill Learning — paired replay runner (T1-1).

Generates the *control arm* a gated promotion needs: re-runs reproducible past
tasks twice — once with the candidate skill injected, once without — in
isolated workspaces, and judges each arm by a *deterministic check* (not the
agent's self-report). The paired outcomes are logged so the evaluator can
promote a skill only when it measurably raises the verified rate.

Structure mirrors the evaluator: a testable orchestration core
(:class:`PairedReplayRunner`) behind two seams —

* :class:`TaskRunner`  — executes one arm and returns whether it verified.
* :class:`ReplayCorpus` — yields reproducible tasks for a skill.

:class:`AgentLoopTaskRunner` is the real adapter (isolated workspace + agent
loop + check); tests drive the core with fakes so the heavy I/O path is not on
the critical test path.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.skills.evaluation import EvalConfig, EvalReport
    from rune.skills.types import Skill

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReplayTask:
    """A reproducible task: goal + materialisable workspace + a check."""

    skill_name: str
    goal: str
    workspace_ref: str = ""   # a directory to copy (or "" for no workspace)
    check_cmd: str = ""       # shell command; exit 0 == verified
    task_intent: str = ""


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Budget caps for a replay session (sub-user-cost, background only)."""

    max_pairs: int = 20          # cap tasks replayed per skill per cycle
    arm_timeout_seconds: int = 300


class TaskRunner(Protocol):
    """Runs one arm of a paired replay and returns whether it verified."""

    async def run_arm(
        self, task: ReplayTask, *, inject_skill: bool,
        skill_body: str, skill_description: str,
    ) -> bool: ...


class ReplayCorpus(Protocol):
    """Yields reproducible tasks to A/B a given skill on."""

    def tasks_for(self, skill_name: str, limit: int) -> list[ReplayTask]: ...


class StoreReplayCorpus:
    """Default corpus backed by the ``replay_corpus`` table."""

    def __init__(self, store: object) -> None:
        self._store = store

    def tasks_for(self, skill_name: str, limit: int) -> list[ReplayTask]:
        try:
            rows = self._store.get_replay_tasks_for_skill(skill_name, limit)
        except Exception:
            return []
        return [
            ReplayTask(
                skill_name=r["skill_name"], goal=r["goal"],
                workspace_ref=r["workspace_ref"], check_cmd=r["check_cmd"],
                task_intent=r["task_intent"],
            )
            for r in rows
        ]


class PairedReplayRunner:
    """Orchestrates paired replay and logs ``offline_paired`` outcomes.

    For each task: run the *without* arm and the *with* arm, then write two
    ``skill_evals`` rows sharing a ``pair_id``. Returns the McNemar counts
    (``b`` = with✓/without✗, ``c`` = with✗/without✓, ``n`` = pairs).
    """

    def __init__(
        self, store: object, runner: TaskRunner,
        cfg: ReplayConfig | None = None,
    ) -> None:
        self._store = store
        self._runner = runner
        self._cfg = cfg or ReplayConfig()

    async def replay_skill(
        self, skill_name: str, skill_body: str, skill_description: str,
        tasks: list[ReplayTask],
    ) -> dict[str, int]:
        b = c = n = 0
        for i, task in enumerate(tasks[: self._cfg.max_pairs]):
            pair_id = f"{skill_name}:{i}"
            try:
                # Control arm first, then treatment — order is irrelevant to the
                # paired analysis but keeps logs readable.
                without_v = await self._runner.run_arm(
                    task, inject_skill=False,
                    skill_body=skill_body, skill_description=skill_description,
                )
                with_v = await self._runner.run_arm(
                    task, inject_skill=True,
                    skill_body=skill_body, skill_description=skill_description,
                )
            except Exception as exc:  # one bad task must not abort the cycle
                log.debug("replay_task_failed", skill=skill_name,
                          pair=pair_id, error=str(exc)[:120])
                continue

            self._log_arm(skill_name, pair_id, "without", without_v, task)
            self._log_arm(skill_name, pair_id, "with", with_v, task)
            n += 1
            if with_v and not without_v:
                b += 1
            elif not with_v and without_v:
                c += 1

        log.info("replay_skill_done", skill=skill_name, pairs=n, b=b, c=c)
        return {"b": b, "c": c, "n": n}

    def _log_arm(
        self, skill_name: str, pair_id: str, arm: str, verified: bool,
        task: ReplayTask,
    ) -> None:
        try:
            self._store.log_skill_eval(
                skill_name, verified=verified, arm=arm, pair_id=pair_id,
                eval_mode="offline_paired", task_intent=task.task_intent,
            )
        except Exception as exc:
            log.debug("replay_log_failed", error=str(exc)[:120])


async def evaluate_skill_via_replay(
    skill: Skill,
    *,
    store: object,
    registry: object,
    runner: TaskRunner,
    corpus: ReplayCorpus | None = None,
    cfg: EvalConfig | None = None,
    replay_cfg: ReplayConfig | None = None,
) -> EvalReport | None:
    """Full pipeline: replay -> evaluate -> transition.

    Returns the :class:`EvalReport`, or ``None`` if there is no replay corpus
    for the skill (nothing to measure → caller leaves it untouched).
    """
    from rune.skills.evaluation import SkillEvaluator

    corpus = corpus or StoreReplayCorpus(store)
    replay_cfg = replay_cfg or ReplayConfig()
    tasks = corpus.tasks_for(skill.name, replay_cfg.max_pairs)
    if not tasks:
        log.debug("replay_no_corpus", skill=skill.name)
        return None

    pr = PairedReplayRunner(store, runner, replay_cfg)
    await pr.replay_skill(skill.name, skill.body, skill.description, tasks)

    evaluator = SkillEvaluator(store, registry, cfg)
    return evaluator.evaluate_and_transition(skill)


# Real adapter

class AgentLoopTaskRunner:
    """Runs a replay arm in an isolated copy of the task's workspace.

    Isolation: copies ``workspace_ref`` to a tempdir and ``chdir``s into it
    (the agent loop reads ``os.getcwd()``); runs sequentially because chdir is
    process-global. The skill is injected into the *with* arm via
    ``run(extra_system_context=...)`` — auto_skill stays off so the registry's
    matcher cannot confound the comparison. ``verified`` comes from the task's
    deterministic ``check_cmd`` (exit 0), never the agent's self-report.

    Subprocess-level isolation (concurrent arms) is a hardening follow-up.
    """

    def __init__(self, *, role: str = "executor",
                 max_iterations: int | None = None,
                 provider: str | None = None,
                 model: str | None = None,
                 temperature: float | None = None,
                 timeout_seconds: float | None = None) -> None:
        self._role = role
        self._max_iterations = max_iterations
        # Optional explicit provider/model — the design allows a cheap/local
        # model for eval since the verdict comes from a deterministic check.
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds

    async def run_arm(
        self, task: ReplayTask, *, inject_skill: bool,
        skill_body: str, skill_description: str,
    ) -> bool:
        work = self._materialize(task.workspace_ref)
        prev_cwd = os.getcwd()
        try:
            if work:
                os.chdir(work)
            await self._run_agent(task.goal, inject_skill, skill_body,
                                  skill_description)
            return await self._check(task.check_cmd, work or prev_cwd)
        finally:
            os.chdir(prev_cwd)
            if work:
                shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _materialize(workspace_ref: str) -> str | None:
        if not workspace_ref or not os.path.isdir(workspace_ref):
            return None
        dest = tempfile.mkdtemp(prefix="rune-replay-")
        # copytree needs a non-existing dest; replace the freshly-made dir.
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(workspace_ref, dest, symlinks=True)
        return dest

    async def _run_agent(
        self, goal: str, inject_skill: bool, skill_body: str,
        skill_description: str,
    ) -> None:
        from rune.agent.loop import create_agent_loop

        if self._provider or self._model:
            from rune.agent.loop import NativeAgentLoop
            from rune.types import AgentConfig
            cfg = AgentConfig(_overridden=True)
            if self._provider:
                cfg.provider = self._provider
            if self._model:
                cfg.model = self._model
            if self._max_iterations:
                cfg.max_iterations = self._max_iterations
            if self._temperature is not None:
                cfg.temperature = self._temperature
            loop = NativeAgentLoop(config=cfg)
        else:
            loop = create_agent_loop(
                role=self._role, max_iterations=self._max_iterations,
            )
        # Keep the comparison clean: registry-driven skill injection off.
        loop._auto_skill = False
        # Replay runs non-interactively in an isolated copy, so "ask"-level
        # actions are auto-approved. The Guardian still hard-denies critical
        # patterns (rm -rf, etc.) regardless of this callback.
        async def _approve(_cmd: str, _reason: str) -> bool:
            return True
        try:
            loop.set_approval_callback(_approve)
        except Exception:
            pass
        extra = None
        if inject_skill:
            extra = (
                f"## Reusable skill: {skill_description}\n{skill_body}"
            )
        # Isolate this throwaway run from the user's real telemetry: the loop
        # skips tool-call logging / behavior prediction when RUNE_IN_BEST_OF is
        # set (same recursion-guard the best-of-K sampler uses).
        _prev_env = os.environ.get("RUNE_IN_BEST_OF")
        os.environ["RUNE_IN_BEST_OF"] = "1"
        try:
            coro = loop.run(goal, extra_system_context=extra)
            if self._timeout_seconds:
                await asyncio.wait_for(coro, timeout=self._timeout_seconds)
            else:
                await coro
        except (TimeoutError, Exception) as exc:
            log.debug("replay_arm_run_failed", error=str(exc)[:120])
        finally:
            if _prev_env is None:
                os.environ.pop("RUNE_IN_BEST_OF", None)
            else:
                os.environ["RUNE_IN_BEST_OF"] = _prev_env

    @staticmethod
    async def _check(check_cmd: str, cwd: str) -> bool:
        if not check_cmd:
            # No deterministic check → cannot trust the outcome; count as fail
            # (fail-closed) so an uncheckable task never inflates a skill's lift.
            log.debug("replay_no_check_cmd")
            return False
        try:
            proc = await asyncio.create_subprocess_shell(
                check_cmd, cwd=cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()
            return rc == 0
        except Exception as exc:
            log.debug("replay_check_failed", error=str(exc)[:120])
            return False
