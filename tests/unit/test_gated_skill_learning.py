"""Tests for Gated Skill Learning Phase 0 (T1-1).

Covers the lifecycle state machine and the observational skill_evals log.
Phase 0 is measurement-only: these assert the data plumbing, not (yet) any
change to injection behaviour.

Design: docs/design/gated-skill-learning.md
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rune.memory.store import MemoryStore
from rune.skills.lifecycle import (
    LEGACY_DEFAULT_STATE,
    SkillState,
    get_state,
    is_injectable,
    set_state,
)
from rune.skills.types import Skill


@pytest.fixture
def store(tmp_dir):
    s = MemoryStore(db_path=tmp_dir / "test_memory.db")
    yield s
    s.close()


class TestLifecycle:
    def test_legacy_skill_defaults_to_active(self):
        # A skill with no explicit state must behave as before (injected).
        s = Skill(name="legacy", description="d")
        assert get_state(s) == LEGACY_DEFAULT_STATE == SkillState.ACTIVE
        assert is_injectable(s)

    def test_set_and_get_state(self):
        s = Skill(name="x", description="d")
        set_state(s, SkillState.CANDIDATE)
        assert get_state(s) == SkillState.CANDIDATE
        assert s.metadata["state"] == "candidate"

    def test_unknown_state_rejected(self):
        s = Skill(name="x", description="d")
        with pytest.raises(ValueError):
            set_state(s, "bogus")

    def test_unknown_stored_state_falls_back_to_default(self):
        s = Skill(name="x", description="d", metadata={"state": "bogus"})
        assert get_state(s) == LEGACY_DEFAULT_STATE

    def test_retired_is_not_injectable(self):
        s = Skill(name="x", description="d")
        set_state(s, SkillState.RETIRED)
        assert not is_injectable(s)

    @pytest.mark.parametrize(
        "state",
        [SkillState.CANDIDATE, SkillState.SHADOW, SkillState.ACTIVE,
         SkillState.DEPRECATED],
    )
    def test_phase0_injection_is_permissive(self, state):
        # Phase 0 keeps everything except RETIRED injectable so behaviour is
        # unchanged. (Phase 1 narrows this to ACTIVE.)
        s = Skill(name="x", description="d")
        set_state(s, state)
        assert is_injectable(s)


class TestSkillEvalsLog:
    def test_roundtrip_summary(self, store):
        store.log_skill_eval("s-a", verified=True, arm="with", tokens=100)
        store.log_skill_eval("s-a", verified=False, arm="with")
        store.log_skill_eval("s-a", verified=True, arm="without")
        summary = store.get_skill_eval_summary("s-a")
        assert summary == {"with_n": 2, "with_s": 1,
                           "without_n": 1, "without_s": 1}

    def test_summary_empty_for_unknown_skill(self, store):
        assert store.get_skill_eval_summary("nope") == {
            "with_n": 0, "with_s": 0, "without_n": 0, "without_s": 0,
        }

    def test_records_are_isolated_per_skill(self, store):
        store.log_skill_eval("a", verified=True)
        store.log_skill_eval("b", verified=False)
        assert store.get_skill_eval_summary("a")["with_n"] == 1
        assert store.get_skill_eval_summary("b")["with_s"] == 0

    def test_prune_removes_nothing_recent(self, store):
        store.log_skill_eval("a", verified=True)
        assert store.prune_skill_evals(30) == 0
        assert store.get_skill_eval_summary("a")["with_n"] == 1

    def test_paired_counts_mcnemar(self, store):
        # 5 pairs where skill helped (with✓ without✗) -> b=5
        for i in range(5):
            pid = f"p{i}"
            store.log_skill_eval("s", verified=True, arm="with",
                                 pair_id=pid, eval_mode="offline_paired")
            store.log_skill_eval("s", verified=False, arm="without",
                                 pair_id=pid, eval_mode="offline_paired")
        # 1 pair where it hurt -> c=1
        store.log_skill_eval("s", verified=False, arm="with",
                             pair_id="ph", eval_mode="offline_paired")
        store.log_skill_eval("s", verified=True, arm="without",
                             pair_id="ph", eval_mode="offline_paired")
        # 1 concordant pair -> neither b nor c
        store.log_skill_eval("s", verified=True, arm="with",
                             pair_id="pc", eval_mode="offline_paired")
        store.log_skill_eval("s", verified=True, arm="without",
                             pair_id="pc", eval_mode="offline_paired")
        counts = store.get_skill_paired_counts("s")
        assert counts == {"b": 5, "c": 1, "n": 7}

    def test_paired_counts_ignore_observational(self, store):
        # Observational (Phase 0) rows must not be read as paired.
        store.log_skill_eval("s", verified=True)  # default observational, no pair
        assert store.get_skill_paired_counts("s") == {"b": 0, "c": 0, "n": 0}


class TestDecision:
    def test_paired_strong_positive_promotes(self):
        from rune.skills.evaluation import PROMOTE, decide_paired
        d = decide_paired(b=18, c=1, n=30, seed=42)
        assert d.action == PROMOTE
        assert d.observed_lift > 0

    def test_paired_regression_rejects(self):
        from rune.skills.evaluation import REJECT, decide_paired
        d = decide_paired(b=1, c=18, n=30, seed=42)
        assert d.action == REJECT

    def test_paired_small_sample_holds(self):
        from rune.skills.evaluation import HOLD, decide_paired
        d = decide_paired(b=2, c=0, n=3, seed=42)
        assert d.action == HOLD

    def test_unpaired_missing_arm_holds(self):
        from rune.skills.evaluation import HOLD, decide_unpaired
        d = decide_unpaired(with_n=50, with_s=45, without_n=0, without_s=0,
                            seed=42)
        assert d.action == HOLD

    def test_unpaired_strong_lift_promotes(self):
        from rune.skills.evaluation import PROMOTE, decide_unpaired
        d = decide_unpaired(with_n=80, with_s=72, without_n=80, without_s=40,
                            seed=42)
        assert d.action == PROMOTE

    def test_unpaired_regression_rejects(self):
        from rune.skills.evaluation import REJECT, decide_unpaired
        d = decide_unpaired(with_n=80, with_s=30, without_n=80, without_s=64,
                            seed=42)
        assert d.action == REJECT


class TestTransitions:
    def test_promote_candidate_to_active(self):
        from rune.skills.evaluation import PROMOTE
        from rune.skills.lifecycle import SkillState, next_state
        assert next_state(SkillState.CANDIDATE, PROMOTE) == SkillState.ACTIVE

    def test_reject_to_deprecated(self):
        from rune.skills.evaluation import REJECT
        from rune.skills.lifecycle import SkillState, next_state
        assert next_state(SkillState.ACTIVE, REJECT) == SkillState.DEPRECATED

    def test_hold_moves_candidate_to_shadow(self):
        from rune.skills.evaluation import HOLD
        from rune.skills.lifecycle import SkillState, next_state
        assert next_state(SkillState.CANDIDATE, HOLD) == SkillState.SHADOW
        assert next_state(SkillState.ACTIVE, HOLD) == SkillState.ACTIVE

    def test_terminal_states_unchanged(self):
        from rune.skills.evaluation import PROMOTE
        from rune.skills.lifecycle import SkillState, next_state
        assert next_state(SkillState.DEPRECATED, PROMOTE) == SkillState.DEPRECATED
        assert next_state(SkillState.RETIRED, PROMOTE) == SkillState.RETIRED


class TestGatedInjection:
    def test_gated_only_active_injectable(self):
        s = Skill(name="x", description="d")
        set_state(s, SkillState.CANDIDATE)
        assert not is_injectable(s, gated=True)
        set_state(s, SkillState.ACTIVE)
        assert is_injectable(s, gated=True)

    def test_build_context_filters_non_active_when_gated(self):
        from rune.skills.executor import build_skill_context_for_goal
        from rune.skills.registry import SkillRegistry

        reg = SkillRegistry()
        cand = Skill(name="deploy-helper",
                     description="deploy the app to production",
                     body="steps")
        set_state(cand, SkillState.CANDIDATE)
        reg.register(cand)

        # Ungated: candidate is injected (legacy behaviour preserved).
        assert build_skill_context_for_goal(
            "deploy the app", reg, gated=False) is not None
        # Gated: candidate filtered out -> no context.
        assert build_skill_context_for_goal(
            "deploy the app", reg, gated=True) is None

        # Promote -> now eligible under gating.
        set_state(cand, SkillState.ACTIVE)
        assert build_skill_context_for_goal(
            "deploy the app", reg, gated=True) is not None


class TestEvaluatorOrchestration:
    def test_evaluate_and_transition_promotes(self, store):
        from rune.skills.evaluation import PROMOTE, SkillEvaluator
        from rune.skills.registry import SkillRegistry

        reg = SkillRegistry()
        skill = Skill(name="winner", description="d")
        set_state(skill, SkillState.CANDIDATE)
        reg.register(skill)

        # Seed strong paired evidence.
        for i in range(20):
            pid = f"w{i}"
            store.log_skill_eval("winner", verified=True, arm="with",
                                 pair_id=pid, eval_mode="offline_paired")
            store.log_skill_eval("winner", verified=False, arm="without",
                                 pair_id=pid, eval_mode="offline_paired")

        ev = SkillEvaluator(store, reg)
        report = ev.evaluate_and_transition(skill, seed=42)
        assert report.decision.action == PROMOTE
        assert report.new_state == SkillState.ACTIVE
        assert get_state(skill) == SkillState.ACTIVE

    def test_run_cycle_holds_without_control_data(self, store):
        # Phase 0 observational data only (no 'without' arm) -> HOLD, candidate
        # drifts to shadow but is never promoted without a control arm.
        from rune.skills.evaluation import HOLD, SkillEvaluator
        from rune.skills.registry import SkillRegistry

        reg = SkillRegistry()
        skill = Skill(name="unproven", description="d")
        set_state(skill, SkillState.CANDIDATE)
        reg.register(skill)
        for _ in range(30):
            store.log_skill_eval("unproven", verified=True)  # observational

        ev = SkillEvaluator(store, reg)
        reports = ev.run_cycle(seed=42)
        assert len(reports) == 1
        assert reports[0].decision.action == HOLD
        assert reports[0].new_state == SkillState.SHADOW


class TestReplayCorpusStore:
    def test_add_and_get(self, store):
        store.add_replay_task(skill_name="s", goal="do x",
                              workspace_ref="/tmp/ws", check_cmd="pytest -q",
                              task_intent="code:fix")
        tasks = store.get_replay_tasks_for_skill("s")
        assert len(tasks) == 1
        assert tasks[0]["goal"] == "do x"
        assert tasks[0]["check_cmd"] == "pytest -q"

    def test_get_empty(self, store):
        assert store.get_replay_tasks_for_skill("nope") == []


class _FakeRunner:
    """Deterministic runner: skill helps on tasks whose goal endswith 'help'."""

    def __init__(self):
        self.calls = []

    async def run_arm(self, task, *, inject_skill, skill_body, skill_description):
        self.calls.append((task.goal, inject_skill))
        helps = task.goal.endswith("help")
        if helps:
            return inject_skill          # with✓ / without✗  -> b
        return not inject_skill          # with✗ / without✓  -> c (skill hurts)


class TestPairedReplayRunner:
    def test_counts_and_paired_rows(self, store):
        import asyncio

        from rune.skills.replay import PairedReplayRunner, ReplayTask

        tasks = [ReplayTask("s", f"task {i} help") for i in range(4)]
        tasks += [ReplayTask("s", f"task {i} hurt") for i in range(2)]
        runner = PairedReplayRunner(store, _FakeRunner())
        counts = asyncio.run(
            runner.replay_skill("s", "body", "desc", tasks))
        assert counts == {"b": 4, "c": 2, "n": 6}
        # The store's McNemar view must agree with the returned counts.
        assert store.get_skill_paired_counts("s") == {"b": 4, "c": 2, "n": 6}

    def test_respects_max_pairs(self, store):
        import asyncio

        from rune.skills.replay import (
            PairedReplayRunner,
            ReplayConfig,
            ReplayTask,
        )

        tasks = [ReplayTask("s", f"task {i} help") for i in range(10)]
        runner = PairedReplayRunner(store, _FakeRunner(),
                                    ReplayConfig(max_pairs=3))
        counts = asyncio.run(runner.replay_skill("s", "b", "d", tasks))
        assert counts["n"] == 3


class TestReplayPipeline:
    def test_replay_promotes_helpful_skill(self, store):
        import asyncio

        from rune.skills.evaluation import PROMOTE
        from rune.skills.registry import SkillRegistry
        from rune.skills.replay import ReplayTask, evaluate_skill_via_replay

        reg = SkillRegistry()
        skill = Skill(name="helper", description="d", body="steps")
        set_state(skill, SkillState.CANDIDATE)
        reg.register(skill)

        class _Corpus:
            def tasks_for(self, name, limit):
                return [ReplayTask("helper", f"t{i} help") for i in range(15)]

        report = asyncio.run(evaluate_skill_via_replay(
            skill, store=store, registry=reg,
            runner=_FakeRunner(), corpus=_Corpus(),
        ))
        assert report is not None
        assert report.decision.action == PROMOTE
        assert get_state(skill) == SkillState.ACTIVE

    def test_no_corpus_returns_none(self, store):
        import asyncio

        from rune.skills.registry import SkillRegistry
        from rune.skills.replay import evaluate_skill_via_replay

        reg = SkillRegistry()
        skill = Skill(name="x", description="d")
        set_state(skill, SkillState.CANDIDATE)

        class _Empty:
            def tasks_for(self, name, limit):
                return []

        report = asyncio.run(evaluate_skill_via_replay(
            skill, store=store, registry=reg,
            runner=_FakeRunner(), corpus=_Empty(),
        ))
        assert report is None
        assert get_state(skill) == SkillState.CANDIDATE  # untouched


class TestAgentLoopRunnerCheck:
    def test_check_passes_on_exit_zero(self):
        import asyncio

        from rune.skills.replay import AgentLoopTaskRunner
        assert asyncio.run(AgentLoopTaskRunner._check("true", os.getcwd()))

    def test_check_fails_on_nonzero(self):
        import asyncio

        from rune.skills.replay import AgentLoopTaskRunner
        assert not asyncio.run(AgentLoopTaskRunner._check("false", os.getcwd()))

    def test_no_check_is_fail_closed(self):
        import asyncio

        from rune.skills.replay import AgentLoopTaskRunner
        assert not asyncio.run(AgentLoopTaskRunner._check("", os.getcwd()))

    def test_materialize_copies_dir(self, tmp_dir):
        from rune.skills.replay import AgentLoopTaskRunner

        src = tmp_dir / "ws"
        src.mkdir()
        (src / "a.txt").write_text("hi")
        dest = AgentLoopTaskRunner._materialize(str(src))
        try:
            assert dest is not None
            assert (Path(dest) / "a.txt").read_text() == "hi"
        finally:
            import shutil
            if dest:
                shutil.rmtree(dest, ignore_errors=True)

    def test_materialize_none_for_missing(self):
        from rune.skills.replay import AgentLoopTaskRunner
        assert AgentLoopTaskRunner._materialize("") is None
        assert AgentLoopTaskRunner._materialize("/no/such/dir/xyz") is None


def _init_git_repo(path: Path) -> None:
    import subprocess
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    (path / "f.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path,
                   check=True, env=env)


class TestReplayCapture:
    def test_head_ref_none_outside_git(self, tmp_dir):
        from rune.skills.capture import capture_head_ref
        assert capture_head_ref(str(tmp_dir)) is None

    def test_head_ref_in_git_repo(self, tmp_dir):
        from rune.skills.capture import capture_head_ref
        _init_git_repo(tmp_dir)
        ref = capture_head_ref(str(tmp_dir))
        assert ref and len(ref) >= 7

    def test_capture_gated_off_by_default(self, store, tmp_dir):
        # Default config has capture_replay off -> no capture, no row.
        from rune.skills.capture import capture_replay_snapshot
        _init_git_repo(tmp_dir)
        ok = capture_replay_snapshot("goal", "s", store=store, cwd=str(tmp_dir))
        assert ok is False
        assert store.get_replay_tasks_for_skill("s") == []

    def test_capture_records_pretask_snapshot(self, store, tmp_dir, monkeypatch):
        from rune.skills import capture as cap_mod
        from rune.skills.capture import capture_head_ref, capture_replay_snapshot

        _init_git_repo(tmp_dir)
        pre_ref = capture_head_ref(str(tmp_dir))

        # Enable the gate.
        class _Cfg:
            class skills:
                capture_replay = True
        monkeypatch.setattr("rune.config.get_config", lambda: _Cfg)
        # Deterministic check command via the documented env override.
        monkeypatch.setenv("RUNE_AUTO_VERIFY_CMD", "true")
        # Corpus dir under tmp to avoid touching real ~/.rune.
        monkeypatch.setattr(cap_mod, "_corpus_root",
                            lambda: tmp_dir / "corpus")

        # Agent "mutates" the tree AND commits after pre_ref was captured.
        import subprocess
        (tmp_dir / "f.txt").write_text("changed by agent")
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "commit", "-aqm", "agent work"], cwd=tmp_dir,
                       check=True, env=env)

        ok = capture_replay_snapshot(
            "goal", "s", store=store, cwd=str(tmp_dir), head_ref=pre_ref)
        assert ok is True
        tasks = store.get_replay_tasks_for_skill("s")
        assert len(tasks) == 1
        assert tasks[0]["check_cmd"] == "true"
        # Snapshot must hold the PRE-task content, not the agent's commit.
        snap = Path(tasks[0]["workspace_ref"]) / "f.txt"
        assert snap.read_text() == "original"
