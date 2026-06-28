"""Tests for dependency-aware wave execution (G2, Phase C).

Deterministic — FAKE worker subprocess (no LLM). Proves I5: a dependent task's
isolated worktree contains its prerequisite's *file changes* (because the prior
wave was merged to main before the next wave's worktrees were created).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rune.agent.parallel_isolated import WorkerOutcome
from rune.agent.wave_orchestrator import (
    WaveTask,
    _compute_waves,
    execute_waves,
    summarize_verification,
)

# Fake worker ops via goal:
#   "WRITE <rel> <content>"        -> write file in cwd (worktree)
#   "PROVE <readrel> <outrel>"     -> read readrel, write outrel="saw:"+content
#   "FAIL"                         -> exit 1 (no result)
_FAKE = r"""
import json, sys, os
spec = json.load(open(sys.argv[1])); res = sys.argv[2]
g = spec["goal"].split(" ")
def done(ans="done"):
    json.dump({"ok": True, "answer": ans, "iterations": 1, "actions": 1,
               "trace_reason": "completed", "error": ""}, open(res, "w"))
if g[0] == "FAIL":
    sys.exit(1)
if g[0] == "WRITE":
    open(g[1], "w").write(g[2]); done()
elif g[0] == "PROVE":
    seen = open(g[1]).read() if os.path.exists(g[1]) else "MISSING"
    open(g[2], "w").write("saw:" + seen); done("saw:" + seen)
"""


def _fake_cmd(spec_path, result_path, ws):
    return [sys.executable, "-c", _FAKE, spec_path, result_path]


def _git_repo(path: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    (path / "seed.txt").write_text("seed")
    subprocess.run(["git", "add", "."], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True, env=env)


class TestWaveComputation:
    def test_levels(self):
        tasks = [WaveTask("a", "g"), WaveTask("b", "g", ["a"]),
                 WaveTask("c", "g", ["a"]), WaveTask("d", "g", ["b", "c"])]
        waves = _compute_waves(tasks)
        assert [sorted(t.id for t in w) for w in waves] == [["a"], ["b", "c"], ["d"]]

    def test_cycle_detected(self):
        tasks = [WaveTask("a", "g", ["b"]), WaveTask("b", "g", ["a"])]
        assert _compute_waves(tasks) is None

    def test_unknown_dependency(self):
        assert _compute_waves([WaveTask("a", "g", ["ghost"])]) is None


class TestWaveExecution:
    def test_dependent_sees_prerequisite_files_i5(self, tmp_dir):
        """The core invariant: B (deps=[A]) sees A's merged file."""
        import asyncio
        _git_repo(tmp_dir)
        tasks = [
            WaveTask("A", "WRITE data.txt FROM_A"),
            WaveTask("B", "PROVE data.txt proof.txt", dependencies=["A"]),
        ]
        res = asyncio.run(execute_waves(
            str(tmp_dir), tasks, isolation="worktree", cmd_builder=_fake_cmd))
        assert res.ok
        assert res.waves_run == 2
        assert (tmp_dir / "data.txt").read_text() == "FROM_A"
        # B saw A's file because A was merged before B's worktree was created.
        assert (tmp_dir / "proof.txt").read_text() == "saw:FROM_A"

    def test_independent_tasks_one_wave(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        tasks = [WaveTask("A", "WRITE a.txt AAA"),
                 WaveTask("B", "WRITE b.txt BBB")]
        res = asyncio.run(execute_waves(
            str(tmp_dir), tasks, isolation="worktree", cmd_builder=_fake_cmd))
        assert res.ok
        assert res.waves_run == 1  # both independent → single wave
        assert (tmp_dir / "a.txt").read_text() == "AAA"
        assert (tmp_dir / "b.txt").read_text() == "BBB"

    def test_cycle_fails_cleanly(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        tasks = [WaveTask("a", "WRITE x x", ["b"]), WaveTask("b", "WRITE y y", ["a"])]
        res = asyncio.run(execute_waves(
            str(tmp_dir), tasks, isolation="worktree", cmd_builder=_fake_cmd))
        assert not res.ok
        assert "cycle" in res.reason

    def test_conflict_in_wave_stops_failclosed(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        # two independent tasks create the same file differently → wave merge fails
        tasks = [WaveTask("A", "WRITE clash.txt one"),
                 WaveTask("B", "WRITE clash.txt two")]
        res = asyncio.run(execute_waves(
            str(tmp_dir), tasks, isolation="worktree", cmd_builder=_fake_cmd))
        assert not res.ok
        assert res.failed_wave == 0
        assert not (tmp_dir / "clash.txt").exists()  # I3: nothing applied


class TestPostMergeGate:
    """I6 Tier-2: post-merge compile/test gate on the final assembled tree."""

    def test_final_wave_check_fail_rolls_back_only_final(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        tasks = [WaveTask("A", "WRITE data.txt FROM_A"),
                 WaveTask("B", "PROVE data.txt proof.txt", dependencies=["A"])]
        res = asyncio.run(execute_waves(
            str(tmp_dir), tasks, isolation="worktree", cmd_builder=_fake_cmd,
            post_merge_check="exit 1"))           # "compile" fails on assembled tree
        assert not res.ok
        assert res.failed_wave == 1               # the final wave
        assert "post-merge" in res.reason
        assert not (tmp_dir / "proof.txt").exists()   # I4: final wave rolled back
        assert (tmp_dir / "data.txt").read_text() == "FROM_A"  # earlier wave intact

    def test_intermediate_waves_not_checked(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        tasks = [WaveTask("A", "WRITE data.txt FROM_A"),
                 WaveTask("B", "PROVE data.txt proof.txt", dependencies=["A"])]
        # The check requires proof.txt, produced ONLY by the final wave. If it
        # were (wrongly) applied to wave 0 that merge would fail; it passing
        # proves only the final wave is gated.
        res = asyncio.run(execute_waves(
            str(tmp_dir), tasks, isolation="worktree", cmd_builder=_fake_cmd,
            post_merge_check="test -f proof.txt"))
        assert res.ok
        assert res.waves_run == 2
        assert (tmp_dir / "proof.txt").read_text() == "saw:FROM_A"


class TestVerificationMetrics:
    """§8: self-report-vs-deterministic mismatch + escalation-recovery stats."""

    def test_summarize_verification(self):
        outs = {
            "a": WorkerOutcome("a", result={"ok": True}, ok=True, escalation_step=0),
            "b": WorkerOutcome("b", result={"ok": True}, ok=True, escalation_step=2),
            # self-reported ok=True but deterministically rejected → the mismatch we measure
            "c": WorkerOutcome("c", result={"ok": True}, ok=False),
            # genuinely crashed (self-reported not-ok) → rejected but NOT a mismatch
            "d": WorkerOutcome("d", result={"ok": False}, ok=False),
        }
        st = summarize_verification(outs)
        assert (st.total, st.accepted, st.rejected) == (4, 2, 2)
        assert st.self_report_mismatch == 1            # only c
        assert st.escalation_recoveries == 1           # b recovered at step 2
        assert st.by_step == {0: 1, 2: 1}
