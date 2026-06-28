"""Tests for parallel isolated workers + merge (G2, Phase B).

Deterministic — uses a trivial FAKE worker subprocess (no LLM) to exercise the
create -> subprocess(isolated cwd) -> collect -> atomic merge pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rune.agent.acceptance import Acceptance
from rune.agent.parallel_isolated import (
    Escalation,
    WorkerSpec,
    resolve_escalation,
    run_isolated_worker,
    run_wave_and_merge,
)

# Fake worker: goal = "WRITE <relpath> <content>". Writes the file in cwd (its
# worktree) and records the isolation env into the result so the test can verify
# the subprocess was correctly isolated. goal = "FAIL" => no result (failure).
_FAKE = r"""
import json, sys, os
spec = json.load(open(sys.argv[1])); res = sys.argv[2]
goal = spec["goal"]
if goal == "FAIL":
    sys.exit(1)  # no result file written -> worker failed
parts = goal.split(" ", 2)
rel, content = parts[1], parts[2]
with open(rel, "w") as fh:      # relative -> lands in cwd (the worktree)
    fh.write(content)
json.dump({"ok": True, "answer": "done", "iterations": 1, "actions": 1,
           "trace_reason": "completed", "error": "",
           "iso_root": os.environ.get("RUNE_ISOLATION_ROOT", ""),
           "is_worker": os.environ.get("RUNE_WORKER", "")},
          open(res, "w"))
"""


def _fake_cmd(spec_path, result_path, ws):
    return [sys.executable, "-c", _FAKE, spec_path, result_path]


# Silent no-op worker: SELF-reports ok=True but writes no file and logs to stderr.
# Exercises I6 (deterministic gate must reject) + I8 (stderr captured).
_FAKE_NOOP = r"""
import json, sys
print("worker ran but wrote nothing", file=sys.stderr)
json.dump({"ok": True, "answer": "", "iterations": 0, "actions": 0,
           "trace_reason": "completed", "error": ""}, open(sys.argv[2], "w"))
"""


def _noop_cmd(spec_path, result_path, ws):
    return [sys.executable, "-c", _FAKE_NOOP, spec_path, result_path]


# Recovers on the step-1 feedback retry: writes a file only once the goal carries
# the escalation directive (i.e. on a retry), self-reporting ok throughout.
_FAKE_RECOVER_ON_RETRY = r"""
import json, sys
spec = json.load(open(sys.argv[1]))
if "RUNE-ESCALATION" in spec["goal"]:
    open("out.txt", "w").write("recovered")
json.dump({"ok": True, "answer": "", "iterations": 1, "actions": 1,
           "trace_reason": "completed", "error": ""}, open(sys.argv[2], "w"))
"""


# Recovers only on the stronger model (step >=2): writes a file iff provider=strong.
_FAKE_RECOVER_ON_STRONG = r"""
import json, sys
spec = json.load(open(sys.argv[1]))
if spec.get("provider") == "strong":
    open("out.txt", "w").write("by-strong")
json.dump({"ok": True, "answer": "", "iterations": 1, "actions": 1,
           "trace_reason": "completed", "error": ""}, open(sys.argv[2], "w"))
"""


def _retry_cmd(spec_path, result_path, ws):
    return [sys.executable, "-c", _FAKE_RECOVER_ON_RETRY, spec_path, result_path]


def _strong_cmd(spec_path, result_path, ws):
    return [sys.executable, "-c", _FAKE_RECOVER_ON_STRONG, spec_path, result_path]


def _git_repo(path: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    (path / "seed.txt").write_text("seed")
    subprocess.run(["git", "add", "."], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True, env=env)


class TestSingleWorker:
    def test_isolated_run_collects_and_env(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        spec = WorkerSpec("w0", goal="WRITE out.txt hello")
        outcome, ws = asyncio.run(run_isolated_worker(
            str(tmp_dir), spec, isolation="worktree", cmd_builder=_fake_cmd))
        try:
            assert outcome.ok
            # subprocess saw the isolation env, pointing at its worktree
            assert outcome.result["iso_root"] == ws.path
            assert outcome.result["is_worker"] == "1"
            # change captured; file is in the worktree, NOT the main tree
            assert "out.txt" in outcome.changeset.changes
            assert (Path(ws.path) / "out.txt").read_text() == "hello"
            assert not (tmp_dir / "out.txt").exists()  # main untouched pre-merge
        finally:
            from rune.agent import worktree
            asyncio.run(worktree.cleanup(ws))


class TestParallelWaveMerge:
    def test_disjoint_workers_merge_to_main(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        specs = [WorkerSpec("w1", "WRITE a.txt AAA"),
                 WorkerSpec("w2", "WRITE b.txt BBB")]
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), specs, isolation="worktree", cmd_builder=_fake_cmd))
        assert res.merge.ok
        assert (tmp_dir / "a.txt").read_text() == "AAA"
        assert (tmp_dir / "b.txt").read_text() == "BBB"

    def test_conflict_is_atomic(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        # both create the SAME new file with different content -> conflict
        specs = [WorkerSpec("w1", "WRITE shared.txt content-from-1"),
                 WorkerSpec("w2", "WRITE shared.txt content-from-2")]
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), specs, isolation="worktree", cmd_builder=_fake_cmd))
        assert not res.merge.ok
        # I3 atomicity: nothing applied to main
        assert not (tmp_dir / "shared.txt").exists()

    def test_failed_worker_discarded_others_merge(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        specs = [WorkerSpec("w1", "WRITE ok.txt good"),
                 WorkerSpec("w2", "FAIL")]
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), specs, isolation="worktree", cmd_builder=_fake_cmd))
        assert res.merge.ok
        assert (tmp_dir / "ok.txt").read_text() == "good"
        # the failed worker contributed nothing
        assert sum(1 for w in res.workers if not w.ok) == 1

    def test_copy_mode_non_git(self, tmp_dir):
        import asyncio
        (tmp_dir / "seed.txt").write_text("seed")  # no git
        specs = [WorkerSpec("w1", "WRITE c.txt CCC")]
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), specs, isolation="auto", cmd_builder=_fake_cmd))
        assert res.merge.ok
        assert (tmp_dir / "c.txt").read_text() == "CCC"


class TestAcceptanceGate:
    """I6 deterministic worker acceptance + I8 log capture (Phase 1)."""

    def test_noop_worker_rejected_despite_self_report_ok(self, tmp_dir):
        import asyncio

        from rune.agent import worktree
        _git_repo(tmp_dir)
        outcome, ws = asyncio.run(run_isolated_worker(
            str(tmp_dir), WorkerSpec("w0", goal="x"),
            isolation="worktree", cmd_builder=_noop_cmd))
        try:
            assert outcome.result.get("ok") is True   # worker SELF-reports success
            assert outcome.ok is False                 # deterministic gate overrules it
            assert "no file change" in outcome.acceptance_reason
            assert outcome.changeset is None           # nothing offered to the merge
            assert "wrote nothing" in outcome.log_tail  # I8: stderr captured, not DEVNULL
        finally:
            asyncio.run(worktree.cleanup(ws))

    def test_noop_worker_contributes_nothing_to_wave(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "x")],
            isolation="worktree", cmd_builder=_noop_cmd))
        assert res.merge.ok                            # merging zero changesets is trivially ok
        assert res.workers[0].ok is False
        assert not (tmp_dir / "x").exists()

    def test_expect_paths_mismatch_rejected(self, tmp_dir):
        import asyncio

        from rune.agent import worktree
        _git_repo(tmp_dir)
        spec = WorkerSpec("w0", goal="WRITE out.txt hi",
                          acceptance=Acceptance(expect_paths=["wanted.txt"]))
        outcome, ws = asyncio.run(run_isolated_worker(
            str(tmp_dir), spec, isolation="worktree", cmd_builder=_fake_cmd))
        try:
            assert outcome.ok is False                 # wrote out.txt, not wanted.txt
            assert "wanted.txt" in outcome.acceptance_reason
        finally:
            asyncio.run(worktree.cleanup(ws))

    def test_expect_paths_match_accepted(self, tmp_dir):
        import asyncio

        from rune.agent import worktree
        _git_repo(tmp_dir)
        spec = WorkerSpec("w0", goal="WRITE wanted.txt hi",
                          acceptance=Acceptance(expect_paths=["wanted.txt"]))
        outcome, ws = asyncio.run(run_isolated_worker(
            str(tmp_dir), spec, isolation="worktree", cmd_builder=_fake_cmd))
        try:
            assert outcome.ok is True
            assert "wanted.txt" in outcome.changeset.changes
        finally:
            asyncio.run(worktree.cleanup(ws))


class TestPostMergeGate:
    """I6 Tier-2: deterministic compile/test gate on the MERGED tree (Phase 2)."""

    def test_post_merge_pass_keeps_merge(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "WRITE a.txt AAA")],
            isolation="worktree", cmd_builder=_fake_cmd,
            post_merge_check="grep -q AAA a.txt"))
        assert res.merge.ok
        assert (tmp_dir / "a.txt").read_text() == "AAA"

    def test_post_merge_fail_rolls_back(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "WRITE a.txt AAA")],
            isolation="worktree", cmd_builder=_fake_cmd,
            post_merge_check="exit 1"))      # the "compile/test" fails
        assert not res.merge.ok
        assert "post-merge check" in res.merge.reason
        assert not (tmp_dir / "a.txt").exists()   # I4: merge rolled back
        # Tier-1 vs Tier-2: the worker DID its job (changed a file); the merged
        # tree just didn't pass the build gate.
        assert res.workers[0].ok


class TestEscalationLadder:
    """I7 escalation: retry rejected workers (feedback → stronger model), bounded."""

    def test_step1_feedback_retry_recovers(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "WRITE out.txt X")],
            isolation="worktree", cmd_builder=_retry_cmd, max_escalation_steps=2))
        assert res.merge.ok
        assert (tmp_dir / "out.txt").read_text() == "recovered"
        assert res.workers[0].ok
        assert res.workers[0].escalation_step == 1   # recovered on the feedback retry

    def test_step2_stronger_model_recovers(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "x")],
            isolation="worktree", cmd_builder=_strong_cmd, max_escalation_steps=2,
            escalation=Escalation(provider="strong", model="big")))
        assert res.merge.ok
        assert (tmp_dir / "out.txt").read_text() == "by-strong"
        assert res.workers[0].ok
        assert res.workers[0].escalation_step == 2   # needed the stronger model

    def test_ladder_exhausted_fail_closed(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        # never produces a file; no stronger model configured → step-2 unavailable.
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "x")],
            isolation="worktree", cmd_builder=_noop_cmd, max_escalation_steps=2))
        assert res.merge.ok                          # zero changesets merge cleanly
        assert res.workers[0].ok is False            # stays rejected, not silent-ok
        assert res.workers[0].escalation_step == 1   # tried step-1, step-2 had no model
        assert not (tmp_dir / "x").exists()

    def test_no_escalation_when_steps_zero(self, tmp_dir):
        import asyncio
        _git_repo(tmp_dir)
        # default (max_escalation_steps=0): no retry even though it would recover.
        res = asyncio.run(run_wave_and_merge(
            str(tmp_dir), [WorkerSpec("w1", "x")],
            isolation="worktree", cmd_builder=_retry_cmd))
        assert res.workers[0].ok is False
        assert res.workers[0].escalation_step == 0


class TestResolveEscalation:
    def test_none_when_no_provider(self):
        assert resolve_escalation(None) is None
        assert resolve_escalation("") is None

    def test_explicit_model_used_verbatim(self):
        assert resolve_escalation("anthropic", "claude-x") == Escalation(
            provider="anthropic", model="claude-x")


class TestRealEntryImportable:
    def test_worker_proc_imports_and_argparse(self):
        # The real entry must at least import + reject a bad spec cleanly.
        import json
        import tempfile

        from rune.agent import worker_proc
        d = tempfile.mkdtemp()
        res = os.path.join(d, "r.json")
        rc = worker_proc.main(["--spec", "/no/such/spec.json", "--result", res])
        assert rc == 1
        assert json.load(open(res))["ok"] is False
