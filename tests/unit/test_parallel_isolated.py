"""Tests for parallel isolated workers + merge (G2, Phase B).

Deterministic — uses a trivial FAKE worker subprocess (no LLM) to exercise the
create -> subprocess(isolated cwd) -> collect -> atomic merge pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rune.agent.parallel_isolated import (
    WorkerSpec,
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
