"""Tests for CLI best-of-K (verifier-guided rejection sampling) wiring.

Covers the new ``rune --message ... --best-of K`` path: subprocess attempt
isolation + recursion guard, verifier-driven selection, file-restore on success,
no-restore on none-pass, and that K==1 leaves the single-attempt path unchanged.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

import rune.cli.best_of as best_of
from rune.cli.best_of import (
    AttemptArtifact,
    _best_of_async,
    _changed_vs_seed,
    _preserve_skipped,
    _restore_artifacts,
    _restore_changed,
    _run_attempt_subprocess,
    _seed_workdir,
    _tree_manifest,
)
from rune.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_record_winner(monkeypatch):
    """Stub winner recording by default so _best_of_async tests don't
    hit the real memory/learning pipeline. Tests that assert on it can read the
    returned mock."""
    m = AsyncMock(return_value=True)
    monkeypatch.setattr(best_of, "_record_winner", m)
    return m


# --- file-restore -----------------------------------------------------------


def test_restore_artifacts_copies_files_and_dirs(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "calc.py").write_text("print(1)")
    sub = work / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("x = 1")

    dest = tmp_path / "dest"
    dest.mkdir()

    copied, skipped = _restore_artifacts(str(work), str(dest), ["calc.py", "pkg"])

    assert sorted(copied) == ["calc.py", "pkg"]
    assert skipped == []
    assert (dest / "calc.py").read_text() == "print(1)"
    assert (dest / "pkg" / "mod.py").read_text() == "x = 1"


def test_restore_artifacts_never_overwrites_existing(tmp_path):
    # A file with the same name already exists in dest — it must NOT be clobbered.
    work = tmp_path / "work"
    work.mkdir()
    (work / "keep.py").write_text("from attempt")
    (work / "new.py").write_text("brand new")

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.py").write_text("USER ORIGINAL")

    copied, skipped = _restore_artifacts(str(work), str(dest), ["keep.py", "new.py"])

    assert copied == ["new.py"]
    assert skipped == ["keep.py"]
    # user's file is preserved untouched
    assert (dest / "keep.py").read_text() == "USER ORIGINAL"
    assert (dest / "new.py").read_text() == "brand new"


def test_preserve_skipped_saves_winner(tmp_path):
    # A colliding winner file must be saved, not discarded.
    work = tmp_path / "work"
    work.mkdir()
    (work / "solution.py").write_text("WINNER")

    dest = tmp_path / "dest"
    dest.mkdir()

    preserved = _preserve_skipped(str(work), str(dest), ["solution.py"])

    assert preserved is not None
    assert os.path.basename(preserved).startswith(".rune-bestof-")
    assert os.path.dirname(preserved) == str(dest)
    assert open(os.path.join(preserved, "solution.py")).read() == "WINNER"


def test_preserve_skipped_none_when_no_collision(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    assert _preserve_skipped(str(work), str(dest), []) is None


def test_restore_artifacts_only_restores_snapshot(tmp_path):
    # A verifier byproduct (__pycache__) appears in the workdir AFTER the
    # snapshot was taken — it must NOT be restored.
    work = tmp_path / "work"
    work.mkdir()
    (work / "solution.py").write_text("x = 1")
    (work / "__pycache__").mkdir()
    (work / "__pycache__" / "solution.cpython.pyc").write_text("junk")

    dest = tmp_path / "dest"
    dest.mkdir()

    # snapshot taken before verification only knew about solution.py
    copied, skipped = _restore_artifacts(str(work), str(dest), ["solution.py"])

    assert copied == ["solution.py"]
    assert skipped == []
    assert (dest / "solution.py").exists()
    assert not (dest / "__pycache__").exists()


def test_snapshot_excludes_cache_dirs(tmp_path):
    from rune.cli.best_of import _snapshot_produced

    work = tmp_path / "work"
    work.mkdir()
    (work / "solution.py").write_text("x = 1")
    (work / "__pycache__").mkdir()
    (work / ".pytest_cache").mkdir()

    assert _snapshot_produced(str(work)) == ["solution.py"]


# --- seeded mode (--include-cwd): seed / diff / restore-changed -------------


def test_seed_workdir_copies_tree_minus_cruft(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "app.py").write_text("a")
    (src / "pkg" / "mod.py").write_text("m")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_text("bytes")

    work = tmp_path / "work"
    _seed_workdir(str(src), str(work))

    assert (work / "app.py").read_text() == "a"
    assert (work / "pkg" / "mod.py").read_text() == "m"
    # VCS/cache cruft excluded
    assert not (work / ".git").exists()
    assert not (work / "__pycache__").exists()


def test_changed_vs_seed_detects_new_and_modified(tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (work / "keep.py").write_text("unchanged")
    (work / "sub" / "edit.py").write_text("v1")

    seed = _tree_manifest(str(work))

    # modify one, add one, leave one untouched
    (work / "sub" / "edit.py").write_text("v2 longer content")
    (work / "new.py").write_text("brand new")

    changed = _changed_vs_seed(str(work), seed)
    assert changed == sorted([os.path.join("sub", "edit.py"), "new.py"])
    assert "keep.py" not in changed


def test_restore_changed_overwrites_with_backup(tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (work / "sub" / "edit.py").write_text("NEW VERSION")
    (work / "fresh.py").write_text("fresh")

    dest = tmp_path / "dest"
    (dest / "sub").mkdir(parents=True)
    (dest / "sub" / "edit.py").write_text("OLD VERSION")

    restored, backup_dir = _restore_changed(
        str(work), str(dest), [os.path.join("sub", "edit.py"), "fresh.py"]
    )

    assert sorted(restored) == sorted([os.path.join("sub", "edit.py"), "fresh.py"])
    # overwrite happened (intended)
    assert (dest / "sub" / "edit.py").read_text() == "NEW VERSION"
    assert (dest / "fresh.py").read_text() == "fresh"
    # original backed up for undo
    assert backup_dir is not None
    assert open(os.path.join(backup_dir, "sub", "edit.py")).read() == "OLD VERSION"
    # nothing to back up for the brand-new file
    assert not os.path.exists(os.path.join(backup_dir, "fresh.py"))


def test_seed_footprint_counts_and_ignores(tmp_path):
    from rune.cli.best_of import _seed_footprint

    (tmp_path / "a.py").write_text("x" * 100)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "big.js").write_text("y" * 10000)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "obj").write_text("z" * 10000)

    files, total = _seed_footprint(str(tmp_path))
    assert files == 1  # only a.py; node_modules/.git ignored
    assert total == 100


def test_check_seed_size_aborts_when_too_large(tmp_path, monkeypatch):
    from rune.cli.best_of import _check_seed_size

    monkeypatch.setenv("RUNE_BESTOF_SEED_MAX_MB", "1")
    (tmp_path / "big.bin").write_text("x" * (2 * 1024 * 1024))  # 2 MB > 1 MB limit
    msg = _check_seed_size(str(tmp_path))
    assert msg is not None and "include-cwd" in msg


def test_check_seed_size_ok_for_small(tmp_path, monkeypatch):
    from rune.cli.best_of import _check_seed_size

    monkeypatch.setenv("RUNE_BESTOF_SEED_MAX_MB", "200")
    (tmp_path / "small.py").write_text("x")
    assert _check_seed_size(str(tmp_path)) is None


def test_changed_vs_seed_warns_on_all_changed(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (work / "a.py").write_text("1")
    (work / "b.py").write_text("22")

    warnings: list = []
    monkeypatch.setattr(best_of.log, "warning", lambda ev, **kw: warnings.append(ev))

    # Seed claims both files but with stale (mtime,size) → ALL appear changed,
    # which signals a broken diff and must warn.
    stale_seed = {"a.py": (0.0, 999), "b.py": (0.0, 999)}
    changed = _changed_vs_seed(str(work), stale_seed)

    assert sorted(changed) == ["a.py", "b.py"]
    assert "bestof_seed_diff_suspicious" in warnings


def test_changed_vs_seed_no_warn_on_partial_change(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (work / "a.py").write_text("1")
    (work / "b.py").write_text("22")
    st_a = os.stat(work / "a.py")
    # a.py matches seed exactly (unchanged); only b.py differs → no warning.
    seed = {"a.py": (st_a.st_mtime, st_a.st_size), "b.py": (0.0, 999)}

    warnings: list = []
    monkeypatch.setattr(best_of.log, "warning", lambda ev, **kw: warnings.append(ev))
    changed = _changed_vs_seed(str(work), seed)

    assert changed == ["b.py"]
    assert "bestof_seed_diff_suspicious" not in warnings


@pytest.mark.asyncio
async def test_best_of_seeded_writes_back_edits(monkeypatch, tmp_path):
    # cwd has an existing file; the seeded attempt "edits" it; restore overwrites
    # with a backup, since seed_cwd=True (no skip/preserve).
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "app.py").write_text("ORIGINAL")

    def fake_seed(src, workdir):
        # emulate seeding: copy app.py in, as the real _seed_workdir would
        import shutil as _sh
        _sh.copy2(os.path.join(src, "app.py"), os.path.join(workdir, "app.py"))

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / "w0"
        w.mkdir()
        fake_seed(seed_from, str(w))
        # the agent edits app.py
        (w / "app.py").write_text("FIXED BY AGENT")
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="done", returncode=0, produced=["app.py"]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return True

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix app", 1, None, None, report=lambda s, **kw: reports.append(kw), seed_cwd=True
    )

    assert code == 0
    assert (dest / "app.py").read_text() == "FIXED BY AGENT"  # overwritten
    kw = reports[0]
    assert kw["copied"] == ["app.py"]
    assert kw["backup_dir"] is not None
    # original preserved for undo
    assert open(os.path.join(kw["backup_dir"], "app.py")).read() == "ORIGINAL"


# --- attempt subprocess: recursion guard + no --best-of leak ----------------


@pytest.mark.asyncio
async def test_run_attempt_sets_guard_and_omits_best_of(monkeypatch):
    captured: dict = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"hello\n", b"")

    async def fake_exec(*cmd, cwd=None, env=None, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["env"] = env
        return _FakeProc()

    monkeypatch.setattr(best_of.asyncio, "create_subprocess_exec", fake_exec)

    art = await _run_attempt_subprocess(2, "do it", model="m1", provider="p1")

    assert art.index == 2
    assert art.stdout == "hello\n"
    assert art.returncode == 0
    assert isinstance(art.produced, list)
    # recursion guard set in the child env
    assert captured["env"][best_of.RECURSION_GUARD_ENV] == "1"
    # child runs the plain single-attempt path: --best-of must NOT be propagated
    assert "--best-of" not in captured["cmd"]
    # but model/provider/message ARE propagated
    assert "--message" in captured["cmd"] and "do it" in captured["cmd"]
    assert "--model" in captured["cmd"] and "m1" in captured["cmd"]
    assert "--provider" in captured["cmd"] and "p1" in captured["cmd"]
    # isolated workdir (a real temp dir)
    assert os.path.isdir(captured["cwd"])


# --- best-of core: selection + restore --------------------------------------


@pytest.mark.asyncio
async def test_best_of_selects_passing_and_restores(monkeypatch, tmp_path):
    # Three attempt workdirs; only #1 contains a "good" artifact.
    works = []
    for i in range(3):
        w = tmp_path / f"w{i}"
        w.mkdir()
        if i == 1:
            (w / "answer.txt").write_text("correct")
        works.append(str(w))

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=index,
            workdir=works[index],
            stdout=f"out{index}",
            returncode=0,
            produced=sorted(os.listdir(works[index])),
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            # passes only the attempt whose workdir has answer.txt
            return os.path.exists(os.path.join(cwd, "answer.txt"))

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    # don't delete the temp workdirs we assert on
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)

    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []

    def report(stdout, **kw):
        reports.append((stdout, kw))

    code = await _best_of_async("task", 3, None, None, report=report)

    assert code == 0
    stdout, kw = reports[0]
    assert kw["solved"] is True
    assert kw["selected_index"] == 1
    assert kw["copied"] == ["answer.txt"]
    # artifact restored into the real cwd
    assert (dest / "answer.txt").read_text() == "correct"


@pytest.mark.asyncio
async def test_best_of_none_pass_no_restore(monkeypatch, tmp_path):
    works = []
    for i in range(2):
        w = tmp_path / f"w{i}"
        w.mkdir()
        (w / "wrong.txt").write_text("nope")
        works.append(str(w))

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=index,
            workdir=works[index],
            stdout=f"out{index}",
            returncode=0,
            produced=sorted(os.listdir(works[index])),
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False  # nothing passes

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)

    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []

    def report(stdout, **kw):
        reports.append((stdout, kw))

    code = await _best_of_async("task", 2, None, None, report=report)

    assert code == 1
    stdout, kw = reports[0]
    assert kw["solved"] is False
    assert kw["copied"] == []
    assert stdout == "out0"  # first attempt surfaced as best-effort
    # F: both attempts wrote files (wrong.txt) but failed → no_artifact == 0
    assert kw["has_check"] is True
    assert kw["no_artifact"] == 0
    # nothing restored
    assert not (dest / "wrong.txt").exists()


@pytest.mark.asyncio
async def test_best_of_preserves_winner_on_collision(monkeypatch, tmp_path):
    # Winner produces solution.py, but cwd already has one → restore skips it and
    # the winner is preserved (not discarded) so the K runs aren't wasted.
    w = tmp_path / "w0"
    w.mkdir()
    (w / "solution.py").write_text("WINNER CODE")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="o", returncode=0, produced=["solution.py"]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return True  # passes

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "solution.py").write_text("USER ORIGINAL")
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async("task", 1, None, None, report=lambda s, **kw: reports.append(kw))

    assert code == 0
    kw = reports[0]
    assert kw["copied"] == []
    assert kw["skipped"] == ["solution.py"]
    # user's file untouched
    assert (dest / "solution.py").read_text() == "USER ORIGINAL"
    # winner preserved in a side dir
    pres = kw["preserved_dir"]
    assert pres is not None
    assert open(os.path.join(pres, "solution.py")).read() == "WINNER CODE"


@pytest.mark.asyncio
async def test_best_of_reports_no_artifact_breakdown(monkeypatch, tmp_path):
    # Attempt 0 wrote a file; attempt 1 produced nothing (generator didn't write).
    w0 = tmp_path / "w0"
    w0.mkdir()
    (w0 / "out.txt").write_text("x")
    w1 = tmp_path / "w1"
    w1.mkdir()
    works = [str(w0), str(w1)]

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=index,
            workdir=works[index],
            stdout=f"out{index}",
            returncode=0,
            produced=sorted(os.listdir(works[index])),
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(tmp_path)

    reports: list = []
    code = await _best_of_async("task", 2, None, None, report=lambda s, **kw: reports.append(kw))

    assert code == 1
    assert reports[0]["no_artifact"] == 1  # only attempt 1 produced nothing


@pytest.mark.asyncio
async def test_best_of_reports_no_check(monkeypatch, tmp_path):
    # When the EG can't build a check, has_check propagates False so the report
    # can say "best-of-K cannot select" instead of "all candidates failed".
    w0 = tmp_path / "w0"
    w0.mkdir()
    (w0 / "out.txt").write_text("x")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=index, workdir=str(w0), stdout="o", returncode=0, produced=["out.txt"]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False

        verify.has_check = False  # no mechanical check available
        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(tmp_path)

    reports: list = []
    code = await _best_of_async("task", 1, None, None, report=lambda s, **kw: reports.append(kw))

    assert code == 1
    assert reports[0]["has_check"] is False


# --- learn a correctness rule from failed attempts -------------------------


@pytest.mark.asyncio
async def test_learn_from_failures_calls_crisp_learner(monkeypatch):
    import rune.memory.rule_learner as rl

    captured = {}

    async def fake_classify(msg):
        return SimpleNamespace(goal_type="code_modify")

    async def fake_learn(tool_name, error_message, domain):
        captured["tool"] = tool_name
        captured["ev"] = error_message
        captured["domain"] = domain
        return "rule_key_1"

    monkeypatch.setattr("rune.agent.goal_classifier.classify_goal", fake_classify)
    monkeypatch.setattr(rl, "learn_from_crisp_failure", fake_learn)

    keys = await best_of._learn_from_failures("fix calc", ["", "bad: -7/2=-3 exp -4", ""])
    assert keys == ["rule_key_1"]
    assert captured["ev"] == "bad: -7/2=-3 exp -4"  # the non-empty evidence
    assert captured["domain"] == "code_modify"


@pytest.mark.asyncio
async def test_learn_from_failures_noop_without_evidence(monkeypatch):
    import rune.memory.rule_learner as rl

    called = False

    async def fake_learn(*a, **k):
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr(rl, "learn_from_crisp_failure", fake_learn)
    keys = await best_of._learn_from_failures("fix", ["", "   ", ""])
    assert keys == []
    assert called is False


@pytest.mark.asyncio
async def test_learn_from_failures_learns_each_distinct(monkeypatch):
    # K attempts fail for DIFFERENT reasons (structural vs logic) — learn from
    # each distinct evidence (deduped), not just the first, so semantic retrieval
    # can later pick the relevant rule.
    import rune.memory.rule_learner as rl

    seen_ev = []

    async def fake_classify(msg):
        return SimpleNamespace(goal_type="code_modify")

    async def fake_learn(tool_name, error_message, domain):
        seen_ev.append(error_message)
        return f"k{len(seen_ev)}"

    monkeypatch.setattr("rune.agent.goal_classifier.classify_goal", fake_classify)
    monkeypatch.setattr(rl, "learn_from_crisp_failure", fake_learn)

    keys = await best_of._learn_from_failures(
        "fix calc",
        ["missing evaluate function", "-7/2 got -3 expected -4", "missing evaluate function", ""],
    )
    # two DISTINCT evidences → two rules; near-dup collapsed
    assert keys == ["k1", "k2"]
    assert seen_ev == ["missing evaluate function", "-7/2 got -3 expected -4"]


@pytest.mark.asyncio
async def test_learn_from_failures_caps_llm_calls(monkeypatch):
    import rune.memory.rule_learner as rl

    calls = 0

    async def fake_classify(msg):
        return SimpleNamespace(goal_type="code_modify")

    async def fake_learn(tool_name, error_message, domain):
        nonlocal calls
        calls += 1
        return f"k{calls}"

    monkeypatch.setattr("rune.agent.goal_classifier.classify_goal", fake_classify)
    monkeypatch.setattr(rl, "learn_from_crisp_failure", fake_learn)

    many = [f"distinct failure {i}" for i in range(10)]
    keys = await best_of._learn_from_failures("fix", many)
    assert calls == best_of._MAX_FAILURE_RULES  # capped
    assert len(keys) == best_of._MAX_FAILURE_RULES


@pytest.mark.asyncio
async def test_best_of_learns_from_failed_attempts_on_solve(monkeypatch, tmp_path):
    # Two attempts: #0 fails (has evidence), #1 passes → solve learns from #0.
    works = [str(tmp_path / f"w{i}") for i in range(2)]
    for w in works:
        os.makedirs(w)

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=index, workdir=works[index], stdout=f"o{index}", returncode=0,
            produced=[],
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return cwd == works[1]  # only #1 passes
        verify.evidence_by_cwd = {works[0]: "mismatch: -7/2=-3 exp -4"}
        return verify

    learned = {}

    async def fake_learn(message, evidence):
        learned["evidence"] = evidence
        return "k"

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_record_winner", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_learn_from_failures", fake_learn)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.chdir(dest)

    code = await _best_of_async("fix calc", 2, None, None, report=lambda s, **kw: None)
    assert code == 0
    # the failed attempt #0's evidence was passed to learning
    assert learned["evidence"] == ["mismatch: -7/2=-3 exp -4"]


# --- winner recording ------------------------------------------------------


@pytest.mark.asyncio
async def test_records_winner_on_solve(monkeypatch, tmp_path, _mock_record_winner):
    w = tmp_path / "w0"
    w.mkdir()
    (w / "answer.txt").write_text("ok")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="WIN", returncode=0, produced=["answer.txt"]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return True

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.chdir(dest)

    await _best_of_async("solve this", 1, None, None, report=lambda s, **kw: None)

    _mock_record_winner.assert_awaited_once()
    # records the task message + the winner's output
    assert _mock_record_winner.call_args.args[0] == "solve this"
    assert _mock_record_winner.call_args.args[1] == "WIN"


@pytest.mark.asyncio
async def test_no_record_on_none_pass(monkeypatch, tmp_path, _mock_record_winner):
    w = tmp_path / "w0"
    w.mkdir()

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="x", returncode=0, produced=[]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False  # nothing passes

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(tmp_path)

    await _best_of_async("task", 1, None, None, report=lambda s, **kw: None)

    _mock_record_winner.assert_not_awaited()


# --- CLI dispatch: k==1 unchanged, k>1 fans out, recursion guard ------------


def test_cli_best_of_1_uses_single_path():
    with (
        patch("rune.cli.main._handle_non_interactive") as single,
        patch("rune.cli.best_of.run_best_of") as bestof,
        patch("rune.cli.main._ensure_llm_key", return_value=True),
    ):
        result = runner.invoke(app, ["--message", "hi"])
    assert result.exit_code == 0
    single.assert_called_once()
    bestof.assert_not_called()


def test_cli_best_of_k_dispatches():
    with (
        patch("rune.cli.main._handle_non_interactive") as single,
        patch("rune.cli.best_of.run_best_of") as bestof,
        patch("rune.cli.main._ensure_llm_key", return_value=True),
    ):
        result = runner.invoke(app, ["--message", "hi", "--best-of", "3"])
    assert result.exit_code == 0
    bestof.assert_called_once()
    assert bestof.call_args.args[1] == 3  # k
    assert bestof.call_args.kwargs.get("seed_cwd") is False  # default greenfield
    single.assert_not_called()


def test_cli_include_cwd_sets_seed():
    with (
        patch("rune.cli.main._handle_non_interactive"),
        patch("rune.cli.best_of.run_best_of") as bestof,
        patch("rune.cli.main._ensure_llm_key", return_value=True),
    ):
        result = runner.invoke(app, ["--message", "hi", "--best-of", "2", "--include-cwd"])
    assert result.exit_code == 0
    bestof.assert_called_once()
    assert bestof.call_args.kwargs.get("seed_cwd") is True


def test_cli_recursion_guard_collapses_to_single(monkeypatch):
    monkeypatch.setenv("RUNE_IN_BEST_OF", "1")
    with (
        patch("rune.cli.main._handle_non_interactive") as single,
        patch("rune.cli.best_of.run_best_of") as bestof,
        patch("rune.cli.main._ensure_llm_key", return_value=True),
    ):
        result = runner.invoke(app, ["--message", "hi", "--best-of", "3"])
    assert result.exit_code == 0
    single.assert_called_once()
    bestof.assert_not_called()


# --- C: per-attempt timeout -------------------------------------------------


@pytest.mark.asyncio
async def test_attempt_times_out_and_is_killed(monkeypatch):
    monkeypatch.setenv("RUNE_BESTOF_ATTEMPT_TIMEOUT_MS", "50")
    killed = {"v": False}

    class _HangProc:
        returncode = None

        async def communicate(self):
            await asyncio.sleep(5)  # longer than the 50ms timeout
            return (b"", b"")

        def kill(self):
            killed["v"] = True

        async def wait(self):
            return 0

    async def fake_exec(*cmd, **kwargs):
        return _HangProc()

    monkeypatch.setattr(best_of.asyncio, "create_subprocess_exec", fake_exec)

    art = await _run_attempt_subprocess(0, "msg", None, None)

    assert art.returncode == best_of._TIMEOUT_RETURNCODE
    assert killed["v"] is True
    assert art.stdout == ""


# --- D: concurrency cap -----------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_capped(monkeypatch, tmp_path):
    # Force cap = min(k, cpu-2) = 1 by reporting 3 cpus.
    monkeypatch.setattr(best_of.os, "cpu_count", lambda: 3)

    live = {"now": 0, "max": 0}

    async def fake_attempt(index, message, model, provider, seed_from=None):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.02)
        live["now"] -= 1
        w = tmp_path / f"w{index}"
        w.mkdir()
        return AttemptArtifact(
            index=index, workdir=str(w), stdout="o", returncode=0, produced=[]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(tmp_path)

    await _best_of_async("task", 3, None, None, report=lambda s, **kw: None)

    assert live["max"] == 1  # never more than the cap ran at once


# --- A: best-of attempt subprocess is a throwaway (no memory side-effects) ---


def _run_non_interactive_with_mocks(monkeypatch):
    """Invoke _handle_non_interactive with all heavy deps mocked; return the
    post_process and promote spies."""
    import rune.agent.agent_context as agent_context
    import rune.agent.goal_classifier as goal_classifier
    import rune.agent.loop as loop_mod
    import rune.mcp.config as mcp_config
    import rune.memory.manager as mem_manager

    monkeypatch.setattr("rune.cli.main._ensure_llm_key", lambda: True)
    monkeypatch.setattr("rune.cli.main._wire_cli_approval", lambda loop: None)

    class _FakeLoop:
        files_written: list = []

        def __init__(self, *a, **k):
            pass

        def on(self, *a, **k):
            pass

        def set_approval_callback(self, *a, **k):
            pass

        async def run(self, *a, **k):
            return SimpleNamespace(reason="completed", evidence_gate=None)

    monkeypatch.setattr(loop_mod, "NativeAgentLoop", _FakeLoop)
    monkeypatch.setattr(mcp_config, "load_mcp_config", lambda: [])

    async def fake_prepare(opts, **k):
        return SimpleNamespace(workspace_root="/tmp", goal="hi", messages=[])

    monkeypatch.setattr(agent_context, "prepare_agent_context", fake_prepare)

    post_spy = AsyncMock(return_value=None)
    monkeypatch.setattr(agent_context, "post_process_agent_result", post_spy)

    async def fake_classify(msg):
        return SimpleNamespace(goal_type="coding")

    monkeypatch.setattr(goal_classifier, "classify_goal", fake_classify)

    promote_spy = AsyncMock(return_value=None)

    class _FakeMgr:
        async def build_memory_context(self, *a, **k):
            return None

        promote_memories = promote_spy

    monkeypatch.setattr(mem_manager, "get_memory_manager", lambda: _FakeMgr())

    from rune.cli.main import _handle_non_interactive

    _handle_non_interactive("hi")
    return post_spy, promote_spy


def test_normal_run_records_memory(monkeypatch):
    monkeypatch.delenv("RUNE_IN_BEST_OF", raising=False)
    post_spy, promote_spy = _run_non_interactive_with_mocks(monkeypatch)
    post_spy.assert_awaited_once()
    promote_spy.assert_awaited_once()


def test_throwaway_run_skips_memory_writes(monkeypatch):
    monkeypatch.setenv("RUNE_IN_BEST_OF", "1")
    post_spy, promote_spy = _run_non_interactive_with_mocks(monkeypatch)
    post_spy.assert_not_awaited()
    promote_spy.assert_not_awaited()


def test_cli_best_of_zero_errors():
    with patch("rune.cli.main._ensure_llm_key", return_value=True):
        result = runner.invoke(app, ["--message", "hi", "--best-of", "0"])
    assert result.exit_code == 2
