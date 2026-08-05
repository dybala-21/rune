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


def test_preserve_unverified_keeps_work_when_nothing_verified(tmp_path):
    """"Could not verify" must not mean "deleted": park it for the user."""
    from rune.cli.best_of import _preserve_unverified

    work = tmp_path / "work"
    work.mkdir()
    (work / "solution.py").write_text("MAYBE RIGHT")
    (work / "pkg").mkdir()
    (work / "pkg" / "mod.py").write_text("X")

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "solution.py").write_text("USER FILE")

    kept, saved = _preserve_unverified(str(work), str(dest), ["solution.py", "pkg"])

    assert kept is not None
    assert os.path.basename(kept).startswith(".rune-bestof-unverified-")
    assert sorted(saved) == ["pkg", "solution.py"]
    # Parked beside the project, never over the user's own file.
    assert open(os.path.join(kept, "solution.py")).read() == "MAYBE RIGHT"
    assert open(os.path.join(kept, "pkg", "mod.py")).read() == "X"
    assert (dest / "solution.py").read_text() == "USER FILE"


def test_preserve_unverified_keeps_nested_edits(tmp_path):
    """Seeded mode reports CHANGED relpaths; the parent dirs won't exist yet.

    This is the case that silently dropped the real edit: only the top-level
    file survived while the user was told their work had been kept.
    """
    from rune.cli.best_of import _preserve_unverified

    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "lib.rs").write_text("THE ACTUAL WORK")
    (work / "Cargo.lock").write_text("lockfile")

    dest = tmp_path / "dest"
    dest.mkdir()

    kept, saved = _preserve_unverified(
        str(work), str(dest), ["src/lib.rs", "Cargo.lock"]
    )

    assert kept is not None
    assert sorted(saved) == ["Cargo.lock", "src/lib.rs"]
    assert open(os.path.join(kept, "src", "lib.rs")).read() == "THE ACTUAL WORK"


def test_preserve_unverified_none_when_attempt_produced_nothing(tmp_path):
    from rune.cli.best_of import _preserve_unverified

    work = tmp_path / "work"
    work.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    assert _preserve_unverified(str(work), str(dest), []) == (None, [])
    # A named-but-missing artifact must not leave an empty dir behind.
    assert _preserve_unverified(str(work), str(dest), ["gone.py"]) == (None, [])
    assert list(dest.iterdir()) == []


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
    # backup lives OUT of the user's repo (no cruft / no accidental commit)
    assert os.path.realpath(backup_dir).startswith(
        os.path.realpath(str(dest))
    ) is False
    assert not any(
        n.startswith(".rune-bestof-backup") for n in os.listdir(dest)
    )


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
    # the message is propagated; attempts past the first also carry a
    # distinct entry point, so match on content rather than equality
    assert "--message" in captured["cmd"]
    assert any("do it" in c for c in captured["cmd"])
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
        method_by_cwd: dict[str, str] = {}

        async def verify(cwd):
            # passes only the attempt whose workdir has answer.txt
            method_by_cwd[cwd] = "`pytest -q`"
            return os.path.exists(os.path.join(cwd, "answer.txt"))

        verify.method_by_cwd = method_by_cwd
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
    # what the winner passed is surfaced for the UX line ("passed `pytest -q`")
    assert kw["verify_method"] == "`pytest -q`"
    # artifact restored into the real cwd
    assert (dest / "answer.txt").read_text() == "correct"


@pytest.mark.asyncio
async def test_best_of_concurrency_override_serializes(monkeypatch, tmp_path):
    """RUNE_BESTOF_CONCURRENCY=1 forces serial attempts: no two run at once even
    with K>1 (a single local model server can't serve parallel attempts)."""
    import asyncio

    monkeypatch.setenv("RUNE_BESTOF_CONCURRENCY", "1")
    in_flight = 0
    max_in_flight = 0

    async def fake_attempt(index, message, model, provider, seed_from=None):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.02)  # overlap window if concurrent
        in_flight -= 1
        return AttemptArtifact(index=index, workdir=str(tmp_path / f"w{index}"),
                               stdout=f"o{index}", returncode=0, produced=[])

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False
        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(tmp_path)

    await _best_of_async("task", 4, None, None, report=lambda s, **kw: None)
    assert max_in_flight == 1  # serialized despite K=4


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
    # on an all-tied field the later attempt is surfaced as best-effort
    assert stdout == "out1"
    # F: both attempts wrote files (wrong.txt) but failed → no_artifact == 0
    assert kw["has_check"] is True
    assert kw["no_artifact"] == 0
    # nothing restored
    assert not (dest / "wrong.txt").exists()


@pytest.mark.asyncio
async def test_best_of_seeded_unsolved_applies_best_effort(monkeypatch, tmp_path):
    # Seeded mode, nothing verifies: the best-effort attempt's edits are APPLIED
    # to the working tree (with backup for undo), the run still exits 1, and the
    # report carries the applied files. Parking is skipped (redundant).
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "app.py").write_text("ORIGINAL")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / "w0"
        w.mkdir()
        (w / "app.py").write_text("UNVERIFIED FIX")
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="out", returncode=0, produced=["app.py"]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False  # nothing passes

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.delenv("RUNE_BESTOF_APPLY_UNVERIFIED", raising=False)
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix app", 1, None, None,
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )

    assert code == 1  # still not a success claim
    kw = reports[0]
    assert kw["solved"] is False
    assert kw["applied"] == ["app.py"]
    assert (dest / "app.py").read_text() == "UNVERIFIED FIX"
    # original backed up for undo
    assert kw["apply_backup"] is not None
    assert open(os.path.join(kw["apply_backup"], "app.py")).read() == "ORIGINAL"
    # no redundant parked copy
    assert kw["unverified_dir"] is None


@pytest.mark.asyncio
async def test_best_of_seeded_unsolved_parks_when_opted_out(monkeypatch, tmp_path):
    # RUNE_BESTOF_APPLY_UNVERIFIED=0 restores the previous parking behavior:
    # nothing written to the tree, work kept in a side dir.
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "app.py").write_text("ORIGINAL")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / "w0"
        w.mkdir()
        (w / "app.py").write_text("UNVERIFIED FIX")
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="out", returncode=0, produced=["app.py"]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.setenv("RUNE_BESTOF_APPLY_UNVERIFIED", "0")
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix app", 1, None, None,
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )

    assert code == 1
    kw = reports[0]
    assert kw["applied"] == []
    assert (dest / "app.py").read_text() == "ORIGINAL"  # tree untouched
    assert kw["unverified_dir"] is not None
    parked = os.path.join(kw["unverified_dir"], "app.py")
    assert open(parked).read() == "UNVERIFIED FIX"


@pytest.mark.asyncio
async def test_nondiscriminating_check_never_claims_verified(monkeypatch, tmp_path):
    # The probe finds the check PASSES the unfixed baseline (vacuous). The run
    # must collapse to K=1, disable the EG component, and end UNVERIFIED even
    # though the vacuous check would have "passed" the candidate — with the
    # best-effort edits still applied (delivery, not a success claim).
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "app.py").write_text("ORIGINAL")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / f"w{index}"
        w.mkdir()
        (w / "app.py").write_text("UNVERIFIED FIX")
        return AttemptArtifact(
            index=index, workdir=str(w), stdout="out", returncode=0,
            produced=["app.py"],
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            # Mirrors the real make_verifier contract: the EG component
            # passes anything unless eg_disabled was set by the probe.
            return not getattr(verify, "eg_disabled", False)

        verify.has_check = True
        verify.eg_disabled = False
        verify.evidence_by_cwd = {}
        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.delenv("RUNE_BESTOF_APPLY_UNVERIFIED", raising=False)
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix app", 3, None, None,
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )

    assert code == 1  # NOT a verified success
    kw = reports[0]
    assert kw["solved"] is False
    assert kw["k"] == 1  # collapsed — vacuous check can't select
    assert kw["check_discriminates"] is False
    # delivery still happened, honestly labeled
    assert kw["applied"] == ["app.py"]
    assert (dest / "app.py").read_text() == "UNVERIFIED FIX"


@pytest.mark.asyncio
async def test_seeded_no_change_candidate_never_verifies(monkeypatch, tmp_path):
    # A candidate that changed NOTHING must not be selectable as "verified"
    # even when the check passes it.
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "app.py").write_text("ORIGINAL")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / "w0"
        w.mkdir()
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="out", returncode=0, produced=[]
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False  # baseline probe fails → check discriminates

        verify.has_check = True
        verify.evidence_by_cwd = {}
        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)

    # Baseline fails, but every candidate "passes": only the no-change guard
    # stands between this and a fake verified claim.
    probe_done = {"v": False}

    async def fake_make_verifier2(instruction, seed_cwd=None):
        async def verify(cwd):
            if not probe_done["v"]:
                probe_done["v"] = True
                return False  # discriminating probe
            return True

        verify.has_check = True
        verify.evidence_by_cwd = {}
        return verify

    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier2)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix app", 1, None, None,
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )

    assert code == 1
    assert reports[0]["solved"] is False
    assert (dest / "app.py").read_text() == "ORIGINAL"


def test_drop_seed_identical_filters_reverted_edits(tmp_path):
    from rune.cli.best_of import _drop_seed_identical

    seed = tmp_path / "seed"

    seed.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (seed / "same.py").write_text("x = 1\n")
    (work / "same.py").write_text("x = 1\n")   # edit-then-revert: mtime differs
    (seed / "diff.py").write_text("x = 1\n")
    (work / "diff.py").write_text("x = 2\n")
    (work / "new.py").write_text("fresh\n")     # no seed counterpart

    out = _drop_seed_identical(str(work), str(seed), ["same.py", "diff.py", "new.py"])
    assert out == ["diff.py", "new.py"]


def test_best_effort_prefers_source_edit_over_test_only(tmp_path):
    from rune.cli.best_of import AttemptArtifact, _rank_best_effort

    # Attempt 0 wrote only a scratch test; attempt 1 edited real source.
    # The source-editing sibling must win the hand-off (otherwise
    # a wrong-file/scratch attempt was handed off over the right-file one).
    a0 = AttemptArtifact(index=0, workdir="w0", stdout="", returncode=0,
                         produced=["test_scratch.py"])
    a1 = AttemptArtifact(index=1, workdir="w1", stdout="", returncode=0,
                         produced=["pkg/core.py"])
    best = _rank_best_effort([a0, a1], {})
    assert best is a1


def test_best_effort_scratch_files_lose_to_seed_source(tmp_path):
    from rune.cli.best_of import AttemptArtifact, _rank_best_effort

    # Attempt 0 wrote only debug scripts (new files, not in the seed);
    # attempt 2 edited a file that exists in the seed. A pile of scratch
    # files is not a fix — the real edit must be delivered.
    seed = tmp_path / "seed"
    (seed / "pkg").mkdir(parents=True)
    (seed / "pkg" / "core.py").write_text("x = 1\n")
    a0 = AttemptArtifact(index=0, workdir="w0", stdout="", returncode=0,
                         produced=["debug_ast.py", "check_fix.py"])
    a2 = AttemptArtifact(index=2, workdir="w2", stdout="", returncode=0,
                         produced=["pkg/core.py"])
    best = _rank_best_effort([a0, a2], {}, seed_from=str(seed))
    assert best is a2


def test_best_effort_tie_prefers_repair_attempt(tmp_path):
    from rune.cli.best_of import AttemptArtifact, _rank_best_effort

    # With zero verifier signal and identical shapes, the repair attempt
    # (highest index) saw failure evidence the others didn't — prefer it.
    seed = tmp_path / "seed"
    (seed / "pkg").mkdir(parents=True)
    (seed / "pkg" / "core.py").write_text("x = 1\n")
    arts = [
        AttemptArtifact(index=i, workdir=f"w{i}", stdout="", returncode=0,
                        produced=["pkg/core.py"])
        for i in range(3)
    ]
    best = _rank_best_effort(arts, {}, seed_from=str(seed))
    assert best is arts[2]


@pytest.mark.asyncio
async def test_seeded_attempt_message_carries_test_hint(monkeypatch, tmp_path):
    import rune.agent.auto_verify as av

    captured: dict = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"ok\n", b"")

    async def fake_exec(*cmd, cwd=None, env=None, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr(best_of.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest", "-q"])

    seed = tmp_path / "seed"

    seed.mkdir()
    (seed / "app.py").write_text("x")

    await _run_attempt_subprocess(0, "fix it", None, None, seed_from=str(seed))

    msg = captured["cmd"][captured["cmd"].index("--message") + 1]
    assert "pytest -q" in msg  # verify loop handed to the attempt
    assert "fix failures before finishing" in msg


@pytest.mark.asyncio
async def test_best_of_greenfield_unsolved_still_parks(monkeypatch, tmp_path):
    # Non-seeded (greenfield) mode keeps the parking contract: apply-unverified
    # is a seeded-mode delivery change only.
    w = tmp_path / "w0"
    w.mkdir()
    (w / "solution.py").write_text("UNVERIFIED")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(
            index=0, workdir=str(w), stdout="out", returncode=0,
            produced=["solution.py"],
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False

        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.delenv("RUNE_BESTOF_APPLY_UNVERIFIED", raising=False)

    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "task", 1, None, None, report=lambda s, **kw: reports.append(kw)
    )

    assert code == 1
    kw = reports[0]
    assert kw["applied"] == []
    assert not (dest / "solution.py").exists()
    assert kw["unverified_dir"] is not None


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


def test_attempt_work_root_is_guardian_safe():
    # macOS tempfile.mkdtemp() lands under /var/folders, which the Guardian
    # blocks as protected, so attempts there produce no files. The attempt root
    # must live under the data dir (in $HOME), not /var.
    from rune.cli.best_of import _attempt_work_root
    from rune.utils.paths import rune_data
    root = _attempt_work_root()
    assert not root.startswith("/var"), f"attempt root under protected /var: {root}"
    assert str(rune_data()) in root
    import os
    assert os.path.isdir(root)


def _art(i, produced):
    from rune.cli.best_of import AttemptArtifact
    return AttemptArtifact(index=i, workdir=f"/w{i}", stdout="", returncode=1,
                           produced=produced)


def test_rank_best_effort_prefers_candidate_that_produced_files():
    """When nothing verifies, don't blindly hand off #0 — #0 may be empty while a
    sibling wrote a real patch. That empty-vs-real gap was RUNE's empty patches."""
    from rune.cli.best_of import _rank_best_effort
    best = _rank_best_effort([_art(0, []), _art(1, []), _art(2, ["src/fix.py"])], {})
    assert best.index == 2


def test_rank_best_effort_prefers_ran_over_did_not_compile():
    from rune.cli.best_of import _rank_best_effort
    ev = {"/w0": "error[E0433]: could not compile `x`",
          "/w1": "1 failed, 0 passed in 0.1s"}
    best = _rank_best_effort([_art(0, ["a.py"]), _art(1, ["a.py"])], ev)
    assert best.index == 1


def test_rank_best_effort_deterministic_and_empty():
    from rune.cli.best_of import _rank_best_effort
    # deterministic on a tie: the later (better-informed) attempt wins
    assert _rank_best_effort([_art(0, []), _art(1, [])], {}).index == 1
    assert _rank_best_effort([], {}) is None


def test_verifier_discriminates_detects_nondiscriminating_check(tmp_path):
    """A check that passes the untouched baseline can't select — collapse to K=1.

    This is the measured 3.23x waste: best-of-K with a verifier that accepts
    anything spends K times the cost for one-shot quality.
    """
    import asyncio

    from rune.cli.best_of import _verifier_discriminates

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "a.py").write_text("x = 1\n")

    async def passes_anything(cwd):
        return True

    async def fails_baseline(cwd):
        return False

    # non-discriminating -> discriminates() is False -> caller drops to K=1
    assert asyncio.run(_verifier_discriminates(passes_anything, str(seed))) is False
    # discriminating -> True -> caller keeps K
    assert asyncio.run(_verifier_discriminates(fails_baseline, str(seed))) is True
    # the user's baseline tree is never mutated by the probe
    assert [p.name for p in seed.iterdir()] == ["a.py"]


def test_verifier_discriminates_defaults_true_on_error(tmp_path):
    """A probe failure must never suppress best-of."""
    import asyncio

    from rune.cli.best_of import _verifier_discriminates

    async def boom(cwd):
        raise RuntimeError("probe blew up")

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "a.py").write_text("x = 1\n")
    assert asyncio.run(_verifier_discriminates(boom, str(seed))) is True


# --- sampling strategies (sequential / race2 / repair) ----------------------


def _mk_verifier(pass_indices=frozenset(), evidence="1 failed: expected 3 got 2"):
    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            i = int(cwd.rsplit("_w", 1)[-1])
            ok = i in pass_indices
            if not ok:
                verify.evidence_by_cwd[cwd] = evidence
            return ok

        verify.has_check = True
        verify.evidence_by_cwd = {}
        return verify

    return fake_make_verifier


def _mk_attempts(tmp_path, spawned):
    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / f"strat_w{index}"
        w.mkdir(exist_ok=True)
        (w / "fix.py").write_text(f"attempt {index}")
        spawned.append((index, message))
        return AttemptArtifact(
            index=index, workdir=str(w), stdout=f"out{index}", returncode=0,
            produced=["fix.py"],
        )

    return fake_attempt


@pytest.mark.asyncio
async def test_sequential_stops_at_first_verified(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices={0}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d1"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "task", 3, None, None, report=lambda s, **kw: reports.append(kw),
        seed_cwd=True,
    )
    assert code == 0
    assert len(spawned) == 1  # early exit: attempts 2-3 never sampled
    assert reports[0]["selected_index"] == 0


@pytest.mark.asyncio
async def test_sequential_repair_feeds_failure_output(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    monkeypatch.delenv("RUNE_BESTOF_REPAIR", raising=False)
    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices=set()))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d2"
    dest.mkdir()
    monkeypatch.chdir(dest)

    await _best_of_async(
        "task", 3, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    assert len(spawned) == 3
    assert "FAILED" not in spawned[0][1]
    assert "expected 3 got 2" in spawned[1][1]  # attempt 2 = repair w/ failure
    assert "expected 3 got 2" not in spawned[2][1]  # attempt 3 = fresh sample


@pytest.mark.asyncio
async def test_race2_runs_two_then_repairs(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    monkeypatch.delenv("RUNE_BESTOF_REPAIR", raising=False)
    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices={2}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d3"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "task", 3, None, "anthropic", report=lambda s, **kw: reports.append(kw),
        seed_cwd=True,
    )
    assert code == 0
    assert len(spawned) == 3
    assert "expected 3 got 2" in spawned[2][1]  # 3rd attempt got repair info
    assert reports[0]["selected_index"] == 2


def _mk_timed_attempts(tmp_path, spawned, finished, delays):
    """Attempts that take different amounts of time, and say if they finished."""
    async def fake_attempt(index, message, model, provider, seed_from=None):
        spawned.append(index)
        await asyncio.sleep(delays.get(index, 0.0))
        w = tmp_path / f"strat_w{index}"
        w.mkdir(exist_ok=True)
        (w / "fix.py").write_text(f"attempt {index}")
        finished.append(index)
        return AttemptArtifact(
            index=index, workdir=str(w), stdout=f"out{index}", returncode=0,
            produced=["fix.py"],
        )

    return fake_attempt


@pytest.mark.asyncio
async def test_race2_winner_cancels_the_running_sibling(monkeypatch, tmp_path):
    """A pass must stop the race, not wait politely for the loser.

    The old shape gathered both attempts before verifying either, so when the
    fast one passed, the slow one's whole remaining runtime had already been
    paid for nothing. Now the sibling is cancelled the moment a winner
    verifies — on the seeded path that is minutes of model time per race.
    """
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    spawned, finished = [], []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess",
                        _mk_timed_attempts(tmp_path, spawned, finished,
                                           {0: 0.0, 1: 30.0}))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices={0}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d_race_cancel"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []
    code = await asyncio.wait_for(
        _best_of_async("task", 3, None, "anthropic",
                       report=lambda s, **kw: reports.append(kw), seed_cwd=True),
        timeout=10,   # far under the loser's 30s: the win must not wait for it
    )
    assert code == 0
    assert reports[0]["selected_index"] == 0
    assert sorted(spawned) == [0, 1]   # both started
    assert finished == [0]             # the loser never completed


@pytest.mark.asyncio
async def test_race2_first_to_land_passer_wins(monkeypatch, tmp_path):
    # Verification runs as attempts land, so a passing attempt 1 that
    # finishes first is the winner — there is no reason to keep paying for
    # attempt 0 to find out whether it would also have passed.
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    spawned, finished = [], []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess",
                        _mk_timed_attempts(tmp_path, spawned, finished,
                                           {0: 30.0, 1: 0.0}))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices={0, 1}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d_race_land"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []
    code = await asyncio.wait_for(
        _best_of_async("task", 3, None, "anthropic",
                       report=lambda s, **kw: reports.append(kw), seed_cwd=True),
        timeout=10,
    )
    assert code == 0
    assert reports[0]["selected_index"] == 1
    assert finished == [1]


@pytest.mark.asyncio
async def test_race2_failed_fast_attempt_does_not_end_the_race(monkeypatch, tmp_path):
    # A fast failure is not a verdict on the race: the slower attempt still
    # gets verified and can win.
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    spawned, finished = [], []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess",
                        _mk_timed_attempts(tmp_path, spawned, finished,
                                           {0: 0.3, 1: 0.0}))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices={0}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d_race_slowwin"
    dest.mkdir()
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "task", 3, None, "anthropic",
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )
    assert code == 0
    assert reports[0]["selected_index"] == 0
    assert sorted(finished) == [0, 1]  # nobody was cancelled


def _mk_repro_verifier(tmp_path, verdicts):
    """A verifier whose repro script grades each candidate per *verdicts*.

    This drives the REAL evidence hand-off in _best_of_async — the earlier
    version of these tests re-implemented that logic inside the test and
    then tested the copy, which would have kept passing however the
    production path broke.
    """
    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            i = int(cwd.rsplit("_w", 1)[-1])
            verify.repro_results[cwd] = verdicts.get(i, False)
            verify.evidence_by_cwd[cwd] = f"AssertionError: attempt {i} wrong"
            return False                      # nobody passes; repair must fire

        async def grade(cwd):
            i = int(cwd.rsplit("_w", 1)[-1])
            verify.repro_results[cwd] = verdicts.get(i, False)
            return verify.repro_results[cwd]

        verify.has_check = True
        verify.repro_script = "assert fixed()"
        verify.repro_results = {}
        verify.grade_repro = grade
        verify.evidence_by_cwd = {}
        return verify

    return fake_make_verifier


@pytest.mark.asyncio
async def test_repair_evidence_is_withheld_when_the_repro_separates_nothing(
        monkeypatch, tmp_path):
    # Both candidates fail the repro the same way: the script has separated
    # nothing, and its output would aim the repair attempt at a requirement
    # the correct fix may not even be meant to satisfy.
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    monkeypatch.delenv("RUNE_BESTOF_REPAIR", raising=False)
    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier",
                        _mk_repro_verifier(tmp_path, {0: False, 1: False}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d_withhold"
    dest.mkdir()
    monkeypatch.chdir(dest)

    await _best_of_async(
        "task", 3, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    repair_msgs = [m for i, m in spawned if i == 2]
    assert repair_msgs and "AssertionError" not in repair_msgs[0]


@pytest.mark.asyncio
async def test_repair_evidence_is_passed_on_when_the_repro_separates(
        monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    monkeypatch.delenv("RUNE_BESTOF_REPAIR", raising=False)
    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier",
                        _mk_repro_verifier(tmp_path, {0: True, 1: False}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d_passon"
    dest.mkdir()
    monkeypatch.chdir(dest)

    await _best_of_async(
        "task", 3, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    repair_msgs = [m for i, m in spawned if i == 2]
    assert repair_msgs and "AssertionError" in repair_msgs[0]


@pytest.mark.asyncio
async def test_children_inherit_the_parent_classification(monkeypatch, tmp_path):
    """One classification per family. Attempt children used to re-classify
    the identical message — K identical model calls a run."""
    import rune.agent.goal_classifier as gc

    calls = []

    async def fake_classify(message, **kw):
        calls.append(message)
        return gc.ClassificationResult(goal_type="code_modify",
                                       confidence=0.9, tier=2,
                                       is_complex_coding=True)
    monkeypatch.setattr(best_of, "classify_goal", fake_classify, raising=False)
    monkeypatch.setattr(gc, "classify_goal", fake_classify)
    monkeypatch.delenv("RUNE_BESTOF_CLASSIFICATION", raising=False)
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "race2")
    spawned_envs = []

    async def fake_attempt(index, message, model, provider, seed_from=None):
        import os as _os
        spawned_envs.append(_os.environ.get("RUNE_BESTOF_CLASSIFICATION", ""))
        w = tmp_path / f"strat_w{index}"
        w.mkdir(exist_ok=True)
        (w / "fix.py").write_text("x")
        return AttemptArtifact(index=index, workdir=str(w), stdout="",
                               returncode=0, produced=["fix.py"])
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices={0}))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d_cls"
    dest.mkdir()
    monkeypatch.chdir(dest)

    await _best_of_async(
        "task", 3, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    assert len(calls) == 1                      # parent classified exactly once
    assert spawned_envs and all(e for e in spawned_envs)
    assert gc.from_wire(spawned_envs[0]).is_complex_coding is True


@pytest.mark.asyncio
async def test_repair_env_opt_out(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    monkeypatch.setenv("RUNE_BESTOF_REPAIR", "0")
    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices=set()))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    dest = tmp_path / "d4"
    dest.mkdir()
    monkeypatch.chdir(dest)

    await _best_of_async(
        "task", 3, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    assert all("expected 3 got 2" not in m for _, m in spawned)


@pytest.mark.asyncio
async def test_provisional_selection_never_claims_verified(monkeypatch, tmp_path):
    # Repo-existing-tests pass selects the candidate but must deliver as
    # UNVERIFIED (exit 1, files applied) — those tests pass pre-fix code too.
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    dest = tmp_path / "dp"
    dest.mkdir()
    (dest / "app.py").write_text("ORIGINAL")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / f"pv_w{index}"
        w.mkdir()
        (w / "app.py").write_text("PROVISIONAL FIX")
        return AttemptArtifact(
            index=index, workdir=str(w), stdout="out", returncode=0,
            produced=["app.py"],
        )

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            verify.provisional_by_cwd[cwd] = True  # targeted-pass semantics
            return True

        verify.has_check = True
        verify.evidence_by_cwd = {}
        verify.provisional_by_cwd = {}
        return verify

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix", 3, None, "anthropic",
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )
    assert code == 1  # NOT a verified success
    kw = reports[0]
    assert kw.get("provisional") is True
    assert kw["solved"] is False
    assert kw["applied"] == ["app.py"]
    assert (dest / "app.py").read_text() == "PROVISIONAL FIX"  # still delivered


def test_drop_build_metadata_filters_packaging_junk():
    from rune.cli.best_of import _drop_build_metadata

    rels = [
        "src/flask/cli.py",
        "src/Flask.egg-info/PKG-INFO",
        "src/Flask.egg-info/SOURCES.txt",
        "pkg.dist-info/METADATA",
        ".eggs/setuptools_scm/x.py",
        "tests/test_cli.py",
    ]
    assert _drop_build_metadata(rels) == ["src/flask/cli.py", "tests/test_cli.py"]


@pytest.mark.asyncio
async def test_fastpath_verified_short_circuits(monkeypatch, tmp_path):
    # Rung-0 repro flip: fix applied, no agentic attempts spawned, and the
    # delivery is provisional (exit 1) — a repro written from the issue
    # text selects a candidate, it does not verify one.
    monkeypatch.setenv("RUNE_FASTPATH", "1")
    dest = tmp_path / "fp"
    dest.mkdir()
    (dest / "app.py").write_text("BROKEN")

    async def fake_fastpath(issue, seed, workdir, model, provider):
        import os

        from rune.agent.fastpath import FastPathResult
        with open(os.path.join(workdir, "app.py"), "w") as fh:
            fh.write("FIXED")
        return FastPathResult(verified=True, applied=["app.py"],
                              method="reproduction script")

    import rune.agent.fastpath as fp_mod
    monkeypatch.setattr(fp_mod, "run_fastpath", fake_fastpath)

    async def no_attempts(*a, **k):
        raise AssertionError("agentic rung must not run")

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", no_attempts)

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            return False
        verify.has_check = True
        verify.evidence_by_cwd = {}
        return verify

    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix", 3, None, "anthropic",
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )
    assert code == 1
    assert reports[0]["solved"] is False
    assert reports[0]["provisional"] is True
    assert reports[0]["applied"] == ["app.py"]
    assert (dest / "app.py").read_text() == "FIXED"


@pytest.mark.asyncio
async def test_fastpath_evidence_reaches_agentic_rung(monkeypatch, tmp_path):
    # Rung-0 fails but found a discriminating repro → attempts get the
    # script as structured evidence in their message.
    monkeypatch.setenv("RUNE_FASTPATH", "1")
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    dest = tmp_path / "fp2"
    dest.mkdir()
    (dest / "app.py").write_text("BROKEN")

    async def fake_fastpath(issue, seed, workdir, model, provider):
        from rune.agent.fastpath import FastPathResult
        return FastPathResult(repro_script="assert fixed()",
                              repro_output="AssertionError")

    import rune.agent.fastpath as fp_mod
    monkeypatch.setattr(fp_mod, "run_fastpath", fake_fastpath)

    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices=set()))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(dest)

    await _best_of_async(
        "fix", 1, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    assert spawned and "assert fixed()" in spawned[0][1]


@pytest.mark.asyncio
async def test_fastpath_evidence_only_reaches_first_attempt(monkeypatch, tmp_path):
    # The repro encodes ONE reading of the issue. Broadcasting it to every
    # attempt pins all K samples to that reading — only attempt 0 gets it.
    monkeypatch.setenv("RUNE_FASTPATH", "1")
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    dest = tmp_path / "fp3"
    dest.mkdir()
    (dest / "app.py").write_text("BROKEN")

    async def fake_fastpath(issue, seed, workdir, model, provider):
        from rune.agent.fastpath import FastPathResult
        return FastPathResult(repro_script="assert fixed()",
                              repro_output="AssertionError")

    import rune.agent.fastpath as fp_mod
    monkeypatch.setattr(fp_mod, "run_fastpath", fake_fastpath)

    spawned: list = []
    monkeypatch.setattr(best_of, "_run_attempt_subprocess", _mk_attempts(tmp_path, spawned))
    monkeypatch.setattr(best_of, "make_verifier", _mk_verifier(pass_indices=set()))
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(dest)

    await _best_of_async(
        "fix", 2, None, "anthropic", report=lambda s, **kw: None, seed_cwd=True,
    )
    assert len(spawned) == 2
    assert "assert fixed()" in spawned[0][1]
    assert "assert fixed()" not in spawned[1][1]


@pytest.mark.asyncio
async def test_fastpath_repro_reaches_real_verifier(monkeypatch, tmp_path):
    # Full flow: fastpath attaches its repro to the REAL make_verifier object
    # and candidate verification actually executes the repro branch.
    import rune.agent.auto_verify as av
    import rune.agent.fastpath as fp_mod
    import rune.agent.rejection_sampler as rs

    monkeypatch.setenv("RUNE_FASTPATH", "1")
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "sequential")
    dest = tmp_path / "flow"
    dest.mkdir()
    (dest / "pkg").mkdir()
    (dest / "pkg" / "mod.py").write_text("x = 1\n")
    (dest / "setup.py").write_text("# marker\n")

    async def fake_fastpath(issue, seed, workdir, model, provider):
        from rune.agent.fastpath import FastPathResult
        return FastPathResult(
            repro_script=(
                "import sys, os\nsys.path.insert(0, os.getcwd())\n"
                "src = open('pkg/mod.py').read()\nassert 'x = 2' in src\n"
            ),
            repro_output="AssertionError",
        )

    monkeypatch.setattr(fp_mod, "run_fastpath", fake_fastpath)
    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

    async def fake_eg(instruction):
        async def v(cwd):
            return False
        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    async def fake_attempt(index, message, model, provider, seed_from=None):
        w = tmp_path / f"flow_w{index}"
        (w / "pkg").mkdir(parents=True)
        (w / "pkg" / "mod.py").write_text("x = 2\n")  # the fix
        return AttemptArtifact(
            index=index, workdir=str(w), stdout="out", returncode=0,
            produced=["pkg/mod.py"],
        )

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.chdir(dest)

    reports: list = []
    code = await _best_of_async(
        "fix the bug", 3, None, "anthropic",
        report=lambda s, **kw: reports.append(kw), seed_cwd=True,
    )
    # The candidate flips the repro → verified solve through the REAL verifier.
    # repro flip selects the candidate; the delivery label is provisional
    assert code == 1
    assert reports[0]["solved"] is False
    assert reports[0]["provisional"] is True


@pytest.mark.asyncio
async def test_provisional_selection_still_distills_the_contrast(monkeypatch, tmp_path):
    """The winner-vs-losers contrast was only learned from a verified win.
    Since a reproduction flip stopped counting as verified, that branch
    barely opens on a repo fix, so the distillation ran almost never — the
    provisional pick carries the same contrast and must feed it too."""
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "app.py").write_text("x = 1\n")

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            passed = os.path.exists(os.path.join(cwd, "fix.py"))
            if passed:
                verify.provisional_by_cwd[cwd] = True
            return passed
        verify.has_check = True
        verify.evidence_by_cwd = {}
        verify.provisional_by_cwd = {}
        verify.method_by_cwd = {}
        return verify

    works = []
    for i in range(2):
        w = tmp_path / f"w{i}"
        w.mkdir()
        if i == 0:
            (w / "fix.py").write_text("correct")
        else:
            (w / "wrong.py").write_text("wrong")
        works.append(str(w))

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(index=index, workdir=works[index],
                               stdout=f"o{index}", returncode=0,
                               produced=sorted(os.listdir(works[index])))

    called = {}

    async def fake_contrast(winner, losers, ev_map):
        called["winner"] = winner.index
        called["losers"] = [lo.index for lo in losers]
        return "rule-key"

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_learn_from_contrast", fake_contrast)
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.setenv("RUNE_BESTOF_STRATEGY", "parallel")
    monkeypatch.delenv("RUNE_CONTRASTIVE_DISTILL", raising=False)
    monkeypatch.chdir(dest)

    code = await _best_of_async("fix", 2, None, None,
                                report=lambda s, **kw: None, seed_cwd=True)

    assert code == 1                       # provisional, not a claim of success
    assert called.get("winner") == 0
    assert called.get("losers") == [1]


@pytest.mark.asyncio
async def test_contrast_distillation_can_be_switched_off(monkeypatch, tmp_path):
    dest = tmp_path / "proj2"
    dest.mkdir()
    (dest / "app.py").write_text("x = 1\n")

    async def fake_make_verifier(instruction, seed_cwd=None):
        async def verify(cwd):
            passed = os.path.exists(os.path.join(cwd, "fix.py"))
            if passed:
                verify.provisional_by_cwd[cwd] = True
            return passed
        verify.has_check = True
        verify.evidence_by_cwd = {}
        verify.provisional_by_cwd = {}
        verify.method_by_cwd = {}
        return verify

    w = tmp_path / "wa"
    w.mkdir()
    (w / "fix.py").write_text("correct")

    async def fake_attempt(index, message, model, provider, seed_from=None):
        return AttemptArtifact(index=index, workdir=str(w), stdout="o",
                               returncode=0, produced=["fix.py"])

    called = {}

    async def fake_contrast(winner, losers, ev_map):
        called["ran"] = True
        return None

    monkeypatch.setattr(best_of, "_run_attempt_subprocess", fake_attempt)
    monkeypatch.setattr(best_of, "make_verifier", fake_make_verifier)
    monkeypatch.setattr(best_of, "_learn_from_contrast", fake_contrast)
    monkeypatch.setattr(best_of, "_verifier_discriminates", AsyncMock(return_value=True))
    monkeypatch.setattr(best_of, "_cleanup", lambda arts: None)
    monkeypatch.setenv("RUNE_CONTRASTIVE_DISTILL", "0")
    monkeypatch.chdir(dest)

    await _best_of_async("fix", 1, None, None,
                         report=lambda s, **kw: None, seed_cwd=True)
    assert "ran" not in called


class TestAttemptDiversity:
    """K attempts were near-copies: all at temperature 0, and race2 gave the
    first two the same message. Grading every attempt against the hidden
    tests showed three attempts at 33% each covering only 50% of runs where
    independence predicts 70%."""

    def test_attempt_zero_keeps_the_default_framing(self):
        from rune.cli.best_of import _diversify
        entry, temp = _diversify(0)
        assert entry == ""
        assert temp == 0.0

    def test_later_attempts_get_a_different_entry_point_and_temperature(self):
        from rune.cli.best_of import _diversify
        e1, t1 = _diversify(1)
        e2, t2 = _diversify(2)
        assert e1 and e2 and e1 != e2
        assert 0.0 < t1 < t2

    def test_it_cycles_for_larger_k(self):
        from rune.cli.best_of import _diversify
        assert _diversify(3) == _diversify(0)

    def test_diversity_can_be_switched_off(self, monkeypatch):
        from rune.cli.best_of import _diversify
        monkeypatch.setenv("RUNE_BESTOF_DIVERSIFY", "0")
        assert _diversify(2) == ("", None)


@pytest.mark.asyncio
async def test_each_attempt_is_spawned_with_its_own_temperature(monkeypatch, tmp_path):
    import rune.agent.auto_verify as av

    captured = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"ok\n", b"")

    async def fake_exec(*cmd, cwd=None, env=None, **kwargs):
        captured[env.get("RUNE_TEMPERATURE")] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr(best_of.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)
    monkeypatch.delenv("RUNE_BESTOF_DIVERSIFY", raising=False)
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "app.py").write_text("x")

    for i in range(3):
        await _run_attempt_subprocess(i, "fix it", None, None, seed_from=str(seed))

    assert sorted(captured) == ["0.0", "0.4", "0.7"]
    # the later attempts also carry a distinct entry point in the message
    msgs = ["".join(c) for c in captured.values()]
    assert len({m for m in msgs}) == 3

