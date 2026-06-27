"""Tests for the atomic merge coordinator (G2, Phase A).

Deterministic — real files + git merge-file, no LLM. Covers I3 (atomicity:
all-or-nothing) and I4 (rollback), plus conflict policies and 3-way merge.
"""

from __future__ import annotations

import os
from pathlib import Path

from rune.agent import merge
from rune.agent.worktree import ChangeSet, FileChange


def _cs(worker_id: str, **changes: FileChange) -> ChangeSet:
    return ChangeSet(worker_id=worker_id, changes=dict(changes))


def _mod(content: str) -> FileChange:
    return FileChange(op="modified", content=content.encode())


def _add(content: str) -> FileChange:
    return FileChange(op="added", content=content.encode())


def _repo(tmp_dir: Path) -> str:
    return str(tmp_dir)


class TestDisjoint:
    def test_two_workers_different_files(self, tmp_dir):
        repo = _repo(tmp_dir)
        cs = [_cs("w1", **{"a.txt": _add("AAA")}),
              _cs("w2", **{"b.txt": _add("BBB")})]
        r = merge.merge_changesets(repo, cs)
        assert r.ok
        assert set(r.applied) == {"a.txt", "b.txt"}
        assert (tmp_dir / "a.txt").read_text() == "AAA"
        assert (tmp_dir / "b.txt").read_text() == "BBB"

    def test_added_and_deleted(self, tmp_dir):
        repo = _repo(tmp_dir)
        (tmp_dir / "old.txt").write_text("bye")
        cs = [_cs("w1", **{"new.txt": _add("hi"),
                           "old.txt": FileChange(op="deleted")})]
        r = merge.merge_changesets(repo, cs)
        assert r.ok
        assert (tmp_dir / "new.txt").read_text() == "hi"
        assert not (tmp_dir / "old.txt").exists()


class TestConflictAndAtomicity:
    def test_identical_edits_not_conflict(self, tmp_dir):
        repo = _repo(tmp_dir)
        cs = [_cs("w1", **{"f.txt": _mod("same")}),
              _cs("w2", **{"f.txt": _mod("same")})]
        r = merge.merge_changesets(repo, cs)
        assert r.ok
        assert (tmp_dir / "f.txt").read_text() == "same"

    def test_3way_disjoint_regions_merges(self, tmp_dir):
        repo = _repo(tmp_dir)
        base = "L1\nL2\nL3\nL4\nL5\nL6\nL7\nL8\nL9\nL10\n"
        (tmp_dir / "f.txt").write_text(base)            # base = current main
        a = base.replace("L2", "L2-edited-by-A")
        b = base.replace("L9", "L9-edited-by-B")
        cs = [_cs("w1", **{"f.txt": _mod(a)}),
              _cs("w2", **{"f.txt": _mod(b)})]
        r = merge.merge_changesets(repo, cs)
        assert r.ok, r.reason
        merged = (tmp_dir / "f.txt").read_text()
        assert "L2-edited-by-A" in merged and "L9-edited-by-B" in merged

    def test_overlapping_edits_conflict_nothing_applied(self, tmp_dir):
        repo = _repo(tmp_dir)
        (tmp_dir / "f.txt").write_text("L1\nL2\nL3\n")
        # both edit the SAME line differently
        a = "L1\nAAA\nL3\n"
        b = "L1\nBBB\nL3\n"
        # plus a disjoint file that WOULD apply — must NOT, due to atomicity
        cs = [_cs("w1", **{"f.txt": _mod(a), "ok.txt": _add("should-not-land")}),
              _cs("w2", **{"f.txt": _mod(b)})]
        r = merge.merge_changesets(repo, cs)
        assert not r.ok
        assert any("auto-merge" in reason or "overlap" in reason
                   for _, reason in r.conflicts)
        # I3: nothing applied — main unchanged, disjoint file not created
        assert (tmp_dir / "f.txt").read_text() == "L1\nL2\nL3\n"
        assert not (tmp_dir / "ok.txt").exists()

    def test_delete_vs_modify_conflict(self, tmp_dir):
        repo = _repo(tmp_dir)
        (tmp_dir / "f.txt").write_text("orig")
        cs = [_cs("w1", **{"f.txt": _mod("changed")}),
              _cs("w2", **{"f.txt": FileChange(op="deleted")})]
        r = merge.merge_changesets(repo, cs)
        assert not r.ok
        assert (tmp_dir / "f.txt").read_text() == "orig"   # unchanged


class TestPolicies:
    def test_fail_closed_any_overlap(self, tmp_dir):
        repo = _repo(tmp_dir)
        (tmp_dir / "f.txt").write_text("base\n")
        cs = [_cs("w1", **{"f.txt": _mod("a\n")}),
              _cs("w2", **{"f.txt": _mod("b\n")})]
        r = merge.merge_changesets(repo, cs, policy=merge.FAIL_CLOSED)
        assert not r.ok

    def test_last_write_takes_last(self, tmp_dir):
        repo = _repo(tmp_dir)
        (tmp_dir / "f.txt").write_text("base\n")
        cs = [_cs("w1", **{"f.txt": _mod("first\n")}),
              _cs("w2", **{"f.txt": _mod("last\n")})]
        r = merge.merge_changesets(repo, cs, policy=merge.LAST_WRITE)
        assert r.ok
        assert (tmp_dir / "f.txt").read_text() == "last\n"


class TestRollback:
    def test_rollback_restores_state(self, tmp_dir):
        repo = _repo(tmp_dir)
        (tmp_dir / "keep.txt").write_text("original")
        cs = [_cs("w1", **{"keep.txt": _mod("modified"),
                           "added.txt": _add("new")})]
        r = merge.merge_changesets(repo, cs)
        assert r.ok
        assert (tmp_dir / "keep.txt").read_text() == "modified"
        assert (tmp_dir / "added.txt").exists()
        # Simulate a failed post-merge gate → roll back.
        merge.rollback(repo, r.snapshot)
        assert (tmp_dir / "keep.txt").read_text() == "original"  # restored
        assert not (tmp_dir / "added.txt").exists()              # added removed


class TestSafety:
    def test_path_escape_rejected(self, tmp_dir):
        repo = str(tmp_dir / "repo")
        os.makedirs(repo)
        cs = [_cs("w1", **{"../escape.txt": _add("bad")})]
        r = merge.merge_changesets(repo, cs)
        assert not r.ok
        assert any("escape" in reason for _, reason in r.conflicts)
        assert not (tmp_dir / "escape.txt").exists()

    def test_none_changesets_filtered(self, tmp_dir):
        r = merge.merge_changesets(_repo(tmp_dir), [None])  # type: ignore[list-item]
        assert r.ok
        assert r.applied == []
