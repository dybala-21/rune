"""Tests for worktree subagent isolation (G2, Phase A).

Deterministic — real git/filesystem, no LLM. Covers the critical invariants:
I1 (isolation enforced), I2 (base = working tree, not HEAD).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rune.agent import isolation, worktree


def _git_repo(path: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    (path / "a.txt").write_text("v1")
    subprocess.run(["git", "add", "."], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True, env=env)


# --- I1: isolation enforcement ------------------------------------------------

class TestIsolationEnforcement:
    def test_noop_when_not_isolating(self, monkeypatch):
        monkeypatch.delenv(isolation.ISOLATION_ENV, raising=False)
        assert isolation.enforce("/anywhere/at/all.txt") is None
        assert isolation.is_within("/anywhere") is True

    def test_inside_root_allowed(self, tmp_dir, monkeypatch):
        root = tmp_dir / "ws"
        root.mkdir()
        monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
        assert isolation.enforce(str(root / "sub" / "f.txt")) is None
        assert isolation.enforce("relative/f.txt") is None  # resolves under root

    def test_absolute_escape_denied(self, tmp_dir, monkeypatch):
        root = tmp_dir / "ws"
        root.mkdir()
        monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
        err = isolation.enforce(str(tmp_dir / "outside.txt"))
        assert err and "Isolation violation" in err

    def test_dotdot_escape_denied(self, tmp_dir, monkeypatch):
        root = tmp_dir / "ws"
        root.mkdir()
        monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
        monkeypatch.chdir(root)
        assert isolation.enforce("../../etc/passwd") is not None

    def test_symlink_escape_denied(self, tmp_dir, monkeypatch):
        root = tmp_dir / "ws"
        root.mkdir()
        (tmp_dir / "secret").mkdir()
        os.symlink(tmp_dir / "secret", root / "link")
        monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
        # writing through the symlink resolves outside root
        assert isolation.enforce(str(root / "link" / "x.txt")) is not None

    def test_tilde_expands_to_home_not_literal_subdir(self, tmp_dir, monkeypatch):
        # The recurring bug: agents write `~/foo.c`. It must be treated as HOME
        # (outside the worktree) -> denied, not as a literal `~` sub-dir inside.
        root = tmp_dir / "ws"
        root.mkdir()
        monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
        assert isolation.enforce("~/foo.c") is not None
        assert not isolation.is_within("~/foo.c")

    def test_prefix_aliasing_not_confused(self, tmp_dir, monkeypatch):
        root = tmp_dir / "work"
        root.mkdir()
        (tmp_dir / "work2").mkdir()
        monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
        # /work2 must NOT be considered inside /work
        assert isolation.enforce(str(tmp_dir / "work2" / "f.txt")) is not None


class TestFileCapabilityEnforcement:
    def test_file_write_denied_outside_isolation(self, monkeypatch):
        import asyncio
        import shutil
        import tempfile

        from rune.capabilities.file import FileWriteParams, file_write
        # Use a workspace under the repo (cwd), not /var tmp — macOS /var/folders
        # is a Guardian-protected path, which would mask the isolation check.
        base = Path(tempfile.mkdtemp(prefix="rune-isotest-", dir=os.getcwd()))
        try:
            root = base / "ws"
            root.mkdir()
            monkeypatch.setenv(isolation.ISOLATION_ENV, str(root))
            # inside → ok (passes Guardian and isolation)
            r_in = asyncio.run(file_write(FileWriteParams(
                path=str(root / "in.txt"), content="hi")))
            assert r_in.success, r_in.error
            # outside → passes Guardian (under repo) but denied by isolation
            r_out = asyncio.run(file_write(FileWriteParams(
                path=str(base / "escape.txt"), content="bad")))
            assert not r_out.success
            assert "Isolation" in (r_out.error or "")
            assert not (base / "escape.txt").exists()
        finally:
            shutil.rmtree(base, ignore_errors=True)


# --- I2 + backend: WorkspaceIsolation ----------------------------------------

class TestWorktreeBackend:
    def test_detects_git(self, tmp_dir):
        assert not worktree.is_git_repo(str(tmp_dir))
        _git_repo(tmp_dir)
        assert worktree.is_git_repo(str(tmp_dir))

    def test_base_is_working_tree_not_head(self, tmp_dir):
        """I2: worktree must reflect UNCOMMITTED edits + untracked files."""
        import asyncio

        _git_repo(tmp_dir)
        (tmp_dir / "a.txt").write_text("v2-uncommitted")   # modify, not committed
        (tmp_dir / "b.txt").write_text("untracked-new")    # untracked
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="worktree"))
        try:
            assert ws.mode == "worktree"
            # The worktree must have the working-tree state, NOT HEAD (v1).
            assert (Path(ws.path) / "a.txt").read_text() == "v2-uncommitted"
            assert (Path(ws.path) / "b.txt").read_text() == "untracked-new"
        finally:
            asyncio.run(worktree.cleanup(ws))

    def test_collect_git_changes(self, tmp_dir):
        import asyncio

        _git_repo(tmp_dir)
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="worktree"))
        try:
            (Path(ws.path) / "a.txt").write_text("edited")   # modify
            (Path(ws.path) / "c.txt").write_text("brand new")  # add
            os.remove(Path(ws.path) / "a.txt") if False else None
            cs = worktree.collect(ws, "w0")
            assert cs.changes["a.txt"].op == "modified"
            assert cs.changes["a.txt"].content == b"edited"
            assert cs.changes["c.txt"].op == "added"
        finally:
            asyncio.run(worktree.cleanup(ws))

    def test_collect_git_deletion(self, tmp_dir):
        import asyncio

        _git_repo(tmp_dir)
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="worktree"))
        try:
            os.remove(Path(ws.path) / "a.txt")
            cs = worktree.collect(ws, "w0")
            assert cs.changes["a.txt"].op == "deleted"
        finally:
            asyncio.run(worktree.cleanup(ws))

    def test_cleanup_removes_worktree(self, tmp_dir):
        import asyncio

        _git_repo(tmp_dir)
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="worktree"))
        path = ws.path
        assert os.path.isdir(path)
        asyncio.run(worktree.cleanup(ws))
        assert not os.path.isdir(path)

    def test_dep_symlink(self, tmp_dir):
        import asyncio

        _git_repo(tmp_dir)
        (tmp_dir / ".gitignore").write_text(".venv/\n")
        (tmp_dir / ".venv").mkdir()
        (tmp_dir / ".venv" / "lib.py").write_text("dep")
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="worktree"))
        try:
            link = Path(ws.path) / ".venv"
            assert link.is_symlink() or link.exists()
            assert (link / "lib.py").read_text() == "dep"
        finally:
            asyncio.run(worktree.cleanup(ws))


class TestCopyBackend:
    def test_copy_mode_for_non_git(self, tmp_dir):
        import asyncio

        (tmp_dir / "a.txt").write_text("v1")
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="auto"))
        try:
            assert ws.mode == "copy"
            assert (Path(ws.path) / "a.txt").read_text() == "v1"
        finally:
            asyncio.run(worktree.cleanup(ws))

    def test_copy_collect_add_modify_delete(self, tmp_dir):
        import asyncio

        (tmp_dir / "a.txt").write_text("v1")
        (tmp_dir / "keep.txt").write_text("keep")
        ws = asyncio.run(worktree.create(str(tmp_dir), "w0", isolation="copy"))
        try:
            (Path(ws.path) / "a.txt").write_text("v2")          # modify
            (Path(ws.path) / "new.txt").write_text("new")       # add
            os.remove(Path(ws.path) / "keep.txt")               # delete
            cs = worktree.collect(ws, "w0")
            assert cs.changes["a.txt"].op == "modified"
            assert cs.changes["new.txt"].op == "added"
            assert cs.changes["keep.txt"].op == "deleted"
        finally:
            asyncio.run(worktree.cleanup(ws))


class TestReaper:
    def test_reap_removes_orphan_tmp(self, tmp_dir):
        import tempfile
        orphan = tempfile.mkdtemp(prefix=worktree._WORKTREE_PREFIX)
        assert os.path.isdir(orphan)
        worktree.reap_orphans(str(tmp_dir))
        assert not os.path.isdir(orphan)
