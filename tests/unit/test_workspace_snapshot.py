"""A shell command must not be able to destroy work irreversibly.

The file capabilities send deletes to a trash directory, but a shell
command reaches the same files without touching those capabilities — `rm`,
`mv`, a `>` redirect. Cloning the workspace before such a command runs is
what makes that recoverable, and the clone is skipped for commands that
cannot write so it stays cheap.
"""

from __future__ import annotations

import pytest

from rune.safety import workspace_snapshot as ws


class TestReadOnlyDetection:
    @pytest.mark.parametrize("cmd", [
        "ls -la", "cat README.md", "grep -r foo src/", "git status",
        "git log --oneline -20", "find . -name '*.py'", "wc -l *.py",
        "PYTHONPATH=. grep x y", "ls | grep foo | head -5",
    ])
    def test_read_only_commands_skip_the_clone(self, cmd):
        assert ws.looks_read_only(cmd)

    @pytest.mark.parametrize("cmd", [
        "rm -rf build", "mv a.py b.py", "cat > out.txt", "echo hi >> log",
        "python3 script.py", "git checkout .", "git clean -fd",
        "find . -name '*.pyc' -delete", "find . -exec rm {} +",
        "ls && rm tmp", "make clean", "npm install",
    ])
    def test_anything_that_might_write_is_snapshotted(self, cmd):
        assert not ws.looks_read_only(cmd)

    def test_unknown_tools_are_treated_as_writers(self):
        # The safe direction: an unrecognised command costs a clone, not data.
        assert not ws.looks_read_only("some-unknown-tool --go")

    def test_empty_input_is_harmless(self):
        assert ws.looks_read_only("")
        assert ws.looks_read_only("   ")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "rune_data", lambda: str(tmp_path / "data"))
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n")
    (root / "notes.md").write_text("keep me\n")
    return root


class TestSnapshotAndRestore:
    def test_a_deleted_file_can_be_put_back(self, workspace):
        assert ws.take(workspace) is not None
        (workspace / "notes.md").unlink()
        assert ws.missing_since_snapshot(workspace) == ["notes.md"]
        assert ws.restore(workspace, "notes.md")
        assert (workspace / "notes.md").read_text() == "keep me\n"

    def test_nested_files_are_covered(self, workspace):
        ws.take(workspace)
        (workspace / "pkg" / "mod.py").unlink()
        assert "pkg/mod.py" in ws.missing_since_snapshot(workspace)
        assert ws.restore(workspace, "pkg/mod.py")
        assert (workspace / "pkg" / "mod.py").read_text() == "x = 1\n"

    def test_an_untouched_workspace_reports_nothing_missing(self, workspace):
        ws.take(workspace)
        (workspace / "new.py").write_text("y = 2\n")
        assert ws.missing_since_snapshot(workspace) == []

    def test_only_the_most_recent_snapshots_are_kept(self, workspace):
        for i in range(6):
            (workspace / f"f{i}.txt").write_text(str(i))
            assert ws.take(workspace) is not None
        root = ws._snapshot_root(workspace.resolve())
        assert len(list(root.iterdir())) <= ws._KEEP

    def test_snapshots_live_outside_the_workspace(self, workspace):
        ws.take(workspace)
        assert not any(p.name.startswith(".rune-snap")
                       for p in workspace.iterdir())

    def test_restore_reports_failure_rather_than_raising(self, workspace):
        ws.take(workspace)
        assert not ws.restore(workspace, "never-existed.txt")

    def test_it_can_be_switched_off(self, workspace, monkeypatch):
        monkeypatch.setenv("RUNE_SHELL_SNAPSHOT", "0")
        assert ws.take(workspace) is None
        assert ws.missing_since_snapshot(workspace) == []


@pytest.mark.asyncio
async def test_shell_delete_is_recoverable_and_reported(workspace, monkeypatch):
    """End to end: `rm` through the shell still leaves a way back, and the
    agent is told what went, so it cannot quietly lose the file."""
    from rune.capabilities.bash import BashParams, bash_execute

    monkeypatch.delenv("RUNE_SHELL_SNAPSHOT", raising=False)
    res = await bash_execute(BashParams(
        command="rm notes.md", cwd=str(workspace), timeout=30))

    assert not (workspace / "notes.md").exists()
    assert "notes.md" in (res.output or "")
    assert res.metadata.get("removed_files") == ["notes.md"]
    assert ws.restore(workspace, "notes.md")
    assert (workspace / "notes.md").read_text() == "keep me\n"


@pytest.mark.asyncio
async def test_a_read_only_command_is_not_snapshotted(workspace, monkeypatch):
    from rune.capabilities.bash import BashParams, bash_execute

    monkeypatch.delenv("RUNE_SHELL_SNAPSHOT", raising=False)
    await bash_execute(BashParams(
        command="ls -la", cwd=str(workspace), timeout=30))
    assert ws.latest(workspace) is None


class TestRecoveryAcrossSnapshots:
    """A file removed two commands ago is still recoverable.

    Each mutating command takes its own snapshot, so the newest one no
    longer holds what the previous command deleted. Looking only at the
    newest copy would recover nothing but the most recent loss.
    """

    def test_a_file_deleted_before_the_last_snapshot_comes_back(self, workspace):
        ws.take(workspace)                      # holds notes.md and mod.py
        (workspace / "notes.md").unlink()
        ws.take(workspace)                      # notes.md already gone here
        (workspace / "pkg" / "mod.py").unlink()

        assert ws.restore(workspace, "notes.md")
        assert (workspace / "notes.md").read_text() == "keep me\n"
        assert ws.restore(workspace, "pkg/mod.py")

    def test_a_file_older_than_every_kept_snapshot_is_gone(self, workspace):
        # Bounded history is a deliberate trade; say so rather than imply
        # the copies are permanent.
        (workspace / "old.txt").write_text("x")
        ws.take(workspace)
        (workspace / "old.txt").unlink()
        for i in range(ws._KEEP + 1):
            (workspace / f"pad{i}.txt").write_text("p")
            ws.take(workspace)
        assert not ws.restore(workspace, "old.txt")

