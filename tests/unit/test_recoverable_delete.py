"""Destructive file operations stay recoverable, and mutations are verified.

The failure this guards against: asked to tidy a directory, an agent
deleted a month-end database snapshot, an irreproducible notebook and an
incident log, then reported success. Build output can be rebuilt; the
user's files cannot, so they go to the trash instead of vanishing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rune.capabilities.file import (
    FileDeleteParams,
    FileEditParams,
    FileWriteParams,
    file_delete,
    file_edit,
    file_write,
)
from rune.safety.recoverable import (
    TRASH_DIRNAME,
    is_regenerable,
    move_to_trash,
    verify_dir,
    verify_gone,
    verify_written,
)


@pytest.fixture(autouse=True)
def _allow_guardian(monkeypatch):
    """Guardian refuses paths under /var, where pytest puts tmp_path."""
    from rune.safety import guardian as g

    class _OK:
        allowed = True
        reason = ""

    monkeypatch.setattr(g.Guardian, "validate_file_path", lambda self, p: _OK())


@pytest.fixture(autouse=True)
def _in_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class TestRegenerableClassification:
    def test_build_and_os_cruft_is_regenerable(self):
        assert is_regenerable(Path("build/out.o"))
        assert is_regenerable(Path("pkg/mod.pyc"))
        assert is_regenerable(Path("a/.DS_Store"))
        assert is_regenerable(Path("x/__pycache__/y.pyc"))

    def test_user_data_is_not_regenerable(self):
        assert not is_regenerable(Path("data/backup_2026-06-30.sql"))
        assert not is_regenerable(Path("notebooks/analysis.ipynb"))
        assert not is_regenerable(Path("tmp/manual_fix_notes.md"))
        assert not is_regenerable(Path("run.log"))


class TestMoveToTrash:
    def test_bytes_survive_the_move(self, tmp_path):
        src = tmp_path / "precious.sql"
        src.write_text("INSERT INTO x VALUES (1);")
        entry = move_to_trash(src)
        assert not src.exists()
        assert Path(entry.stored).read_text() == "INSERT INTO x VALUES (1);"
        assert TRASH_DIRNAME in entry.stored

    def test_same_name_twice_does_not_collide(self, tmp_path):
        first = tmp_path / "notes.md"
        first.write_text("one")
        a = move_to_trash(first)
        first.write_text("two")
        b = move_to_trash(first)
        assert a.stored != b.stored
        assert Path(a.stored).read_text() == "one"
        assert Path(b.stored).read_text() == "two"


@pytest.mark.asyncio
class TestFileDelete:
    async def test_user_file_goes_to_trash_and_is_recoverable(self, tmp_path):
        target = tmp_path / "data" / "backup.sql"
        target.parent.mkdir()
        target.write_text("-- month end snapshot")
        res = await file_delete(FileDeleteParams(path=str(target)))
        assert res.success
        assert not target.exists()
        stored = Path(res.metadata["trashed"])
        assert stored.read_text() == "-- month end snapshot"

    async def test_build_output_is_deleted_outright(self, tmp_path):
        target = tmp_path / "build" / "tmp.o"
        target.parent.mkdir()
        target.write_text("junk")
        res = await file_delete(FileDeleteParams(path=str(target)))
        assert res.success
        assert res.metadata["trashed"] is None
        assert not target.exists()

    async def test_directory_of_user_files_is_recoverable(self, tmp_path):
        d = tmp_path / "notebooks"
        d.mkdir()
        (d / "analysis.ipynb").write_text("{}")
        res = await file_delete(FileDeleteParams(path=str(d), recursive=True))
        assert res.success
        stored = Path(res.metadata["trashed"])
        assert (stored / "analysis.ipynb").read_text() == "{}"

    async def test_trash_can_be_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNE_TRASH", "0")
        target = tmp_path / "gone.txt"
        target.write_text("x")
        res = await file_delete(FileDeleteParams(path=str(target)))
        assert res.success
        assert res.metadata["trashed"] is None
        assert not target.exists()


@pytest.mark.asyncio
class TestTestFileProtection:
    async def test_rewriting_an_existing_test_is_blocked(self, tmp_path):
        t = tmp_path / "tests" / "test_fees.py"
        t.parent.mkdir()
        t.write_text("def test_a():\n    assert late_fee(100000, 21) == 8000\n")
        res = await file_write(FileWriteParams(
            path=str(t), content="def test_a():\n    assert True\n"))
        assert not res.success
        assert "existing test" in (res.error or "")
        assert "8000" in t.read_text()  # untouched

    async def test_editing_an_existing_test_is_blocked(self, tmp_path):
        t = tmp_path / "tests" / "test_fees.py"
        t.parent.mkdir()
        t.write_text("assert late_fee(100000, 21) == 8000\n")
        res = await file_edit(FileEditParams(
            path=str(t), search="8000", replace="6000"))
        assert not res.success
        assert "8000" in t.read_text()

    async def test_new_test_files_are_allowed(self, tmp_path):
        t = tmp_path / "tests" / "test_new.py"
        t.parent.mkdir()
        res = await file_write(FileWriteParams(
            path=str(t), content="def test_x():\n    assert 1\n"))
        assert res.success
        assert t.is_file()

    async def test_source_files_are_unaffected(self, tmp_path):
        s = tmp_path / "pkg" / "mod.py"
        s.parent.mkdir()
        s.write_text("x = 1\n")
        res = await file_write(FileWriteParams(path=str(s), content="x = 2\n"))
        assert res.success
        assert s.read_text() == "x = 2\n"

    async def test_opt_out_allows_editing_tests(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNE_PROTECT_TESTS", "0")
        t = tmp_path / "tests" / "test_a.py"
        t.parent.mkdir()
        t.write_text("assert 1\n")
        res = await file_write(FileWriteParams(path=str(t), content="assert 2\n"))
        assert res.success


class TestPostconditions:
    def test_write_check_reads_back_from_disk(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hello")
        assert verify_written(p, 5).ok
        assert not verify_written(tmp_path / "missing.txt", 5).ok

    def test_empty_after_a_nonempty_write_is_a_failure(self, tmp_path):
        p = tmp_path / "b.txt"
        p.write_text("")
        assert not verify_written(p, 10).ok

    def test_delete_and_dir_checks(self, tmp_path):
        p = tmp_path / "c.txt"
        p.write_text("x")
        assert not verify_gone(p).ok
        p.unlink()
        assert verify_gone(p).ok
        # a silently failed mkdir must not read as success
        assert not verify_dir(tmp_path / "nope").ok
        assert verify_dir(tmp_path).ok

    def test_verification_can_be_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNE_VERIFY_MUTATIONS", "0")
        assert verify_written(tmp_path / "missing.txt", 5).ok
