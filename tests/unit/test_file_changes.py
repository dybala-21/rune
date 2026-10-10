from pathlib import Path
from types import SimpleNamespace

import pytest

from rune.capabilities.file import (
    FileDeleteParams,
    FileEditParams,
    FileReadParams,
    FileWriteParams,
    file_delete,
    file_edit,
    file_read,
    file_write,
)


@pytest.fixture(autouse=True)
def authorized_workspace(tmp_path, monkeypatch):
    def validate(path):
        return SimpleNamespace(allowed=Path(path).resolve().is_relative_to(tmp_path.resolve()),
                               requires_approval=False, reason="outside test workspace")

    monkeypatch.setattr("rune.capabilities.file.get_guardian", lambda: SimpleNamespace(
        validate_file_path=validate, validate_file_read_path=validate))


async def test_actual_changes_cover_create_replace_and_delete_without_git(tmp_path, monkeypatch):
    from rune.agent.execution_journal import ExecutionJournal
    from rune.agent.loop import _tool_result_event_payload
    from rune.api.run_snapshot import RunSnapshots
    from rune.api.run_store import RunStore

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    path = tmp_path / "계산.py"
    store = RunStore(tmp_path / "runs.db")
    runs = RunSnapshots(store)
    runs.start("r1", "s1", "코드 수정")
    runs.record("run_context", {"runId": "r1", "workspace": str(tmp_path), "recoveryVersion": 1})
    journal = ExecutionJournal(store, "r1", str(tmp_path))
    try:
        created = await file_write(FileWriteParams(path=str(path), content="value = 1\nvalue = 1\n"))
        assert created.success, created.error
        assert created.metadata["fileChange"]["kind"] == "created"
        assert "--- /dev/null" in created.metadata["fileChange"]["patch"]
        params = {"path": str(path), "search": "value = 1", "replace": "value = 2", "all": True}
        changed = await journal.execute("file_edit", params, lambda: file_edit(FileEditParams(**params)))
        assert changed.success, changed.error
        patch = changed.metadata["fileChange"]["patch"]
        assert patch.count("-value = 1") == 2 and patch.count("+value = 2") == 2
        assert _tool_result_event_payload("file_edit", changed)["fileChange"]["patch"] == patch
        # The tool receipt survives even if the UI event was never emitted.
        runs.interrupt_active("server_shutdown")
        runs.close()
        runs = RunSnapshots(RunStore(tmp_path / "runs.db"))
        assert runs.get("r1")["fileChanges"][0]["patch"] == patch
        path.write_text("user_changed = True\n")
        assert runs.get("r1")["fileChanges"][0]["patch"] == patch
        runs.start("r2", "s1", "코드 수정", parent_id="r1")
        assert runs.get("r2")["fileChanges"][0]["patch"] == patch
        deleted = await file_delete(FileDeleteParams(path=str(path)))
        assert deleted.success, deleted.error
        assert deleted.metadata["fileChange"]["kind"] == "deleted"
        assert "-user_changed = True" in deleted.metadata["fileChange"]["patch"]
    finally:
        runs.close()


async def test_failed_edit_does_not_publish_a_proposed_diff(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("answer = 1\n")
    result = await file_edit(FileEditParams(path=str(path), search="answer = 1", replace="answer = ("))
    assert not result.success
    assert "syntax error" in result.output
    assert "fileChange" not in (result.metadata or {})
    assert path.read_text() == "answer = 1\n"


@pytest.mark.parametrize("body", ["a" * 300_000, "binary\0value"])
def test_unavailable_previews_are_explicit_and_bounded(tmp_path, body):
    from rune.capabilities.file_changes import file_change

    path = tmp_path / "large.dat"
    path.write_text(body)
    change = file_change(path, "", existed=False)
    assert change["patch"] == "" and "unavailable" in change["notice"]


@pytest.mark.parametrize("extension", [".docx", ".XLSX", ".pptx"])
async def test_text_tools_cannot_create_or_overwrite_office_files(tmp_path, extension):
    path = tmp_path / ("document" + extension)
    params = FileWriteParams(path=str(path), content="plain text")
    rejected = await file_write(params)
    assert not rejected.success and not path.exists()
    assert rejected.metadata["action_status"] == "not_executed"
    assert "document_create" in rejected.error
    path.write_bytes(b"PK\x03\x04\xffexisting document")
    before = path.read_bytes()
    assert not (await file_write(params)).success
    assert not (await file_edit(FileEditParams(path=str(path), search="document", replace="text"))).success
    assert path.read_bytes() == before
    path.write_text("This is not a valid Office file")
    read = await file_read(FileReadParams(path=str(path)))
    assert not read.success and "document_read" in read.error


async def test_office_guard_checks_both_symlink_names(tmp_path):
    target, alias = tmp_path / "report.xlsx", tmp_path / "alias.txt"
    target.write_text("unchanged")
    alias.symlink_to(target)
    assert not (await file_write(FileWriteParams(path=str(alias), content="changed"))).success
    target.rename(tmp_path / "report.txt")
    alias.unlink()
    alias = tmp_path / "alias.xlsx"
    alias.symlink_to(tmp_path / "report.txt")
    assert not (await file_write(FileWriteParams(path=str(alias), content="changed"))).success
    assert (tmp_path / "report.txt").read_text() == "unchanged"
