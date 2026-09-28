"""Uploads survive retries without leaking across conversations or duplicating bytes."""

import base64
import json
import sqlite3
from types import SimpleNamespace

import pytest

from rune.api.attachment_store import AttachmentStore
from rune.api.run_admission import RunAdmission
from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from rune.api import conversation_wiring

    monkeypatch.setattr(conversation_wiring, "get_conv_manager", lambda: None)
    store = RunStore(tmp_path / "conversation.db")
    runs = RunSnapshots(store)
    runs.open()
    yield store, runs, RunAdmission(runs, store, SimpleNamespace(entries={}))
    runs.close()


def file(data=b"name,value\na,12\n"):
    return {"name": "rows.csv", "mimeType": "text/csv", "data": base64.b64encode(data).decode()}


async def test_upload_and_run_commit_together_and_references_survive_restart(storage):
    store, runs, admission = storage
    run_id, _, _ = await admission.accept("read", "one", request_id="req", attachments=[file()])
    saved = runs.get(run_id)
    refs = saved["execution"]["attachments"]
    assert "data" not in refs[0]
    assert AttachmentStore(store.db).hydrate("one", refs)[0]["data"] == file()["data"]
    runs.record("agent_complete", {"runId": run_id})
    runs.close()
    runs.open()
    replay, _, reused = await admission.accept("read", "one", request_id="req", attachments=refs)
    assert reused and replay == run_id
    new_id, _, reused = await admission.accept("read", "one", request_id="new", attachments=[file()])
    assert not reused and new_id != run_id
    assert store.db.execute("SELECT COUNT(*) FROM web_attachments").fetchone()[0] == 1
    assert len(json.dumps(runs.get(new_id))) < 2000
    with pytest.raises(ValueError, match="different content"):
        await admission.accept("read", "one", request_id="req", attachments=[file(b"changed")])
    with pytest.raises(ValueError, match="not available"):
        await admission.accept("read", "other", request_id="req", attachments=refs)


async def test_rejected_request_does_not_store_its_upload(storage):
    store, runs, admission = storage
    await admission.accept("running", "one")
    with pytest.raises(ValueError, match="still working"):
        await admission.accept("next", "one", attachments=[file()])
    assert store.db.execute("SELECT COUNT(*) FROM web_attachments").fetchone()[0] == 0


async def test_duplicate_files_and_deleted_conversation(storage):
    from rune.conversation.store import ConversationStore

    store, runs, admission = storage
    run_id, _, _ = await admission.accept("read", "one", request_id="r1", attachments=[file(), file()])
    assert store.db.execute("SELECT COUNT(*) FROM web_attachments").fetchone()[0] == 1
    runs.record("agent_complete", {"runId": run_id})
    conversations = ConversationStore(store._path)
    await conversations.delete("one")
    assert store.db.execute("SELECT COUNT(*) FROM web_attachments").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM web_run_requests").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM web_runs").fetchone()[0] == 0
    conversations._conn.close()


async def test_atomic_acceptance_rolls_back_upload_on_conflict(storage):
    store, runs, admission = storage
    run_id, _, _ = await admission.accept("first", "one", request_id="same")
    uploads = AttachmentStore(store.db).prepare("one", [file()])
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        runs.start("failed", "one", "second", request=("same", "hash"), uploads=uploads)
    assert store.db.execute("SELECT COUNT(*) FROM web_attachments").fetchone()[0] == 0
    assert store.latest_id("one") == run_id


def test_workspace_copy_reuses_original_but_never_overwrites_user_edits(tmp_path):
    from pathlib import Path

    from rune.agent.document_attachments import save_document

    reference = "a" * 32
    original = Path(save_document("rows.csv", file()["data"], str(tmp_path), reference))
    assert save_document("rows.csv", file()["data"], str(tmp_path), reference) == str(original)
    original.write_text("user edit")
    replacement = Path(save_document("rows.csv", file()["data"], str(tmp_path), reference))
    assert replacement != original and original.read_text() == "user edit"
    assert replacement.read_bytes() == base64.b64decode(file()["data"])


async def test_corrupt_saved_upload_stops_before_model_input(storage):
    store, runs, admission = storage
    run_id, _, _ = await admission.accept("read", "one", attachments=[file()])
    refs = runs.get(run_id)["execution"]["attachments"]
    store.db.execute("UPDATE web_attachments SET content = ?", (b"corrupted",))
    with pytest.raises(ValueError, match="integrity check"):
        AttachmentStore(store.db).hydrate("one", refs)
