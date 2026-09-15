import hashlib
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from rune.api.files import download_file
from rune.computer.artifacts import Artifacts


@pytest.mark.parametrize("change", ["original", "blob", "symlink", "conversation"])
async def test_download_is_bound_to_published_content_and_conversation(tmp_path, monkeypatch, change):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "rune"))
    monkeypatch.setattr("rune.api.conversation_wiring.get_workspace", AsyncMock(return_value=None))
    file = tmp_path / "출시 계획.rtf"
    content = b"saved original"
    file.write_bytes(content)
    store = Artifacts()
    receipt = store.publish("one", "run-1", str(file), content, hashlib.sha256(content).hexdigest())
    blob = store.directory("one") / f"{receipt['id']}.blob"
    if change == "original":
        file.write_bytes(b"later user edit")
        response = await download_file("one", str(file))
        assert response.body == content
        return
    if change == "blob":
        blob.write_bytes(b"corrupt")
    if change == "symlink":
        blob.unlink()
        blob.symlink_to(file)
    with pytest.raises(HTTPException) as error:
        await download_file("other" if change == "conversation" else "one", str(file))
    assert error.value.status_code == (404 if change == "conversation" else 409)


async def test_unregistered_desktop_file_is_not_downloadable(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "rune"))
    monkeypatch.setattr("rune.api.conversation_wiring.get_workspace", AsyncMock(return_value=str(tmp_path / "workspace")))
    file = tmp_path / "private.rtf"
    file.write_bytes(b"not shared")
    with pytest.raises(HTTPException) as error:
        await download_file("one", str(file))
    assert error.value.status_code == 403
