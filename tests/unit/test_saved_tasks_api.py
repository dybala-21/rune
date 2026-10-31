"""List saved tasks without starting a run."""

from uuid import uuid4

from rune.memory.task_store import TaskStore
from rune.utils.paths import user_workspace
from tests.unit.test_api_server_multiturn import FakeLoop
from tests.unit.test_api_server_multiturn import client as client
from tests.unit.test_api_server_multiturn import isolated_wiring as isolated_wiring


def task(goal):
    return {"id": uuid4().hex[:12], "goal": goal, "status": "pending", "kind": "manual"}


def test_saved_task_api_scopes_search_pages_without_executing(client, tmp_path):
    workspace = str(user_workspace().resolve())
    store = TaskStore()
    visible = [task(f"Orchid report {i}") for i in range(12)]
    store.create("local", workspace, visible)
    store.create("channel:other", workspace, [task("Private orchid")])
    store.create("local", str(tmp_path / "other"), [task("Other workspace orchid")])
    def read(**extra):
        response = client.get("/api/tasks", params={"sessionId": "new-chat", **extra})
        assert response.status_code == 200, response.text
        return response.json()
    first, second = read(), read(offset=10)
    assert first["hasMore"] and not second["hasMore"]
    assert {row["id"] for page in (first, second) for row in page["tasks"]} == {row["id"] for row in visible}
    assert len(read(query="Orchid")["tasks"]) == 10
    assert not read(query="nonexistentmarker")["tasks"]
    assert not FakeLoop.captured
    assert client.get("/api/tasks", params={"sessionId": "new-chat", "offset": -1}).status_code == 422


def test_saved_task_api_rejects_nonlocal_conversation_and_uses_pinned_workspace(client, isolated_wiring, tmp_path):
    manager = client.portal.call(isolated_wiring.get_conv_manager)
    remote = manager.start_conversation("slack:other")
    assert client.get("/api/tasks", params={"sessionId": remote.id}).status_code == 403
    local = manager.start_conversation("web:local")
    root = tmp_path / "pinned"
    root.mkdir()
    client.portal.call(isolated_wiring.set_workspace, local.id, str(root))
    wanted = task("Pinned report")
    TaskStore().create("local", str(root.resolve()), [wanted])
    response = client.get("/api/tasks", params={"sessionId": local.id})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["tasks"]] == [wanted["id"]]
