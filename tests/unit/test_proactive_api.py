"""User responses reach durable state instead of a placeholder acknowledgement."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from rune.api.handlers.proactive import router
from rune.memory.store import MemoryStore
from rune.proactive.engine import ProactiveEngine
from rune.proactive.types import Suggestion


@pytest.fixture
def api(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory.db")
    engine = ProactiveEngine()
    engine.load_persisted_suggestions(store)
    engine.add_suggestion(Suggestion(id="report", title="Report", description="Inspect the supplied CSV"))
    monkeypatch.setattr("rune.proactive.engine._engine", engine)
    monkeypatch.setattr("rune.memory.store.get_memory_store", lambda: store)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        yield client, engine
    store.close()


def test_response_roundtrip_and_conflicts(api):
    client, engine = api
    prefix = "/api/v1/proactive"
    assert client.get(f"{prefix}/suggestions").json()["pendingSuggestions"][0]["id"] == "report"
    data = {"suggestionId": "report", "response": "accept"}
    for _ in range(2):
        result = client.post(f"{prefix}/feedback", json=data)
        assert result.status_code == 200
        assert result.json()["executionStatus"] == "queued"
    assert engine.get_stats()["interaction_count"] == 1
    assert client.get(f"{prefix}/suggestions").json()["pendingSuggestions"] == []
    data["response"] = "dismiss"
    assert client.post(f"{prefix}/feedback", json=data).status_code == 409
    data["suggestionId"] = "missing"
    assert client.post(f"{prefix}/feedback", json=data).status_code == 404
    engine.record_execution("report", "unverified", {"output": "Created, awaiting verification"})
    snapshot = client.get(f"{prefix}/suggestions/report").json()
    assert snapshot["executionStatus"] == "unverified"
    assert snapshot["response"] == "accepted"
    assert snapshot["result"]["output"] == "Created, awaiting verification"


def test_cross_origin_response_is_rejected(api):
    client, engine = api
    response = client.post("/api/v1/proactive/feedback", json={"suggestionId": "report", "response": "accept"},
                           headers={"Origin": "https://untrusted.example", "Sec-Fetch-Site": "cross-site"})
    assert response.status_code in (401, 403)
    assert engine.get_stats()["interaction_count"] == 0


def test_feed_restores_offline_proposals_and_outcomes_without_execution(api, monkeypatch):
    from datetime import UTC, datetime, timedelta

    client, engine = api
    engine.handle_response("report", True)
    engine.record_execution("report", "completed", {"output": "2450", "usage": {"private": "not needed by cards"}})
    engine.add_suggestion(Suggestion(id="offline", title="New report"))
    engine.add_suggestion(Suggestion(id="dismissed", title="Unwanted action"))
    engine.handle_response("dismissed", False)
    engine.add_suggestion(Suggestion(id="expired", title="Old offer", expires_at=datetime.now(UTC) - timedelta(seconds=1)))
    restored = ProactiveEngine()
    restored.load_persisted_suggestions(engine._store)
    monkeypatch.setattr("rune.proactive.engine._engine", restored)
    before = restored.get_stats()["interaction_count"]
    for _ in range(2):
        response = client.get("/api/v1/proactive/feed")
        assert response.status_code == 200
        items = {item["id"]: item for item in response.json()["suggestions"]}
        assert items["offline"]["response"] == "pending"
        assert items["report"]["executionStatus"] == "completed"
        assert items["report"]["result"] == {"output": "2450"}
        assert items["dismissed"]["response"] == "dismissed"
        assert items["expired"]["response"] == "expired"
    assert restored.get_stats()["interaction_count"] == before
    assert restored.get_suggestion("offline").execution_status is None


def test_feed_is_bounded_and_preserves_creation_order(api):
    from datetime import UTC, datetime, timedelta

    client, engine = api
    now = datetime.now(UTC)
    engine.add_suggestion(Suggestion(id="new", title="Latest", created_at=now + timedelta(seconds=1)))
    engine.add_suggestion(Suggestion(id="old", title="Older", created_at=now - timedelta(days=1)))
    items = client.get("/api/v1/proactive/feed?limit=2").json()["suggestions"]
    assert [item["id"] for item in items] == ["report", "new"]
    assert client.get("/api/v1/proactive/feed?limit=101").status_code == 422
