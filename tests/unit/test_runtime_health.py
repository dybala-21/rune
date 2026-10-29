"""Readiness can fail while the HTTP server remains alive."""

from tests.unit.test_api_server_multiturn import client as client
from tests.unit.test_api_server_multiturn import isolated_wiring as isolated_wiring


def test_storage_degradation_does_not_masquerade_as_a_dead_process(client):
    client.app.state.runtime_health = lambda: {
        "status": "degraded", "checks": {"daemon": True, "storage": False},
    }
    live = client.get("/health")
    ready = client.get("/ready")
    assert live.status_code == 200 and live.json()["ready"] is False
    assert ready.status_code == 503
    client.app.state.runtime_health = lambda: {
        "status": "ok", "checks": {"daemon": True, "storage": True},
    }
    assert client.get("/api/v1/ready").status_code == 200


def test_shutdown_stops_advertising_readiness(client):
    client.app.state.runtime_health = lambda: {
        "status": "degraded", "checks": {"daemon": False},
    }
    assert client.get("/health").status_code == 503
    assert client.get("/ready").status_code == 503


def test_optional_connection_failure_does_not_block_local_readiness(client):
    client.app.state.runtime_health = lambda: {
        "status": "ok", "checks": {"daemon": True},
        "connections": {"mcp": "failed", "channels": "connecting"},
    }
    result = client.get("/ready")
    assert result.status_code == 200
    assert result.json()["connections"] == {"mcp": "failed", "channels": "connecting"}
