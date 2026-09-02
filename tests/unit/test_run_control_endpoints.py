"""Stopping a run, and finding out whether one is still going.

Stop sent no run id, so the server cancelled whichever run was newest — which
in a second tab, or against a scheduled run, was not the one the user stopped.
And a client whose event stream dropped mid-run had no way to learn the run had
ended, so the composer stayed disabled until the page was reloaded.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from rune.api.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    app = create_app()
    # Loopback address so the localhost auth bypass applies.
    with TestClient(app, client=("127.0.0.1", 50000)) as tc:
        yield tc


class TestAbort:
    def test_naming_a_run_that_is_not_active_reports_it(self, client):
        body = client.post("/api/abort", json={"runId": "no-such-run"}).json()

        assert body["ok"] is True
        assert body["stopped"] is False
        assert "not active" in body["reason"]

    def test_omitting_the_run_id_still_works_for_older_clients(self, client):
        body = client.post("/api/abort", json={}).json()

        assert body["ok"] is True
        assert "stopped" not in body


class TestActiveRuns:
    def test_it_reports_nothing_running(self, client):
        body = client.post(
            "/api/v1/rpc", json={"method": "runs.active", "params": {}}
        ).json()

        assert body["success"] is True
        assert body["data"]["runIds"] == []
