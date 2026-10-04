"""Real models use the approval UI and broker; the external API is a synthetic fixture."""

import asyncio
import json
import tempfile
from pathlib import Path

from rune.connectors.broker import create_broker
from rune.connectors.models import ConnectorPolicy
from rune.connectors.store import ConnectorStore, authority_key
from rune.types import CapabilityResult
from tests.e2e import test_live_workflows as workflows
from tests.e2e.workflow_harness import gui_calls
from tests.integration.test_cloud_transport import serving

workflow = workflows.workflow


async def test_connected_api_read_uses_approval_and_broker(workflow, monkeypatch, request):
    calls = []

    async def upstream(policy, secret, action):
        assert secret == "synthetic-ledger-credential"
        assert policy.permits(action)
        calls.append(action.model_dump())
        return CapabilityResult(success=True, output='{"total": 7349}', metadata={"status_code": 200})

    monkeypatch.setattr("rune.connectors.broker.send_request", upstream)
    with tempfile.TemporaryDirectory(prefix="rune-broker-e2e-", dir="/private/tmp") as directory:
        root = Path(directory)
        monkeypatch.setenv("RUNE_BROKER_HOME", str(root))
        store = ConnectorStore(root)
        store.put(ConnectorPolicy(name="ledger", origin="https://ledger.example", path_prefix="/v1/totals", methods=["GET"]), "synthetic-ledger-credential")
        revision = store.list()[0]["revision"]

        def approve(run, *_args, **_kwargs):
            action = run["approval"].get("action", {})
            params = action.get("params", {})
            return (action.get("tool") == "connector_request" and params.get("connector") == "ledger"
                    and params.get("origin") == "https://ledger.example"
                    and params.get("revision") == revision and params.get("method") == "GET"
                    and params.get("path") == "/v1/totals" and not params.get("body") and not params.get("query"))

        monkeypatch.setattr(workflow, "_can_approve_edit", approve)
        async with serving(create_broker(store, authority_key(root, create=True)), uds=root / "broker.sock"):
            run = await asyncio.to_thread(workflow.submit, "연결된 ledger API의 /v1/totals를 읽고 total 값만 숫자로 답해줘. 커넥터 목록에서 연결을 확인하고 실제 API 응답으로 답해. 브라우저나 셸은 필요 없어.")
        workflows.completed(run)
        assert (run.get("answer") or run["text"]).strip().replace(",", "") == "7349"
        assert len(calls) == 1 and not gui_calls(run)
        assert "synthetic-ledger-credential" not in json.dumps(workflow.runs)
        assert workflow.runs[0]["approved_fixture_edits"]
        request.node.live_passed = True
