"""Check source selection with real models and controlled HTTP responses."""

import json

import httpx
import pytest

from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow
from tests.e2e.workflow_harness import gui_calls

SOURCE_URL = "https://source-fixture.example/observations"


@pytest.fixture
def source_responses(monkeypatch):
    from rune.capabilities.search.provider import SearchResult
    from rune.config import get_config

    cfg = get_config()
    cfg.llm.cost_budget_usd = .20
    cfg.llm.model_request_limit = 7
    records = [{"station": f"Station-{i}", "value": 999, "observed_at": "2026-10-16T07:00:00Z"} for i in range(400)]
    records.extend([
        {"station": "North", "value": 120, "observed_at": "2020-01-01T00:00:00Z"},
        {"station": "North", "value": 17.125, "observed_at": "2026-10-16T07:00:00Z"},
    ])
    payload = {"unit": "celsius", "observations": records}
    requests = []
    original_send = httpx.AsyncClient.send

    async def send(client, request, *args, **kwargs):
        if request.url.host == "source-fixture.example":
            requests.append(str(request.url))
            return httpx.Response(200, json=payload, request=request)
        return await original_send(client, request, *args, **kwargs)

    class Search:
        name = "fixture"

        async def search(self, options):
            return [SearchResult(title="Indexed observation", url=SOURCE_URL,
                                 description="North: 120 celsius. Check the source for the observation time.")]

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr("rune.capabilities.web.get_search_provider", lambda: Search())
    return requests


@pytest.mark.parametrize("workflow", [{"max_tokens": 768}], indirect=True)
def test_latest_observation_keeps_its_entity_unit_and_time(source_responses, workflow, request):
    run = workflow.submit(
        f"{SOURCE_URL} 자료에서 North 관측소의 가장 최신 측정값을 확인해줘. "
        "다른 관측소나 과거 값과 섞지 말고, 원문의 station, value, unit, observed_at 네 필드만 JSON으로 반환해. "
        "단위가 상위 필드에 있으면 해당 단위를 사용해. 설명이나 코드블록 없이 JSON 본문만 반환해."
    )
    completed(run)
    assert json.loads((run.get("answer") or run["text"]).strip()) == {
        "station": "North", "value": 17.125, "unit": "celsius", "observed_at": "2026-10-16T07:00:00Z",
    }
    assert source_responses and not gui_calls(run)
    assert len(source_responses) <= 3
    request.node.live_passed = True
