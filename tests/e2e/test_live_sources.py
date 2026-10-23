"""Check source selection with real models and controlled HTTP responses."""

import json
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from rune.agent.source_values import numeric_literals
from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow
from tests.e2e.workflow_harness import gui_calls

SOURCE_URL = "https://source-fixture.example/observations"
OBSERVED_AT = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@pytest.fixture
def source_responses(monkeypatch):
    from rune.capabilities.search.provider import SearchResult
    from rune.config import get_config

    cfg = get_config()
    cfg.llm.cost_budget_usd = .30
    cfg.llm.model_request_limit = 7
    records = [{"station": f"Station-{i}", "value": 999, "observed_at": OBSERVED_AT} for i in range(400)]
    records.extend([
        {"station": "North", "value": 120, "observed_at": "2020-01-01T00:00:00Z"},
        {"station": "North", "value": 17.125, "observed_at": OBSERVED_AT},
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
        "station": "North", "value": 17.125, "unit": "celsius", "observed_at": OBSERVED_AT,
    }
    assert source_responses and not gui_calls(run)
    assert len(source_responses) <= 3
    request.node.live_passed = True


@pytest.mark.parametrize("workflow", [{"max_tokens": 768}], indirect=True)
def test_original_source_fields_survive_a_followup_without_refetch(source_responses, workflow, request):
    from rune.config import get_config

    get_config().llm.cost_budget_usd = .25
    get_config().llm.model_request_limit = 5
    expected = {"station":"North", "value":17.125, "unit":"celsius", "observed_at":OBSERVED_AT}
    first = workflow.submit(f"{SOURCE_URL}에서 North의 최신 관측 기록을 확인해. "
        "원문의 station, value, unit, observed_at 네 필드만 JSON으로 반환해. 상위 단위도 포함하고 설명이나 코드블록은 빼줘.")
    completed(first)
    assert json.loads((first.get("answer") or first["text"]).strip()) == expected
    reads = len(source_responses)
    acknowledgement = workflow.submit("고마워. 추가 작업 없이 짧게 대답만 해줘.")
    completed(acknowledgement)
    assert len(source_responses) == reads and not gui_calls(acknowledgement)
    follow = workflow.submit("방금 확인한 같은 기록의 원문 네 필드를 JSON으로 다시 보여줘. 코드블록이나 설명은 빼줘.")
    completed(follow)
    assert json.loads((follow.get("answer") or follow["text"]).strip()) == expected
    assert len(source_responses) == reads and not gui_calls(first) and not gui_calls(follow)
    assert workflow.decisions[-1]["classification"]["resume_task"], "The follow-up must select the saved task"
    review = follow["trust"]["sourceReview"]
    assert review["reusedSources"] > 0 and review["freshSources"] == 0
    assert review["status"] == "supported" and review["scope"] == "record_fields"
    request.node.live_passed = True


@pytest.mark.parametrize("workflow", [{"max_tokens": 1024}], indirect=True)
def test_casual_concern_checks_sources_and_keeps_context(source_responses, workflow, request):
    run = workflow.submit(
        "North 관측소 수치가 최근 120까지 올랐다던데 걱정되네. 지금 상황 좀 확인해줘.",
        question_answers=(f"{SOURCE_URL}의 North 기상 관측소를 말한 거야. 섭씨 기온을 확인해줘.",),
    )
    completed(run)
    answer = run.get("answer") or run["text"]
    assert Decimal("17.125") in numeric_literals(answer), answer
    assert source_responses and not gui_calls(run)
    assert any(item["classification"]["needs_current_source"] for item in workflow.decisions)
    assert len(source_responses) <= 3

    follow = workflow.submit(
        "방금 확인한 최신 관측 기록만 다시 정리해줘. 원문의 station, value, unit, observed_at 네 필드만 "
        "JSON으로 반환해. 단위가 상위 필드에 있으면 포함해. 설명이나 코드블록은 빼줘."
    )
    completed(follow)
    assert json.loads((follow.get("answer") or follow["text"]).strip()) == {
        "station":"North", "value":17.125, "unit":"celsius", "observed_at":OBSERVED_AT,
    }
    assert not gui_calls(follow) and len(source_responses) <= 6
    request.node.live_passed = True


@pytest.mark.parametrize("workflow", [{"max_tokens": 768}], indirect=True)
def test_public_status_matches_the_retrieved_response(workflow, monkeypatch, request):
    from rune.config import get_config

    get_config().llm.cost_budget_usd = .10
    get_config().llm.model_request_limit = 7
    source_url = "https://www.githubstatus.com/api/v2/status.json"
    responses = []
    original_send = httpx.AsyncClient.send

    async def send(client, req, *args, **kwargs):
        response = await original_send(client, req, *args, **kwargs)
        if str(req.url) == source_url and response.status_code == 200:
            await response.aread()
            responses.append(response.json())
        return response

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    run = workflow.submit(
        f"{source_url}에서 GitHub의 현재 서비스 상태를 확인해줘. 원문의 status와 page.updated_at을 참고해서 "
        "indicator, description, updated_at 세 필드만 JSON으로 반환해. 설명이나 코드블록은 빼줘."
    )
    completed(run)
    assert responses and not gui_calls(run)
    observed = responses[-1]
    assert json.loads((run.get("answer") or run["text"]).strip()) == {
        **observed["status"], "updated_at":observed["page"]["updated_at"],
    }
    request.node.live_passed = True
