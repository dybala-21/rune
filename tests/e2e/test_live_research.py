"""Research through frame-based sources, including gaps in recent coverage."""

import httpx
import pytest

from rune.agent.source_values import numeric_literals
from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow
from tests.e2e.workflow_harness import gui_calls


@pytest.mark.parametrize("recent", [True, False], ids=["recent-notice", "historical-only"])
@pytest.mark.parametrize("workflow", [{"max_tokens": 1024, "model_request_limit": 8,
                                        "token_budget_override": 50000, "cost_budget_usd": .30}], indirect=True)
def test_research_reads_frames_and_reports_source_limits(workflow, monkeypatch, request, recent):
    from datetime import UTC, datetime
    from decimal import Decimal

    from rune.capabilities.search.provider import SearchResult

    date = datetime.now(UTC).strftime("%Y-%m-%d") if recent else "2024-04-05"
    pages = {
        "/": '<html><frameset><frame src="/main/index"></frameset><noframes><body></body></noframes></html>',
        "/main/index": '<html><body><table><tr><td><iframe src="../news"></iframe></td></tr></table></body></html>',
        "/news": f'<html><main><h1>Harbor Quest maintenance complete</h1><time>{date}</time>'
                 '<p>Notice ID: HQ-217. Service status: login_open. '
                 'Login has reopened after server maintenance. No player statistics were published.</p></main></html>',
    }
    reads = []
    original_send = httpx.AsyncClient.send

    async def send(client, req, *args, **kwargs):
        if req.url.host == "research-fixture.example":
            reads.append(req.url.path)
            return httpx.Response(200 if req.url.path in pages else 404, text=pages.get(req.url.path, "Not found"),
                                  headers={"content-type": "text/html"}, request=req)
        return await original_send(client, req, *args, **kwargs)

    class Search:
        name = "fixture"

        async def search(self, options):
            return [SearchResult(title="Harbor Quest official website", url="https://research-fixture.example/",
                                 description="Service notices. An undated excerpt mentions 3,200 concurrent players.")]

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr("rune.capabilities.web.get_search_provider", lambda: Search())
    run = workflow.submit("Harbor Quest 게임 근황을 조사해줘. 서비스 운영, 최근 공지, 이용자 동향을 출처와 날짜를 붙여 정리해줘. "
                          "날짜 형식은 YYYY-MM-DD로 써줘. 공지 본문의 Notice ID와 Service status 값도 그대로 포함해줘.")
    completed(run)
    answer = run.get("answer") or run["text"]
    assert "/news" in reads and date in answer
    assert "HQ-217" in answer and "login_open" in answer
    assert "https://research-fixture.example/" in answer
    assert Decimal("3200") not in numeric_literals(answer)
    assert not gui_calls(run)
    assert not any(call["toolName"] == "task_blocked" for call in run["toolCalls"])
    request.node.live_passed = True
