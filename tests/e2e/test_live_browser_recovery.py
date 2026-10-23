"""Recover a rejected browser save without losing the draft or repeating a write."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.e2e.acceptance_fixtures import BOOKING_HTML
from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow
from tests.e2e.workflow_harness import gui_calls


@pytest.fixture
def unavailable_booking_site():
    state = {"loads": 0, "attempts": [], "saves": []}
    html = BOOKING_HTML.replace(
        "if(response.ok){",
        """if(!response.ok){
            document.getElementById('status').textContent = 'Service unavailable. Nothing was saved. Retry once.';
            const dialog = document.getElementById('confirm');
            const alert = document.createElement('p');
            alert.setAttribute('role', 'alert');
            alert.textContent = 'Service unavailable. Nothing was saved. Retry once.';
            dialog.appendChild(alert);
        }
        if(response.ok){""",
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            if self.path != "/booking":
                self.send_error(404)
                return
            state["loads"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["attempts"].append(data)
            rejected = len(state["attempts"]) == 1
            if not rejected:
                state["saves"].append(data)
            self.send_response(503 if rejected else 200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/booking", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_rejected_save_recovers_and_followup_keeps_draft(workflow, unavailable_booking_site, request):
    from rune.config import get_config

    get_config().llm.cost_budget_usd = .70
    get_config().llm.model_request_limit = 14
    url, state = unavailable_booking_site
    first = workflow.submit(
        f"브라우저에서 {url} 연습 화면을 열어. 오전 10시에 6명이 사용할 수 있고 프로젝터가 있는 "
        "회의실을 찾아 제목 Release review로 임시 예약해줘. 화면에서 검토·저장을 거쳐 저장 결과를 확인해. "
        "일시적인 오류로 저장되지 않았다고 나오면 현재 입력을 유지해서 한 번만 재시도해. "
        "성공한 저장은 중복하지 마."
    )
    completed(first)
    draft = {"room": "Cedar", "time": "10:00", "people": "6", "title": "Release review"}
    assert state == {"loads": 1, "attempts": [draft, draft], "saves": [draft]}, state
    second = workflow.submit(
        "방금 열어둔 같은 화면에서 인원만 7명으로 바꾸고 검토·저장해줘. "
        "회의실, 시간, 제목은 유지하고 성공한 저장은 한 번만 해."
    )
    completed(second)
    updated = {**draft, "people": "7"}
    assert state == {"loads": 1, "attempts": [draft, draft, updated], "saves": [draft, updated]}, state
    assert gui_calls(first) and gui_calls(second)
    request.node.live_passed = True
