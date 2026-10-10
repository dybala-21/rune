"""Task outcomes through web admission, storage, execution and completion."""

import base64
import csv
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from starlette.testclient import TestClient

from scripts.e2e_core import CHECKS, CODE, CSV
from tests.e2e.workflow_harness import Workflow, gui_calls


@pytest.fixture
def workflow(live_model, live_report, monkeypatch, tmp_path, request):
    from rune.agent import failover, loop
    from rune.api import conversation_wiring
    from rune.api.run_maintenance import RunMaintenance
    from rune.api.server import create_app
    from rune.config import get_config
    from rune.types import AgentConfig
    from rune.utils.paths import rune_data

    cached_models = rune_data() / "models"
    workspace, home = tmp_path / "work", tmp_path / "state"
    workspace.mkdir()
    home.mkdir()
    if cached_models.is_dir():
        (home / "data").mkdir()
        (home / "data" / "models").symlink_to(cached_models, target_is_directory=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.setenv("RUNE_WORKSPACE", str(workspace))
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(workspace))
    cfg = get_config()
    cfg.filesystem.allow_paths = [str(workspace)]
    cfg.browser.default_profile = "managed"
    cfg.proactive.enabled = False
    cfg.llm.reasoning_effort = None
    cfg.llm.reasoning_efforts = {}
    cfg.llm.route_simple_queries = False
    provider, model = live_model
    original_loop = loop.NativeAgentLoop
    decisions = []

    def create_loop(config=None):
        from rune.agent.goal_classifier import to_wire

        agent = original_loop(config or AgentConfig(
            provider=provider, model=model, max_iterations=16, timeout_seconds=180, _overridden=True))

        async def classified(value):
            decisions.append({"classification": json.loads(to_wire(value)), "tools": agent._select_tools(value)})

        agent.on("goal_classified", classified)
        return agent

    monkeypatch.setattr(loop, "NativeAgentLoop", create_loop)
    profiles = failover.build_profiles_from_config()[:1]
    monkeypatch.setattr(failover, "build_profiles_from_config", lambda: profiles)
    # Do not learn from synthetic tasks.
    monkeypatch.setattr(RunMaintenance, "enqueue", lambda *args, **kwargs: None)
    conversation_wiring._reset_for_tests()
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
        result = Workflow(client, workspace, provider, model)
        result.reporter = live_report
        result.decision_backend = cfg.llm.decision_routing.backend
        result.decisions = decisions
        yield result
        outcome = "passed" if getattr(request.node, "live_passed", False) else "failed"
        result.write_report(outcome)
    conversation_wiring._reset_for_tests()


def completed(run):
    assert run["status"] == "completed", run.get("error") or run.get("trust")
    assert run.get("success") is True, run.get("error") or run.get("trust")
    assert (run.get("trust") or {}).get("completionStatus") not in {"failed", "incomplete", "partial", "blocked"}, run.get("trust")


def test_simple_request_avoids_computer(workflow, request):
    run = workflow.submit("173 × 29 − 417의 값을 숫자만으로 답해줘.")
    completed(run)
    assert (run.get("answer") or run["text"]).strip().replace(",", "") == "4600"
    assert not gui_calls(run)
    request.node.live_passed = True


def test_csv_attachment_and_reload(workflow, request):
    attachment = {"name": "expenses.csv", "mimeType": "text/csv", "data": base64.b64encode(CSV.encode()).decode()}
    run = workflow.submit("첨부 expenses.csv의 department별 amount 합계를 summary.csv로 저장해. 열은 department,total이고 department 오름차순으로 정렬해. 원본은 유지하고 최종 답변에 부서별 합계를 보고해.", [attachment])
    completed(run)
    assert not gui_calls(run)
    output = workflow.workspace / "summary.csv"
    assert output.is_file()
    with output.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["department", "total"]
        rows = list(reader)
    assert [(r["department"], float(r["total"])) for r in rows] == [("engineering", 1000), ("sales", 1000), ("support", 450)]
    sources = list(workflow.workspace.glob("rune-attachment-*.csv"))
    assert len(sources) == 1 and sources[0].read_text() == CSV
    history = workflow.client.post("/api/v1/rpc", json={"method": "sessions.turns", "params": {"sessionId": workflow.session}}).json()["data"]
    refs = next(t["attachments"] for t in history["turns"] if t["role"] == "user")
    assert refs[0]["ref"] and "data" not in refs[0]
    follow = workflow.submit("이전에 보낸 원본 CSV를 다시 확인해서 전체 amount 합계를 숫자만으로 답해줘. 파일을 수정하지 마.", refs)
    completed(follow)
    assert not gui_calls(follow)
    assert (follow.get("answer") or follow["text"]).strip().replace(",", "") == "2450"
    assert len(list(workflow.workspace.glob("rune-attachment-*.csv"))) == 1
    assert sources[0].read_text() == CSV
    request.node.live_passed = True


def test_code_fix_reports_actual_tests(workflow, request):
    (workflow.workspace / "stats.py").write_text(CODE)
    (workflow.workspace / "test_stats.py").write_text(CHECKS)
    run = workflow.submit("stats.py의 average 버그를 고쳐줘. 빈 목록에는 ValueError를 발생시켜야 해. 기존 test_stats.py는 수정하지 말고, 수정 전후 테스트를 실행해서 통과·실패 수와 수정 전 실패한 테스트 이름을 실제 출력 기준으로 보고해.",
                          editable_files=("stats.py",), allow_new_tests=True)
    completed(run)
    assert not gui_calls(run)
    result = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=workflow.workspace, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert (workflow.workspace / "test_stats.py").read_text() == CHECKS
    verification = run["trust"]["verification"]
    assert verification["tests_passed_after_edit"] is True
    failures = {f"test_stats.AverageTests.test_{name}" for name in ("empty", "positive", "single")}
    assert any(
        check["write_sequence"] == 0 and check["report"]
        and (check["report"]["tests_run"] or 0) >= 4 and (check["report"]["failure_events"] or 0) >= 3
        and failures <= {
            case["identity"].replace("test_stats.py::", "test_stats.").replace("::", ".")
            for case in check["report"]["cases"] if case["status"] == "fail"
        }
        for check in verification["history"]
    ), verification
    assert any(check["sequence"] > verification["last_write"] and check["report"]
               and (check["report"]["tests_run"] or 0) >= 4 and check["report"]["failure_events"] == 0
               and check["status"] == "pass" for check in verification["history"]), verification
    request.node.live_passed = True


@pytest.fixture
def form_site():
    state = {"loads": 0, "saves": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            if self.path != "/form":
                self.send_error(404)
                return
            state["loads"] += 1
            body = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>회의 일정 연습</title>
            <h1>회의 일정 임시 저장</h1><label>도시 <input id="city" value="부산"></label>
            <label>인원 <input id="people" type="number" value="2"></label>
            <button onclick="fetch('/save',{method:'POST',body:JSON.stringify({city:document.querySelector('#city').value,people:document.querySelector('#people').value})}).then(()=>document.querySelector('#status').textContent='임시 저장됨')">임시 저장</button>
            <p id="status" role="status">아직 저장하지 않음</p></html>'''.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            state["saves"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/form", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_browser_followup_retains_page(workflow, form_site, request):
    url, state = form_site
    first = workflow.submit(f"브라우저에서 {url} 연습 화면을 열어 도시를 군산, 인원을 3으로 바꾸고 임시 저장을 한 번 눌러줘. 저장됐는지도 확인해.")
    completed(first)
    assert gui_calls(first)
    assert state["saves"] == [{"city": "군산", "people": "3"}]
    second = workflow.submit("방금 열어둔 화면에서 인원만 4로 변경하고 임시 저장을 한 번 더 눌러줘. 도시는 유지하고 같은 탭에서 이어서 해.")
    completed(second)
    assert gui_calls(second)
    assert state == {"loads": 1, "saves": [{"city": "군산", "people": "3"}, {"city": "군산", "people": "4"}]}
    request.node.live_passed = True
