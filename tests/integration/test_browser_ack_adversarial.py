"""Server acceptance completes a save only when the final action's own write carried the requested values."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pytest

from rune.agent.browser_requirements import request_key
from rune.agent.goal_classifier import ClassificationResult
from rune.capabilities.browser.core import BrowserNavigateParams, browser_navigate
from tests.integration.test_browser_foundation import screen as screen

GOAL = "Book it for 6 people as Release review and save once."
PLAN = {"complete": True, "items": [
    {"requirement": "6 people", "expected": "6", "kind": "value", "value_type": "number", "canonical": "6"},
    {"requirement": "Release review", "expected": "Release review", "kind": "value"},
    {"requirement": "save once", "kind": "result"},
]}
RESULT_ONLY = {"complete": True, "items": [{"requirement": "save once", "kind": "result"}]}
ROOM = "Book room Cedar as Release review and save once."


@pytest.fixture
def site():
    state = {"mode": "ok", "saves": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, status: int, body: bytes = b"", content_type: str = "text/html; charset=utf-8", location=None):
            self.send_response(status)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            mode = state["mode"]
            if self.path != "/form":
                return self._send(200, b"<h1>Saved</h1>")
            if mode == "navigate":
                return self._send(200, b'''<form method="post" action="/save">
                    <label>People <input id="people" name="people" type="number" value="2"></label>
                    <label>Title <input id="title" name="title" value=""></label>
                    <button id="save">Save</button></form>''')
            telemetry = {"label": "Release review x", "n": 16} if mode == "telemetry" else {"event": "click"}
            send = "" if mode in {"telemetry", "validate"} else (
                "fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify("
                "{people:document.querySelector('#people').value,title:document.querySelector('#title').value})})")
            validate = ("fetch('/validate',{method:'POST',body:JSON.stringify({title:this.value})})"
                        if mode == "validate" else "")
            page = f'''<label>People <input id="people" type="number" value="2"></label>
                <label>Title <input id="title" value="" onchange="{validate}"></label>
                <button id="save" onclick="fetch('/api/ui-state',{{method:'POST',body:'{json.dumps(telemetry).replace('"', '&quot;')}'}});{send}">Save</button>
                <p id="status" role="status">Saved</p>'''
            return self._send(200, page.encode())

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
            mode = state["mode"]
            if self.path != "/save":
                return self._send(200, b"{}", "application/json")
            state["saves"].append(raw)
            if mode == "navigate":
                assert parse_qs(raw)["title"] == ["Release review"]
                return self._send(303, location="/done")
            body = {"graphql_error": {"data": None, "errors": [{"message": "title is invalid"}]},
                    "success_false": {"success": False, "message": "title is invalid"}}.get(mode, {"id": 7})
            return self._send(200, json.dumps(body).encode(), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/form", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("mode,plan,expected", [
    ("ok", PLAN, "acknowledged"),
    ("navigate", PLAN, "acknowledged"),          # A full-page save is judged by what it sent, not the next page.
    ("graphql_error", PLAN, "inconclusive"),      # 200 with GraphQL errors did not save.
    ("success_false", PLAN, "inconclusive"),      # 200 with an explicit failure flag did not save.
    ("validate", PLAN, "inconclusive"),           # Only a validation call while typing was accepted.
    ("telemetry", PLAN, "inconclusive"),          # A body that merely contains the values is not the save.
    ("ok", RESULT_ONLY, "inconclusive"),          # Nothing requested to compare: needs a stored-record readback.
])
async def test_server_acceptance_needs_the_final_write_to_carry_the_values(screen, site, mode, plan, expected):
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = site
    state["mode"] = mode
    goal = GOAL if plan is PLAN else "save once"
    classification = ClassificationResult("browser", .99, 2, browser_requirements=plan,
                                          browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    def tool(name):
        return _build_typed_tool(cap_def=registry.get(name), reg=registry, cache=None, stall=None,
                                 opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended)).function

    assert (await browser_navigate(BrowserNavigateParams(url=url))).success
    await tool("browser_observe")()
    ids = {item.requirement: item.id for item in evidence.requirements.items}
    inputs = [] if plan is RESULT_ONLY else [
        {"type": "act", "params": {"action": "type", "selector": "#people", "value": "6"}},
        {"type": "act", "params": {"action": "type", "selector": "#title", "value": "Release review", "checks": [
            {"selector": "#people", "property": "value", "expected": "6", "requirement_ids": [ids["6 people"]]},
            {"selector": "#title", "property": "value", "expected": "Release review", "requirement_ids": [ids["Release review"]]},
        ]}},
    ]
    await tool("browser_batch")(actions=[*inputs, {"type": "act", "params": {"action": "click", "selector": "#save"}}])
    assert len(state["saves"]) == (0 if mode in {"validate", "telemetry"} else 1)
    acceptance = evidence.acceptance()
    assert acceptance["status"] == expected, acceptance
    assert "leaves" not in json.dumps(acceptance) and "Release review x" not in json.dumps(acceptance)


@pytest.fixture
def select_site():
    state = {"saves": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            # Labels are what people ask for; values are what the form sends.
            body = b'''<label>Room <select id="room"><option value="11">Birch</option><option value="12">Cedar</option>
                <option value="13">Pine</option></select></label><label>Title <input id="title" value=""></label>
                <button id="save" onclick="fetch('/save',{method:'POST',body:JSON.stringify({room:document.querySelector('#room').value,
                  title:document.querySelector('#title').value})})">Save</button>'''
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            state["saves"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/form", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("requested,chosen,expected", [("Cedar", "Cedar", "acknowledged"), ("Pine", "Cedar", "inconclusive")])
async def test_a_select_is_matched_by_the_label_whose_value_was_sent(screen, select_site, requested, chosen, expected):
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = select_site
    goal = ROOM.replace("Cedar", requested)
    plan = {"complete": True, "items": [
        {"requirement": f"room {requested}", "expected": requested, "kind": "value"},
        {"requirement": "Release review", "expected": "Release review", "kind": "value"},
        {"requirement": "save once", "kind": "result"},
    ]}
    classification = ClassificationResult("browser", .99, 2, browser_requirements=plan,
                                          browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    def tool(name):
        return _build_typed_tool(cap_def=registry.get(name), reg=registry, cache=None, stall=None,
                                 opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended)).function

    assert (await browser_navigate(BrowserNavigateParams(url=url))).success
    await tool("browser_observe")()
    await screen.select_option("#room", label=chosen)
    await tool("browser_batch")(actions=[
        {"type": "act", "params": {"action": "type", "selector": "#title", "value": "Release review"}},
        {"type": "act", "params": {"action": "click", "selector": "#save"}},
    ])
    assert state["saves"] == [{"room": "12", "title": "Release review"}]
    assert evidence.acceptance()["status"] == expected
