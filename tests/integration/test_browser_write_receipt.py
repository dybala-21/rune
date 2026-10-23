"""A save without a readable record completes on the site's own acceptance, never on page text."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rune.agent.browser_requirements import request_key
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.run_outcome import run_outcome
from rune.capabilities.browser.core import BrowserNavigateParams, browser_navigate
from tests.integration.test_browser_foundation import screen as screen

GOAL = "Book it for 6 people as Release review and save once."
PLAN = {"complete": True, "items": [
    {"requirement": "6 people", "expected": "6", "kind": "value", "value_type": "number", "canonical": "6"},
    {"requirement": "Release review", "expected": "Release review", "kind": "value"},
    {"requirement": "save once", "kind": "result"},
]}


@pytest.fixture
def site():
    state = {"status": 200, "posts": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            # The page claims success regardless of the response, so page text alone proves nothing.
            data = b'''<label>People <input id="people" type="number" value="2"></label>
                <label>Title <input id="title" value=""></label>
                <button id="save" onclick="fetch('/api/ui-state',{method:'POST',body:'{&quot;event&quot;:&quot;click&quot;}'});
                  if (window.blocked) { status.textContent='Saved'; return; }
                  fetch('/save',{method:'POST',body:JSON.stringify(window.partial ? {people:document.querySelector('#people').value}
                    : {people:document.querySelector('#people').value,title:document.querySelector('#title').value})}).then(()=>status.textContent='Saved')">Save</button>
                <button id="discard" onclick="fetch('/discard',{method:'POST'}).then(()=>status.textContent='Discarded')">Discard draft</button>
                <p id="status" role="status">Not saved</p>'''
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/api/ui-state":
                self.send_response(204 if state.get("telemetry_no_content") else 200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            state["posts"] += 1
            self.send_response(state["status"])
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


@pytest.mark.parametrize("status,edit_after,expected", [
    (200, False, "acknowledged"),
    (202, False, "inconclusive"),    # Accepted for later processing is not a completed save.
    (503, False, "inconclusive"),
    (500, False, "inconclusive"),
    (200, True, "inconclusive"),
    (200, "discard", "inconclusive"),  # A later, unrequested write cannot stand in for the save.
    (200, "blocked", "inconclusive"),  # Only telemetry was accepted; the save itself was never sent.
    (200, "partial", "inconclusive"),  # The accepted save omitted a requested field.
])
async def test_server_acceptance_completes_a_save_without_a_readable_record(screen, site, status, edit_after, expected):
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = site
    state["status"] = status
    classification = ClassificationResult("browser", .99, 2, browser_requirements=PLAN,
                                          browser_requirements_key=request_key((GOAL,)))
    evidence = TaskEvidence.for_request(GOAL, classification)
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    def tool(name):
        return _build_typed_tool(cap_def=registry.get(name), reg=registry, cache=None, stall=None,
                                 opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended)).function

    assert (await browser_navigate(BrowserNavigateParams(url=url))).success
    if edit_after == "blocked":
        await screen.evaluate("window.blocked = true")
    if edit_after == "partial":
        await screen.evaluate("window.partial = true")  # A buggy page that sends only one of the fields.
    await tool("browser_observe")()
    ids = {item.requirement: item.id for item in evidence.requirements.items}
    await tool("browser_batch")(actions=[
        {"type": "act", "params": {"action": "type", "selector": "#people", "value": "6"}},
        {"type": "act", "params": {"action": "type", "selector": "#title", "value": "Release review", "checks": [
            {"selector": "#people", "property": "value", "expected": "6", "requirement_ids": [ids["6 people"]]},
            {"selector": "#title", "property": "value", "expected": "Release review", "requirement_ids": [ids["Release review"]]},
        ]}},
        {"type": "act", "params": {"action": "click", "selector": "#save"}},
    ])
    if edit_after == "discard":
        await tool("browser_act")(action="click", selector="#discard")
    elif edit_after is True:
        await tool("browser_act")(action="type", selector="#title", value="Other")
    assert state["posts"] == {"discard": 2, "blocked": 0}.get(edit_after, 1)

    contract = resolve_intent_contract(classification, .99)
    # Only an accepted write may finish without a record readback.
    blocker = evidence.blocker(contract, 0)
    acceptance = evidence.acceptance()
    assert acceptance["status"] == expected, acceptance
    assert (blocker is None) is (expected == "acknowledged")
    outcome = run_outcome({"reason": "completed", "requirement_acceptance": acceptance})
    assert outcome.success is (expected == "acknowledged")
    assert outcome.verified is False
    if expected == "acknowledged":
        assert acceptance["writeReceipt"]["requests"][0]["status"] == 200
        assert outcome.status == "completed"
