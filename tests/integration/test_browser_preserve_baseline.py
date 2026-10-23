"""A follow-up save keeps the untouched field's baseline from the page it started on."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.capabilities.browser.core import BrowserNavigateParams, browser_navigate
from tests.integration.test_browser_foundation import screen as screen


@pytest.fixture
def site():
    state = {"saves": [{"city": "Gunsan", "people": "3"}], "fake": False, "posts": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            saved = state["saves"][-1]
            if self.path == "/form":
                # The status already reads "Saved" from the previous turn.
                body = f'''<label>City <input id="city" value="{saved["city"]}"></label>
                    <label>People <input id="people" type="number" value="{saved["people"]}"></label>
                    <button onclick="fetch('/save',{{method:'POST',body:JSON.stringify({{city:city.value,people:people.value}})}})
                    .then(()=>status.textContent='Saved')">Save draft</button>
                    <p id="status" role="status">Saved</p><a href="/stored">Saved record</a>'''
            elif self.path == "/stored":
                body = (f'<label>City <input id="city" readonly value="{saved["city"]}"></label>'
                        f'<label>People <input id="people" readonly value="{saved["people"]}"></label>')
            else:
                self.send_error(404)
                return
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["posts"] += 1
            if not state["fake"]:
                state["saves"].append(body)
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("fake,status", [(False, "pass"), (True, "fail")])
async def test_followup_save_confirms_preserved_field_from_the_initial_view(screen, site, fake, status):
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = site
    state["fake"] = fake
    assert (await browser_navigate(BrowserNavigateParams(url=url + "/form"))).success
    goal = "Change only people to 4 and save the draft once more. Keep the city."
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    def tool(name):
        return _build_typed_tool(cap_def=registry.get(name), reg=registry, cache=None, stall=None,
                                 opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended)).function

    # The follow-up seed observes the open page before the model plans.
    observed = await tool("browser_observe")()
    refs = {line.split("]")[1].strip(): line[1:].split("]")[0]
            for line in str(observed).splitlines() if line.startswith("[")}
    city, people, save = refs['textbox "City"'], refs['spinbutton "People"'], refs['button "Save draft"']

    out = await tool("browser_batch")(
        taskRequirements={"items": [
            {"requirement": "Change only people to 4", "kind": "value", "expected": "4", "value_type": "number"},
            {"requirement": "Keep the city", "kind": "preserve", "expected": "Gunsan"},
            {"requirement": "save the draft once more", "kind": "result", "expected": None},
        ]},
        actions=[
            {"type": "act", "params": {"action": "type", "selector": people, "value": "4", "checks": [
                {"property": "value", "expected": "4", "selector": people, "requirement": "Change only people to 4", "value_type": "number"},
                {"property": "value", "expected": "Gunsan", "selector": city, "requirement": "Keep the city"},
            ]}},
            {"type": "act", "params": {"action": "click", "selector": save}},
            {"type": "observe", "params": {"readbackUrl": url + "/stored", "resultRequirement": "save the draft once more"}},
        ],
    )
    snapshot = json.loads(str(out).split("Result readback: ")[-1].split("\n")[0])["snapshot"]
    ids = {item.requirement: item.id for item in evidence.requirements.items}
    await tool("browser_observe")(readbackSnapshot=snapshot, resultRequirement="save the draft once more", checks=[
        {"property": "value", "expected": "4", "selector": 'input[id="people"]',
         "requirement_ids": [ids["Change only people to 4"]], "value_type": "number"},
        {"property": "value", "expected": "Gunsan", "selector": 'input[id="city"]',
         "requirement_ids": [ids["Keep the city"]]},
    ])

    assert state["posts"] == 1
    assert state["saves"][-1] == {"city": "Gunsan", "people": "3" if fake else "4"}
    assert evidence.acceptance()["status"] == status, evidence.acceptance()
