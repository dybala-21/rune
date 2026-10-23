"""Check stored values independently without resubmitting the open form."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.run_outcome import run_outcome
from rune.capabilities.browser.capabilities import BrowserObserveParams, browser_observe
from rune.capabilities.browser.core import BrowserNavigateParams, browser_navigate
from tests.integration.test_browser_foundation import screen as screen
from tests.integration.test_browser_receipts import act, check, start


def test_discovered_selectors_preserve_unicode_and_exclude_ambiguous_or_secret_fields():
    from lxml import html

    from rune.capabilities.browser.readback import document_fields

    document = html.fromstring('''<form><input id="인원" value="3"><input id="duplicate"><input id="duplicate">
        <input id="secret" type="password"><input id="token" type="hidden"></form>''')
    fields = document_fields(document)
    assert fields == [{"selector": 'input[id="인원"]', "property": "value"}]
    assert document.cssselect(fields[0]["selector"])[0].get("value") == "3"


@pytest.fixture
def site():
    state = {'saved': '1', 'posts': 0, 'reads': 0, 'mode': 'saved', 'scripts': True}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            if self.path == '/form':
                body = '''<label>People<input id="people" value="1"></label>
                    <button id="save" onclick="fetch('/save',{method:'POST',body:people.value})
                    .then(()=>receipt.textContent='Saved')">Save</button><output id="receipt">Pending</output>
                    <a href="/stored">Saved record</a>'''
            elif self.path in {'/stored', '/unlinked'}:
                state['reads'] += 1
                if state['mode'] == 'redirect':
                    self.send_response(302)
                    self.send_header('Location', '/save')
                    self.end_headers()
                    return
                script = '<script>fetch("/save",{method:"POST"})</script>' if state['scripts'] else ''
                if state['mode'] == 'spoof':
                    script = '<script>stored.value="3"</script>'
                body = ('<div id="app"></div>' if state['mode'] == 'spa' else
                        f'<label>People<input id="stored" value="{state["saved"]}"></label><p id="city">군산</p>{script}')
            else:
                self.send_error(404)
                return
            data = body.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            if self.path == '/stored' and state.get('cache'):
                self.send_header('Cache-Control', 'public, max-age=3600')
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            value = self.rfile.read(int(self.headers.get('Content-Length', '0'))).decode()
            state['posts'] += 1
            if state['mode'] == 'saved':
                state['saved'] = value
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('mode,status', [('saved', 'pass'), ('fake', 'fail'), ('spa', 'unavailable'), ('redirect', 'unavailable')])
async def test_result_readback_checks_server_fields_without_a_second_write(screen, site, mode, status):
    url, state = site
    state['mode'] = mode
    assert (await browser_navigate(BrowserNavigateParams(url=url + '/form'))).success
    evidence = await start('Set people to 3, save once and confirm the save', [
        {'requirement': 'people to 3', 'expected': '3'},
        {'requirement': 'save once', 'kind': 'result'},
        {'requirement': 'confirm the save', 'kind': 'result', 'confirms': 'save once'},
    ])
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save', checks=[
        check('#people', 'people to 3', '3', 'value'), check('#receipt', 'confirm the save'),
    ])
    # A 2xx is only acceptance; the readback decides the verdict.
    assert evidence.acceptance()['status'] == 'acknowledged'
    params = {'readbackUrl': url + '/stored', 'checks': [check('#stored', 'people to 3', '3', 'value')]}
    evidence.prepare_tool('browser_observe', params)
    assert len(params['checks']) == 1
    result = await browser_observe(BrowserObserveParams(**params))
    evidence.observe('browser_observe', result)
    assert result.metadata['independent_readback']['status'] == status
    assert evidence.acceptance()['status'] == status
    verdict = run_outcome({'reason': 'completed' if status == 'pass' else evidence.stop_reason(),
                          'requirement_acceptance': evidence.acceptance()})
    assert verdict.success is (status == 'pass')
    assert evidence.blocker(resolve_intent_contract(ClassificationResult('browser', .99, 2), .99), 0) is None
    assert state['posts'] == 1 and state['reads'] == 1
    assert state['saved'] == ('3' if mode == 'saved' else '1')
    assert screen.url == url + '/form' and await screen.locator('#people').input_value() == '3'
    if status != 'pass':
        evidence.observe('browser_observe', await browser_observe(BrowserObserveParams()))
        assert evidence.acceptance()['status'] == status


@pytest.mark.parametrize("mode,status", [("saved", "pass"), ("fake", "fail")])
@pytest.mark.parametrize("plan_source", ["routed", "executor"])
async def test_guessed_message_does_not_retry_or_replace_stored_value_checks(screen, site, mode, status, plan_source):
    from rune.agent.task_evidence import TaskEvidence

    url, state = site
    state["mode"] = mode
    await browser_navigate(BrowserNavigateParams(url=url + "/form"))
    evidence = await start("Set people to 3 and save", [
        {"requirement": "people to 3", "expected": "3"}, {"requirement": "save", "kind": "result"},
    ])
    if plan_source == "executor":
        evidence = TaskEvidence.for_request("Set people to 3 and save", ClassificationResult("browser", .99, 2))
        await evidence.prepare_requirements()
        evidence.prepare_tool("browser_observe", {"taskRequirements": {"items": [
            {"requirement": "people to 3", "kind": "value", "expected": "3"},
            {"requirement": "save", "kind": "result", "expected": "The meeting was saved successfully."},
        ]}})
    await act(evidence, action="type", selector="#people", value="3")
    result = await act(evidence, action="click", selector="#save", checks=[
        check("#people", "people to 3", "3", "value"),
        check("#receipt", "save", "The meeting was saved successfully."),
    ])
    receipt = next(row for row in result.metadata["result_checks"]["checks"] if row["kind"] == "result")
    assert receipt["observed"] and receipt["actual"] == "Saved"
    assert result.metadata["result_checks"]["passed"]
    assert evidence.acceptance()["status"] != "pass"
    params = {"readbackUrl": url + "/stored", "checks": [
        check("#stored", "people to 3", "3", "value"),
        check("#receipt", "save", "Saved"),
    ]}
    evidence.prepare_tool("browser_observe", params)
    assert len(params["checks"]) == 1
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    assert evidence.acceptance()["status"] == status
    assert state["posts"] == state["reads"] == 1


async def test_explicit_message_requirement_cannot_be_replaced_by_observation(screen, site):
    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + "/form"))
    evidence = await start("Set people to 3, save and show Done", [
        {"requirement": "people to 3", "expected": "3"},
        {"requirement": "save and show Done", "kind": "result", "expected": "Done"},
    ])
    await act(evidence, action="type", selector="#people", value="3")
    result = await act(evidence, action="click", selector="#save", checks=[
        check("#receipt", "save and show Done", "Done"),
    ])
    assert not result.metadata["result_checks"]["passed"]
    params = {"readbackUrl": url + "/stored", "checks": [check("#stored", "people to 3", "3", "value")]}
    evidence.prepare_tool("browser_observe", params)
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    assert state["saved"] == "3" and state["posts"] == 1
    assert evidence.acceptance()["status"] != "pass"


async def test_failed_saved_record_blocks_repeated_input_after_navigation(screen, site):
    from rune.agent.tool_errors import ToolPreparationError
    from rune.types import CapabilityResult

    url, state = site
    state["mode"] = "fake"
    evidence = await start("Set people to 3 and save", [
        {"requirement": "people to 3", "expected": "3"}, {"requirement": "save", "kind": "result"},
    ])
    evidence.observe("browser_navigate", await browser_navigate(BrowserNavigateParams(url=url + "/form")))
    await act(evidence, action="type", selector="#people", value="3")
    await act(evidence, action="click", selector="#save")
    params = {"readbackUrl": url + "/stored", "checks": [check("#stored", "people to 3", "3", "value")]}
    evidence.prepare_tool("browser_observe", params)
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    evidence.observe("browser_navigate", await browser_navigate(BrowserNavigateParams(url=url + "/form")))
    for action, selector in (("type", "#people"), ("click", "#save")):
        with pytest.raises(ToolPreparationError, match="result_unconfirmed") as error:
            evidence.prepare_tool("browser_act", {"action": action, "selector": selector, "value": "3"})
        evidence.observe("browser_act", CapabilityResult(success=False, error=str(error.value),
            metadata={"action_status": "not_executed", "validation_error": error.value.details}))
    assert state["posts"] == 1 and evidence.acceptance()["status"] == "fail"
    state["saved"] = "3"
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    assert not evidence.readback_write_block and evidence.acceptance()["status"] == "pass"
    assert state["posts"] == 1


async def test_first_input_batch_binds_requirements_through_the_tool_adapter(screen, site):
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = site
    evidence = TaskEvidence.for_request("Set people to 3 and save", ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    evidence.observe("browser_navigate", await browser_navigate(BrowserNavigateParams(url=url + "/form")))
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    tool = _build_typed_tool(cap_def=registry.get("browser_batch"), reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended))
    await tool.function(taskRequirements={"items": [
        {"requirement": "people to 3", "expected": "3", "kind": "value"},
        {"requirement": "save", "expected": None, "kind": "result"},
    ]}, actions=[{"action": "type", "selector": "#people", "value": "3"}, {"action": "click", "selector": "#save"}])
    assert state["saved"] == "3" and state["posts"] == 1
    assert evidence.browser_actions == 2 and evidence.requirements.complete
    # The page links to its record, so finishing asks for that readback once.
    assert evidence.acceptance()["status"] == "acknowledged"
    contract = resolve_intent_contract(ClassificationResult("browser", .99, 2), .99)
    assert "readbackUrl" in evidence.blocker(contract, 0)
    assert evidence.blocker(contract, 0) is None


@pytest.mark.parametrize("mode,status", [("saved", "pass"), ("fake", "fail")])
async def test_recovered_plan_keeps_one_write_and_checks_the_stored_record(screen, site, mode, status):
    from copy import deepcopy
    from unittest.mock import AsyncMock

    from rune.agent.browser_plan import configure_tools
    from rune.agent.browser_result import recovery_tools
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_adapter import ToolAdapterOptions, ToolWrapper, build_tool_set
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = site
    state["mode"] = mode
    goal = "Set people to 3 and save"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    async def ended(name, result):
        evidence.observe(name, result)
    tools = build_tool_set(ToolAdapterOptions(enable_guardian=False, prepare_tool=evidence.prepare_tool,
                                            on_tool_end=ended), registry=registry)
    configure_tools(evidence, tools)
    tools["file_read"] = ToolWrapper(name="file_read", description="Read a file", function=AsyncMock())
    agent = LiteLLMAgent("vertex_ai/gemini-2.5-flash", tools=list(tools.values()), compact_tools=True,
                        deferred_tools={"file_read"})
    plan = {"items": [{"requirement": "people to 3", "kind": "value", "expected": "3"},
                      {"requirement": "save", "kind": "result", "expected": None}]}
    async with agent.run_stream(goal, tool_recovery=lambda: recovery_tools(evidence)) as stream:
        await stream._execute_tool("browser_navigate", {"url": url + "/form"})
        await stream._execute_tool("browser_observe", {"taskRequirements": deepcopy(plan)})
        fixed = evidence.requirements
        await stream._execute_tool("BrowserBatchTaskrequirementsItems", {})
        await stream._execute_tool("tool_search", {"query": "browser_batch"})
        actions = [{"type": "act", "params": {"action": "type", "selector": "#people", "value": "3"}},
                   {"type": "act", "params": {"action": "click", "selector": "#save", "readback": {
                       "url": url + "/form", "checks": [check("#receipt", "save", "Saved successfully")],
                   }}}]
        invalid = deepcopy(plan)
        invalid["items"][0]["requirement_ids"] = [fixed.items[-1].id]
        rejected = await stream._execute_tool("browser_batch", {"actions": deepcopy(actions), "taskRequirements": invalid})
        assert rejected.code == "fixed_requirements" and state["posts"] == 0
        assert stream._recovery_tools() is None
        observed = await stream._execute_tool("browser_observe", {})
        assert not getattr(observed, "code", None)
        for item, bound in zip(plan["items"], fixed.items, strict=True):
            item["requirement_ids"] = [bound.id]
        plan["items"][-1]["expected"] = "Saved successfully"
        result = await stream._execute_tool("browser_batch", {"actions": actions, "taskRequirements": plan})
        assert not getattr(result, "code", None)
        assert evidence.requirements is fixed and state["posts"] == 1 and state["reads"] == 0
        assert evidence.acceptance()["status"] != "pass"
        await stream._execute_tool("browser_observe", {"readbackUrl": url + "/stored", "checks": [
            check("#stored", "people to 3", "3", "value"),
        ]})
    assert evidence.independent_readback["status"] == status
    assert evidence.acceptance()["status"] == status
    assert state["posts"] == state["reads"] == 1
    assert state["saved"] == ("3" if mode == "saved" else "1")
    assert screen.url == url + "/form"


@pytest.mark.parametrize('target', ['http://example.com/stored', '/unknown', 'file:///tmp/secret'])
async def test_readback_rejects_unobserved_or_cross_origin_urls(screen, site, target):
    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    result = await browser_observe(BrowserObserveParams(readbackUrl=target, checks=[
        check('#stored', 'people to 3', '3', 'value')]))
    assert result.metadata['independent_readback']['status'] == 'unavailable'
    assert state['posts'] == state['reads'] == 0


@pytest.mark.parametrize('tool,metadata', [
    ('browser_act', {'action_status': 'dispatched', 'action': 'type'}),
    ('browser_navigate', {'skipped_navigation': False}),
])
async def test_saved_record_check_is_invalidated_by_later_changes(tool, metadata):
    from rune.types import CapabilityResult

    evidence = await start('save', [{'requirement': 'save', 'kind': 'result'}])
    evidence.observe('browser_observe', CapabilityResult(success=True, metadata={
        'independent_readback': {'status': 'pass', 'scope': 'http_document'}}))
    evidence.observe(tool, CapabilityResult(success=True, metadata=metadata))
    assert evidence.independent_readback['status'] == 'inconclusive'
    assert evidence.acceptance()['status'] != 'pass'


async def test_readback_uses_response_encoding(screen, site):
    url, _ = site
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    result = await browser_observe(BrowserObserveParams(readbackUrl=url + '/stored', checks=[check('#city', 'city', '군산')]))
    assert result.metadata['independent_readback']['status'] == 'pass'


async def test_unrelated_readback_cannot_hide_a_known_mismatch(screen, site):
    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    evidence = await start('Set people to 3 and city to 군산', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'city to 군산', 'expected': '군산'},
    ])
    for mode, field in [('saved', check('#stored', 'people to 3', '3', 'value')),
                        ('spa', check('#stored', 'people to 3', '3', 'value')),
                        ('saved', check('#city', 'city to 군산', '군산'))]:
        state['mode'] = mode
        params = {'readbackUrl': url + '/stored', 'checks': [field]}
        evidence.prepare_tool('browser_observe', params)
        evidence.observe('browser_observe', await browser_observe(BrowserObserveParams(**params)))
        assert evidence.acceptance()['status'] == 'fail'
    state['saved'] = '3'
    params = {'readbackUrl': url + '/stored', 'checks': [check('#stored', 'people to 3', '3', 'value')]}
    evidence.prepare_tool('browser_observe', params)
    evidence.observe('browser_observe', await browser_observe(BrowserObserveParams(**params)))
    assert evidence.independent_readback['status'] == 'pass'
    assert state['posts'] == 0


@pytest.mark.parametrize('mode,status', [('saved', 'pass'), ('fake', 'fail'), ('spa', 'unavailable')])
async def test_save_reads_the_record_in_the_same_tool_call(screen, site, mode, status):
    url, state = site
    state['mode'] = mode
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    evidence = await start('Set people to 3, save once and confirm the save', [
        {'requirement': 'people to 3', 'expected': '3'},
        {'requirement': 'save once', 'kind': 'result'},
        {'requirement': 'confirm the save', 'kind': 'result', 'confirms': 'save once'},
    ])
    await act(evidence, action='type', selector='#people', value='3')
    result = await act(evidence, action='click', selector='#save',
        checks=[check('#people', 'people to 3', '3', 'value'), check('#receipt', 'confirm the save', 'Invented wording')],
        readback={'url': url + '/stored', 'checks': [check('#stored', 'people to 3', '3', 'value')]})
    assert result.metadata['independent_readback']['status'] == status
    assert evidence.acceptance()['status'] == status
    assert all(row['requirement'] != 'save once' for row in result.metadata['result_checks']['checks'])
    assert bool(result.metadata['remaining_browser_requirements']) is (status != 'pass')
    assert evidence.blocker(resolve_intent_contract(ClassificationResult('browser', .99, 2), .99), 0) is None
    assert state['posts'] == state['reads'] == 1
    assert screen.url == url + '/form' and await screen.locator('#people').input_value() == '3'


async def test_replayed_save_reads_the_record_without_dispatching_again(screen, site):
    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    evidence = await start('Set people to 3 and save', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save', 'kind': 'result'}])
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save')
    result = await act(evidence, action='click', selector='#save',
        readback={'url': url + '/stored', 'checks': [check('#stored', 'people to 3', '3', 'value')]})
    assert result.metadata['input_replayed']
    assert evidence.browser_actions == 2
    assert evidence.acceptance()['status'] == 'pass'
    assert state['posts'] == state['reads'] == 1


async def test_page_message_alone_cannot_finish_a_save(screen, site):
    url, state = site
    state['mode'] = 'fake'
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    evidence = await start('Set people to 3 and save', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save', 'kind': 'result'}])
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save', checks=[
        check('#people', 'people to 3', '3', 'value'), check('#receipt', 'save')])
    acceptance = evidence.acceptance()
    assert acceptance['pageStatus'] == 'pass'
    # Without a readback, a 200 that stored nothing looks like a save, so the run is never verified.
    assert acceptance['status'] == 'acknowledged'
    assert acceptance['pendingResults']
    verdict = run_outcome({'reason': 'completed', 'requirement_acceptance': acceptance})
    assert verdict.success and not verdict.verified and verdict.status == 'completed'
    assert state['saved'] == '1' and state['posts'] == 1


async def test_partial_record_fields_do_not_complete_the_whole_save(screen, site):
    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    evidence = await start('Set people to 3 and city to 군산, then save', [
        {'requirement': 'people to 3', 'expected': '3'},
        {'requirement': 'city to 군산', 'expected': '군산'}, {'requirement': 'save', 'kind': 'result'}])
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save',
              readback={'url': url + '/stored', 'checks': [check('#stored', 'people to 3', '3', 'value')]})
    assert evidence.independent_readback['status'] == 'pass'
    assert evidence.acceptance()['status'] == 'inconclusive'
    assert state['posts'] == state['reads'] == 1
    contract = resolve_intent_contract(ClassificationResult("browser", .99, 2), .99)
    assert "city to 군산" in evidence.blocker(contract, 0)
    params = {"readbackUrl": url + "/stored", "readbackSnapshot": evidence.independent_readback["snapshot"],
              "checks": [check("#stored", "people to 3", "3", "value"), check("#city", "city to 군산", "군산", "text")]}
    evidence.prepare_tool("browser_observe", params)
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    assert evidence.acceptance()["status"] == "pass"
    assert state["posts"] == state["reads"] == 1


@pytest.mark.parametrize('mode,stopped', [('saved', False), ('fake', False), ('fake', True)])
@pytest.mark.parametrize('plan_source', ['routed', 'executor', 'corrected'])
@pytest.mark.parametrize('readback_path', ['inline', 'deferred'])
async def test_production_loop_reads_saved_values_with_bounded_recovery(screen, site, monkeypatch, tmp_path, mode, plan_source, stopped, readback_path):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from rune.agent.browser_requirements import request_key
    from rune.agent.loop import NativeAgentLoop
    from rune.agent.tool_adapter import build_tool_set
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    url, state = site
    state['mode'] = mode
    goal = 'Set people to 3 and save once'
    plan = {'complete': True, 'items': [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save once', 'kind': 'result'}]}
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    monkeypatch.setattr('rune.agent.loop.build_tool_set', lambda opts: build_tool_set(opts, registry))
    auxiliary = AsyncMock(side_effect=AssertionError('No auxiliary model requests'))
    monkeypatch.setattr('rune.llm.client.get_llm_client', auxiliary)
    turns, called = [], []

    class Stream:
        tool_budget_exhausted = False
        _task_blocked = ''
        def __init__(self, tools, recovery):
            self.tools = tools
            self.recovery = recovery
        async def stream_text(self, **kwargs):
            calls = [('browser_navigate', {'url': url + '/form', **({'taskRequirements': plan} if plan_source == 'executor' else {})}),
                     ('browser_act', {'action': 'type', 'selector': '#people', 'value': '3'}),
                     ('browser_act', {'action': 'click', 'selector': '#save',
                       'checks': [check('#receipt', 'save once', 'Guessed success phrase')],
                       'readback': {'url': url + '/stored', 'checks': [check('#stored', 'people to 3', '3', 'value')]}})]
            if len(turns) > 1:
                assert self.recovery() == {'browser_observe', 'browser_navigate', 'ask_user', 'task_blocked'}
                calls = [('browser_observe', {'readbackUrl': url + '/stored',
                                             'checks': [check('#stored', 'people to 3', '3', 'value')]})]
            for name, args in calls:
                if stopped or readback_path == 'deferred':
                    args.pop('readback', None)
                called.append(name)
                output = await self.tools[name].function(**args)
                assert '[ERROR]' not in str(output), str(output)
                if name == 'browser_navigate' and plan_source == 'corrected':
                    rejected = await self.tools['browser_batch'].function(
                        taskRequirements={'items': [{'requirement': 'save once', 'kind': 'result',
                                                    'expected': 'Invented wording', 'selector': '#receipt'}]},
                        actions=[{'type': 'act', 'params': {'action': 'type', 'selector': '#people', 'value': '3'}}],
                    )
                    assert 'invalid_requirements' in str(rejected)
                    assert self.recovery() == {'browser_observe', 'ask_user', 'task_blocked'}
                    assert state['posts'] == 0 and await screen.locator('#people').input_value() == '1'
                    corrected = await self.tools['browser_observe'].function(taskRequirements=plan)
                    assert '[ERROR]' not in str(corrected)
                    assert self.recovery() is None
                    called.extend(['browser_batch', 'browser_observe'])
            yield 'Saved successfully.'
        async def get_output(self): return 'Saved successfully.'
        def usage(self): return SimpleNamespace(input_tokens=10, output_tokens=10)
        def get_failure_state(self): return {}, set()
        def all_messages(self): return []

    class Agent:
        def __init__(self, *args, tools, **kwargs):
            self.tools = {tool.name: tool for tool in tools}
        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            turns.append(kwargs)
            if stopped and len(turns) > 1:
                from rune.agent.request_budget import BudgetExceeded
                raise BudgetExceeded('No requests remain')
            yield Stream(self.tools, kwargs['tool_recovery'])

    monkeypatch.setattr('rune.agent.loop.LiteLLMAgent', Agent)
    classification = ClassificationResult('browser', .99, 2, browser_requirements=plan if plan_source == 'routed' else None,
                                          browser_requirements_key=request_key((goal,)))
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    trace = await loop._execute_loop(goal, '', ['browser_navigate', 'browser_observe', 'browser_act', 'browser_batch'], 2, classification)
    assert run_outcome(trace).success is (mode == 'saved')
    assert len(turns) == (2 if stopped or readback_path == 'deferred' else 1)
    assert called == ['browser_navigate'] + (['browser_batch', 'browser_observe'] if plan_source == 'corrected' else []) + ['browser_act', 'browser_act'] + (
        ['browser_observe'] if not stopped and readback_path == 'deferred' else [])
    assert state['posts'] == 1 and state['reads'] == (0 if stopped else 1)
    assert screen.url == url + '/form'
    if mode == 'fake':
        assert 'Saved successfully.' not in loop._last_answer_text
        if stopped:
            assert trace.reason == 'request_budget_exhausted'
            assert 'unconfirmed' in loop._last_answer_text
        else:
            assert '1' in loop._last_answer_text and '3' in loop._last_answer_text
        from rune.api.conversation_wiring import record_assistant_turn
        from rune.conversation.manager import ConversationManager
        from rune.conversation.store import ConversationStore
        store = ConversationStore(tmp_path / 'history.db')
        manager = ConversationManager(store)
        conv = manager.start_conversation('test')
        await record_assistant_turn(manager, conv.id, loop, 'Saved successfully.', embed=False, require_save=True)
        manager._active.clear()
        manager._active[conv.id] = await store.load(conv.id)
        history = manager._build_messages_within_budget(conv.id, 2000)
        assert history[-1]['content'] == loop._last_answer_text
    auxiliary.assert_not_called()


@pytest.mark.parametrize('navigation', ['url', 'link'])
@pytest.mark.parametrize('mode,status', [('saved', 'pass'), ('fake', 'fail'), ('spoof', 'fail')])
async def test_record_navigation_reuses_response_with_only_observed_refs(screen, site, mode, status, navigation):
    from rune.capabilities.browser.helpers import get_element_store

    url, state = site
    state.update(mode=mode, scripts=False)
    evidence = await start('Set people to 3 and save once', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save once', 'kind': 'result'}])
    evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/form')))
    def ref(name):
        return next(item.ref for item in get_element_store().all.values() if item.name == name)
    await act(evidence, action='type', selector=ref('People'), value='3', checks=[check(ref('People'), 'people to 3', '3', 'value')])
    await act(evidence, action='click', selector=ref('Save'))
    if navigation == 'link':
        await act(evidence, action='click', selector=ref('Saved record'))
    else:
        evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/stored')))
    params = {'checks': [check(ref('People'), 'people to 3', '3', 'value')]}
    evidence.prepare_tool('browser_observe', params)
    assert len(params['checks']) == 1, 'Old form checks must not follow the record navigation'
    result = await browser_observe(BrowserObserveParams(**params))
    evidence.observe('browser_observe', result)
    report = result.metadata['independent_readback']
    assert report['source'] == 'navigation_response' and report['status'] == status
    assert evidence.acceptance()['status'] == status
    assert state['posts'] == state['reads'] == 1
    assert report['checks'][0]['actual'] == ('3' if mode == 'saved' else '1')
    if mode == 'spoof':
        assert await screen.locator('#stored').input_value() == '3'


async def test_readback_accepts_an_observed_link_ref_and_rejects_it_after_reload(screen, site):
    from rune.capabilities.browser.helpers import get_element_store

    url, state = site
    nav = await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    link = next(item for item in get_element_store().all.values() if item.name == 'Saved record')
    assert url + '/stored' in nav.output
    checks = [check('#stored', 'people to 1', '1', 'value')]
    first = await browser_observe(BrowserObserveParams(readbackUrl=link.ref, checks=checks))
    assert first.metadata['independent_readback']['status'] == 'pass'
    reloaded = await browser_navigate(BrowserNavigateParams(url=url + '/form', reload=True))
    assert nav.metadata['document_id'] != reloaded.metadata['document_id']
    stale = await browser_observe(BrowserObserveParams(readbackUrl=link.ref, checks=checks))
    assert stale.metadata['independent_readback']['status'] == 'unavailable'
    assert state['reads'] == 1 and state['posts'] == 0


async def test_current_form_values_do_not_count_as_a_fetched_record(screen, site):
    url, state = site
    evidence = await start('Set people to 3 and save', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save', 'kind': 'result'}])
    evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/form')))
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save')
    params = {'checks': [check('#people', 'people to 3', '3', 'value')]}
    evidence.prepare_tool('browser_observe', params)
    result = await browser_observe(BrowserObserveParams(**params))
    assert 'independent_readback' not in result.metadata
    evidence.observe('browser_observe', result)
    assert evidence.acceptance()['status'] != 'pass'
    assert state['reads'] == 0 and state['posts'] == 1


async def test_unobserved_navigation_cannot_supply_saved_record_evidence(screen, site):
    url, state = site
    state['scripts'] = False
    evidence = await start('Set people to 3 and save', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save', 'kind': 'result'}])
    evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/form')))
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save')
    evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/unlinked')))
    params = {'checks': [check('#stored', 'people to 3', '3', 'value')]}
    evidence.prepare_tool('browser_observe', params)
    result = await browser_observe(BrowserObserveParams(**params))
    assert 'independent_readback' not in result.metadata
    evidence.observe('browser_observe', result)
    assert evidence.acceptance()['status'] != 'pass'
    assert state['posts'] == state['reads'] == 1


async def test_cached_record_cannot_confirm_a_later_fake_save(screen, site):
    url, state = site
    state.update(saved='3', scripts=False, cache=True)
    await browser_navigate(BrowserNavigateParams(url=url + '/form'))
    await browser_navigate(BrowserNavigateParams(url=url + '/stored'))
    state.update(saved='1', mode='fake')
    evidence = await start('Set people to 3 and save', [
        {'requirement': 'people to 3', 'expected': '3'}, {'requirement': 'save', 'kind': 'result'}])
    evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/form')))
    await act(evidence, action='type', selector='#people', value='3')
    await act(evidence, action='click', selector='#save')
    evidence.observe('browser_navigate', await browser_navigate(BrowserNavigateParams(url=url + '/stored')))
    assert await screen.locator('#stored').input_value() == '3'
    assert state['reads'] == state['posts'] == 1
    params = {'checks': [check('#stored', 'people to 3', '3', 'value')]}
    evidence.prepare_tool('browser_observe', params)
    result = await browser_observe(BrowserObserveParams(**params))
    assert 'independent_readback' not in result.metadata
    evidence.observe('browser_observe', result)
    assert evidence.acceptance()['status'] != 'pass'
    params['readbackUrl'] = url + '/stored'
    evidence.prepare_tool('browser_observe', params)
    evidence.observe('browser_observe', await browser_observe(BrowserObserveParams(**params)))
    assert evidence.acceptance()['status'] == 'fail'
    assert state['reads'] == 2 and state['posts'] == 1


@pytest.mark.parametrize("mode,expected", [("saved", "pass"), ("fake", "fail")])
@pytest.mark.parametrize("selector,prop", [("input[type=number]", "value"), ("body", "text")])
@pytest.mark.parametrize("repeat_url", [True, False])
async def test_bad_selector_can_be_corrected_from_the_same_response_once(screen, site, mode, expected, selector, prop, repeat_url):
    url, state = site
    state["mode"] = mode
    evidence = await start("Set people to 3 and save", [
        {"requirement": "people to 3", "expected": "3"}, {"requirement": "save", "kind": "result"}])
    evidence.observe("browser_navigate", await browser_navigate(BrowserNavigateParams(url=url + "/form")))
    await act(evidence, action="type", selector="#people", value="3")
    await act(evidence, action="click", selector="#save")
    params = {"readbackUrl": url + "/stored", "checks": [check(selector, "people to 3", "3", prop)]}
    evidence.prepare_tool("browser_observe", params)
    result = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", result)
    report = result.metadata["independent_readback"]
    assert report["status"] == "unavailable" and report["snapshot"]
    assert report["fields"] == [{"selector": 'input[id="stored"]', "property": "value"}]
    contract = resolve_intent_contract(ClassificationResult("browser", .99, 2), .99)
    assert "readbackSnapshot" in evidence.blocker(contract, 0)
    evidence.recovery_requested = True
    assert evidence.blocker(contract, 0) is None
    params.update(readbackSnapshot=report["snapshot"],
                  checks=[check(report["fields"][0]["selector"], "people to 3", "3", "value")])
    if not repeat_url:
        params.pop("readbackUrl")
    evidence.prepare_tool("browser_observe", params)
    result = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", result)
    assert result.metadata["independent_readback"]["source"] == "readback_snapshot"
    assert evidence.acceptance()["status"] == expected
    assert state["reads"] == state["posts"] == 1
    replay = await browser_observe(BrowserObserveParams(**params))
    assert replay.metadata["independent_readback"]["status"] == "unavailable"
    assert state["reads"] == state["posts"] == 1


@pytest.mark.parametrize("change", ["input", "document", "expiry", "url", "token", "page", "session"])
@pytest.mark.parametrize("repeat_url", [True, False])
async def test_readback_snapshot_does_not_survive_input_or_document_change(screen, site, change, repeat_url):
    import time

    from rune.capabilities.browser.capabilities import BrowserActParams, browser_act
    from rune.capabilities.browser.session import current_session

    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + "/form"))
    params = {"readbackUrl": url + "/stored", "checks": [check("#missing", "people to 3", "3", "value")]}
    first = await browser_observe(BrowserObserveParams(**params))
    params["readbackSnapshot"] = first.metadata["independent_readback"]["snapshot"]
    if not repeat_url:
        params.pop("readbackUrl")
    if change == "input":
        assert (await browser_act(BrowserActParams(action="type", selector="#people", value="3"))).success
    elif change == "document":
        await screen.reload()
    elif change == "expiry":
        current_session().readback_document["created"] = time.monotonic() - 61
    elif change == "url":
        params["readbackUrl"] = url + "/form"
    elif change == "token":
        params["readbackSnapshot"] = "unknown"
    elif change == "page":
        current_session().readback_document["page"] = object()
    else:
        current_session().readback_document = None
    result = await browser_observe(BrowserObserveParams(**params))
    assert result.metadata["independent_readback"]["status"] == "unavailable"
    assert "snapshot expired" in result.output
    assert state["reads"] == 1 and state["posts"] == 0


@pytest.mark.parametrize("mode,expected", [("saved", "pass"), ("fake", "fail")])
async def test_record_discovery_and_checks_share_one_get_without_reopening_the_form(screen, site, mode, expected):
    url, state = site
    state["mode"] = mode
    evidence = await start("Set people to 3 and save", [
        {"requirement": "people to 3", "expected": "3"}, {"requirement": "save", "kind": "result"}])
    evidence.observe("browser_navigate", await browser_navigate(BrowserNavigateParams(url=url + "/form")))
    await act(evidence, action="type", selector="#people", value="3")
    await act(evidence, action="click", selector="#save")
    params = {"readbackUrl": url + "/stored"}
    evidence.prepare_tool("browser_observe", params)
    discovery = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", discovery)
    report = discovery.metadata["independent_readback"]
    assert report["status"] == "unavailable" and report["checks"] == []
    assert evidence.acceptance()["status"] != "pass"
    params = {"readbackSnapshot": report["snapshot"], "checks": [
        check(report["fields"][0]["selector"], "people to 3", "3", "value"),
        check("#receipt", "save", "Saved"),
    ]}
    evidence.prepare_tool("browser_observe", params)
    assert len(params["checks"]) == 1
    result = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", result)
    assert result.metadata["independent_readback"]["source"] == "readback_snapshot"
    assert evidence.acceptance()["status"] == expected
    assert state["reads"] == state["posts"] == 1
    assert screen.url == url + "/form"
    assert await screen.locator("#people").input_value() == "3"
