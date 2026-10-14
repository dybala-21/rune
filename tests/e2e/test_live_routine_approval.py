"""Test routine approval with a real model and a fake submission endpoint."""

import json

import httpx
import pytest
from fastapi import FastAPI

from rune.api.handlers import cron as api
from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy
from rune.types import CapabilityResult


@pytest.mark.parametrize("with_draft", [False, True], ids=["submission", "document"])
async def test_changed_input_requires_a_new_approval(live_model, live_report, tmp_path, monkeypatch, with_draft):
    from rune.agent import failover, loop
    from rune.capabilities.registry import get_capability_registry
    from rune.config import get_config
    from rune.utils.paths import rune_data

    cfg = get_config()
    cached_models = rune_data() / 'models'
    home, work = tmp_path / 'state', tmp_path / 'work'
    work.mkdir()
    home.mkdir()
    if cached_models.is_dir():
        (home / 'data').mkdir()
        (home / 'data' / 'models').symlink_to(cached_models, target_is_directory=True)
    monkeypatch.setenv('RUNE_HOME', str(home))
    monkeypatch.setenv('RUNE_WORKSPACE', str(work))
    monkeypatch.setenv('RUNE_ISOLATION_ROOT', str(work))
    monkeypatch.setenv('RUNE_APPROVAL_MODE', 'standard')
    monkeypatch.setenv('RUNE_HYBRID_API', '1')
    cfg.proactive.enabled = False
    cfg.llm.reasoning_effort = None
    cfg.llm.reasoning_efforts = {}
    cfg.llm.route_simple_queries = False
    profiles = failover.build_profiles_from_config()[:1]
    monkeypatch.setattr(failover, 'build_profiles_from_config', lambda: profiles)
    calls, submissions, results, classifications, requests = [], [], [], [], []
    original_loop = loop.NativeAgentLoop
    from rune.agent import request_budget
    from rune.utils.tokenizer import count_tokens

    original_estimate = request_budget.input_estimate

    def estimate(params):
        total = original_estimate(params)
        messages = params.get('messages', [])
        requests.append({'estimatedInput': total, 'toolCount': len(params.get('tools', [])),
                         'tools': count_tokens(json.dumps(params.get('tools', []), ensure_ascii=False)),
                         'system': count_tokens(json.dumps([m for m in messages if m.get('role') == 'system'], ensure_ascii=False)),
                         'conversation': count_tokens(json.dumps([m for m in messages if m.get('role') != 'system'], ensure_ascii=False))})
        return total
    monkeypatch.setattr(request_budget, 'input_estimate', estimate)

    def create_loop(config):
        agent = original_loop(config)
        async def record(call):
            calls.append(call)
        async def classified(value):
            classifications.append({'type': value.goal_type, 'intents': sorted(value.intent_categories)})
        agent.on('tool_call', record)
        agent.on('goal_classified', classified)
        return agent
    monkeypatch.setattr(loop, 'NativeAgentLoop', create_loop)

    async def submit(params):
        assert params.url == 'https://reports.example.test/submit' and params.method.upper() == 'POST'
        submissions.append(params.body)
        return CapabilityResult(success=True, output=json.dumps({'accepted': params.body, 'receipt': 'fixture-1'}))
    monkeypatch.setattr(get_capability_registry().get('web_fetch'), 'execute', submit)
    store = MemoryStore(tmp_path / 'tasks.db')
    monkeypatch.setattr(cron, '_get_store', lambda: store)
    monkeypatch.setattr(api, 'get_memory_store', lambda: store)
    (work / 'source.txt').write_text('Shipment 731: ready')
    policy = RoutinePolicy(workspace=str(work), input_paths=['source.txt'], output_paths=['draft.txt'] if with_draft else [],
                           max_steps=6, timeout_seconds=90, token_budget=25000,
                           total_tokens=60000, total_model_requests=14, total_cost_usd=0.35)
    goal = ('Read source.txt and submit its exact text once as the body '
            'of web_fetch POST to https://reports.example.test/submit. Do not use a browser or shell. '
            'After the submission succeeds, report the accepted text and receipt briefly. Do not create files.')
    if with_draft:
        goal = ('Read source.txt, copy its exact text into draft.txt, then submit that exact text once as the body '
                'of web_fetch POST to https://reports.example.test/submit. Do not use a browser or shell. '
                'After the submission succeeds, verify the draft and report the accepted text and receipt briefly.')
    job_id = store.create_cron_job(name='Approval fixture', schedule='* * * * *', max_runs=1,
                                   command=cron._pack_command('', goal, '', '', policy))
    outcome, usage = 'failed', {}

    def latest():
        records = ExecutionStore(home / 'data' / 'routine-executions.db')
        try:
            return records.recent(f'cron:{job_id}:')[0], records.usage(f'cron:{job_id}:')
        finally:
            records.close()

    try:
        await cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id)))
        first, usage = latest()
        results.append(first['result'])
        assert first['result']['status'] == 'needs_approval', first['result'].get('error')
        assert first['result']['pending_approval']['action']['tool'] == 'web_fetch'
        store.close()
        store = MemoryStore(tmp_path / 'tasks.db')
        for _ in range(3):
            await cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id)))
        assert latest()[1] == usage and not submissions
        (work / 'source.txt').write_text('Shipment 731: delivered')
        app = FastAPI()
        app.include_router(api.router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            route = f'/cron/{job_id}/approval'
            response = await client.post(route, json={'operationId': first['id'], 'approved': True})
            assert response.status_code == 409 and not submissions
            response = await client.post(route + '/refresh', json={'operationId': first['id']})
            assert response.status_code == 200, response.text
            second, usage = latest()
            results.append(second['result'])
            assert second['result']['status'] == 'needs_approval', second['result'].get('error')
            assert second['result']['pending_approval']['action']['params']['body'] == 'Shipment 731: delivered'
            assert not submissions
            response = await client.post(route, json={'operationId': second['id'], 'approved': True})
            assert response.status_code == 200, response.text
            completed, usage = latest()
            results.append(completed['result'])
            assert completed['result']['success'], completed['result'].get('error')
            assert submissions == ['Shipment 731: delivered']
            if with_draft:
                assert (work / 'draft.txt').read_text() == 'Shipment 731: delivered'
            assert 'delivered' in completed['result']['output'] and 'fixture-1' in completed['result']['output']
            await client.post(route, json={'operationId': second['id'], 'approved': True})
            assert latest()[1] == usage and len(submissions) == 1
        assert not any(call.get('toolName', call.get('name', '')).startswith(('browser_', 'desktop_')) for call in calls)
        outcome = 'passed'
    finally:
        from scripts.e2e_provenance import background_run
        live_report({'outcome': outcome, 'scope': 'Native loop and HTTP approval; store reopen, changed input, one synthetic external effect',
                     'results': results, 'runs': [background_run(result) for result in results],
                     'usage': usage, 'tool_calls': calls, 'submissions': submissions,
                     'classifications': classifications, 'request_sizes': requests})
        print(json.dumps({'outcome': outcome, 'usage': usage, 'submissions': len(submissions)}), flush=True)
        store.close()
