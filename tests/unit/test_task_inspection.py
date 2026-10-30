"""Inspections read saved work without acquiring a writer's permissions."""

from unittest.mock import AsyncMock, Mock

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.agent.tool_adapter import (
    StallState,
    ToolAdapterOptions,
    _build_typed_tool,
    build_tool_set,
)
from rune.agent.tool_output import RejectedToolResult
from rune.capabilities.file import register_file_capabilities
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.task_ops import TaskCreateParams, TaskListParams, task_create, task_list
from rune.capabilities.types import CapabilityDefinition
from rune.memory.task_store import TaskStore
from rune.safety.execution_environment import environment_scope
from rune.safety.task_inspection import (
    inspecting_task,
    inspection_blocker,
    inspection_scope,
    restrict_to_inspection,
)
from rune.types import CapabilityResult, CompletionTrace
from tests.unit.test_task_continuity import flow as flow
from tests.unit.test_task_continuity import run


async def test_inspection_reads_files_but_cannot_write_while_another_run_owns_the_task(flow, monkeypatch):
    original = flow.manager.start_conversation('web:session')
    loop, _ = await run(flow, original, 'Update the release document')
    identity = loop._task_context['id']
    store = TaskStore()
    before = store.get('local', str(flow.root), identity)
    store.claim('local', str(flow.root), identity, before['revision'], 'active-writer')
    path = flow.root / 'release.txt'
    path.write_text('Keep the appendix')
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    released = []
    original_release = TaskStore.release

    def release(*args):
        released.append(args[-1])
        original_release(*args)

    async def inspect(loop, **kwargs):
        assert inspecting_task()
        assert 'task_list' in loop._select_tools(kwargs['classification'])
        read = await registry.execute('file_read', {'path': str(path)})
        assert read.success and 'Keep the appendix' in read.output
        write = await registry.execute('file_write', {'path': str(path), 'content': 'Lost appendix'})
        assert not write.success and write.metadata['action_status'] == 'not_executed'
        assert write.metadata['failure_kind'] == 'task_inspection_read_only'
        return CompletionTrace(reason='completed')

    monkeypatch.setattr(TaskStore, 'release', release)
    monkeypatch.setattr(NativeAgentLoop, '_execute_loop', inspect)
    flow.requested.update(id=identity, scope='inspect', type='chat')
    try:
        _, trace = await run(flow, flow.manager.start_conversation('web:local'), 'What remains?')
        assert trace.reason == 'completed'
        assert path.read_text() == 'Keep the appendix'
        assert store.get('local', str(flow.root), identity) == before
        assert released == []
        assert not inspecting_task()
    finally:
        original_release(store, 'local', str(flow.root), 'active-writer')


async def test_nested_run_inherits_inspection_but_the_next_turn_can_edit(flow, monkeypatch):
    loop, _ = await run(flow, flow.manager.start_conversation('web:local'), 'Fix invoice.mjs')
    identity = loop._task_context['id']
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    path = flow.root / 'invoice.mjs'
    path.write_text('export const amount = 1;\n')
    reached = []

    async def execute(loop, *, goal, **kwargs):
        reached.append(goal)
        if goal == 'Inspect the saved task':
            child = NativeAgentLoop()
            result = await child.run('Nested edit', context={'workspace_root': str(flow.root)},
                                    classification=ClassificationResult('code_modify', .99, 2))
            assert result.reason == 'completed'
        else:
            result = await registry.execute('file_write', {'path': str(path), 'content': 'export const amount = 2;\n'})
            assert result.success is (goal == 'Apply the change')
        return CompletionTrace(reason='completed')

    monkeypatch.setattr(NativeAgentLoop, '_execute_loop', execute)
    flow.requested.update(id=identity, scope='inspect', type='code_read')
    _, trace = await run(flow, flow.manager.start_conversation('web:local'), 'Inspect the saved task')
    assert trace.reason == 'completed'
    assert reached == ['Inspect the saved task', 'Nested edit']
    assert path.read_text() == 'export const amount = 1;\n'
    assert not inspecting_task()
    flow.requested.update(scope='task', type='code_modify')
    _, trace = await run(flow, flow.manager.start_conversation('web:local'), 'Apply the change')
    assert trace.reason == 'completed'
    assert path.read_text() == 'export const amount = 2;\n'


@pytest.mark.parametrize('name,params', [
    ('file_write', {}), ('file_edit', {}), ('file_delete', {}), ('document_create', {}),
    ('task_create', {}), ('task_update', {}), ('memory_save', {}), ('safety_tune', {}),
    ('cron_create', {}), ('cron_update', {}), ('bash_execute', {'command': 'echo unsafe'}),
    ('delegate_task', {}), ('browser_act', {}), ('browser_batch', {}), ('desktop_act', {}),
    ('connector_request', {}), ('mcp.example.read_data', {}), ('new_tool', {}),
    ('web_fetch', {'method': 'POST'}), ('web_fetch', {'method': 'GET', 'body': 'side effect'}),
    ('browser_screenshot', {'path': 'overwrite.png'}),
])
async def test_inspection_blocks_before_approvals_cache_or_execution(name, params):
    invoke = AsyncMock(return_value=CapabilityResult(success=True))
    approval = AsyncMock(return_value=True)
    registry = CapabilityRegistry()
    cap = CapabilityDefinition(name=name, description='Declared safe', group='safe', execute=invoke)
    registry.register(cap)
    cache = Mock()
    wrapper = _build_typed_tool(cap_def=cap, opts=ToolAdapterOptions(approval_callback=approval),
                                reg=registry, cache=cache, stall=StallState())
    with inspection_scope():
        restrict_to_inspection()
        result = await wrapper.function(**params)
        assert isinstance(result, RejectedToolResult) and result.code == 'task_inspection_read_only'
        direct = await registry.execute(name, params)
        assert not direct.success and direct.metadata['action_status'] == 'not_executed'
        assert name not in build_tool_set(ToolAdapterOptions(), registry=registry) or name in {'web_fetch', 'browser_screenshot'}
    cache.generate_key.assert_not_called()
    approval.assert_not_awaited()
    invoke.assert_not_awaited()
    assert not inspecting_task()


@pytest.mark.parametrize('name,params', [
    ('task_list', {}), ('file_read', {}), ('file_query', {}), ('document_read', {}), ('code_find_refs', {}),
    ('web_fetch', {}), ('web_fetch', {'method': 'HEAD'}), ('browser_observe', {}), ('browser_screenshot', {}),
])
def test_inspection_keeps_supported_read_operations(name, params):
    with inspection_scope():
        restrict_to_inspection()
        assert inspection_blocker(name, params) is None


async def test_manual_task_detail_preserves_the_last_requirement_and_list_stays_short(tmp_path):
    from rune.agent.tool_adapter import _format_tool_output

    goal = ('Keep the sources. ' * 500)[:7900] + ' 반드시 법적 부록을 보존할 것.'
    with environment_scope(str(tmp_path)):
        created = await task_create(TaskCreateParams(goal=goal))
        identity = created.metadata['task_id']
        listed = await task_list(TaskListParams())
        detail = await task_list(TaskListParams(task_id=identity))
    assert '법적 부록' not in listed.output
    delivered = _format_tool_output('task_list', {'task_id': identity}, detail)
    assert goal in delivered
    assert '[...truncated' not in delivered
    assert len(listed.output) < 1000


async def test_inspection_does_not_restore_tests_or_run_generated_checks(tmp_path, monkeypatch):
    from rune.agent.evidence_gate import EvidenceGate

    path = tmp_path / 'test_release.py'
    path.write_text('assert True\n')
    gate = EvidenceGate('Inspect existing test results', str(tmp_path))
    path.write_text('assert 1 == 1\n')
    extract = AsyncMock(side_effect=AssertionError('No model call during inspection'))
    monkeypatch.setattr('rune.agent.evidence_spec.extract_spec', extract)
    with inspection_scope():
        restrict_to_inspection()
        state, _ = await gate.verdict()
    assert state == 'skip'
    assert path.read_text() == 'assert 1 == 1\n'
    assert gate.last_tests_restored == []
    extract.assert_not_awaited()


async def test_ambiguous_inspection_cannot_write_or_create_a_completed_task(flow, monkeypatch):
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    path = flow.root / 'should-not-exist.txt'

    async def inspect(loop, **kwargs):
        assert inspecting_task()
        result = await registry.execute('file_write', {'path': str(path), 'content': 'unexpected'})
        assert not result.success
        return CompletionTrace(reason='completed')

    monkeypatch.setattr(NativeAgentLoop, '_execute_loop', inspect)
    flow.requested.update(id='', scope='inspect', type='artifact')
    loop, trace = await run(flow, flow.manager.start_conversation('web:local'), 'Inspect previous work')
    assert trace.reason == 'completed'
    assert loop._task_context is None
    assert not path.exists() and TaskStore().list('local', str(flow.root)) == []


async def test_missing_selected_task_stops_before_any_action(flow):
    flow.requested.update(id='missing12345', scope='task')
    _, trace = await run(flow, flow.manager.start_conversation('web:local'), 'Continue the saved work')
    assert 'not in the loaded history' in trace.reason
    assert flow.executions == []
    assert TaskStore().list('local', str(flow.root)) == []
