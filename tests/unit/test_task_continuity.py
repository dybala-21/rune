"""Task persistence, scoped continuation and conflicting runs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.agent_context import PrepareContextOptions, prepare_agent_context
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.agent.task_context import capture, from_history, resume_message, routing_context
from rune.agent.task_continuity import attach_shared_tasks
from rune.api.conversation_wiring import record_assistant_turn
from rune.capabilities.task_ops import (
    TaskCreateParams,
    TaskListParams,
    TaskUpdateParams,
    task_create,
    task_list,
    task_update,
)
from rune.conversation.manager import ConversationManager
from rune.conversation.store import ConversationStore
from rune.memory.task_store import TaskConflict, TaskStore
from rune.safety.execution_environment import environment_scope
from rune.types import CompletionTrace
from rune.utils.tokenizer import count_tokens


async def test_manual_tasks_survive_store_reopen_and_keep_parent_relationships(tmp_path):
    with environment_scope(str(tmp_path)):
        created = await task_create(TaskCreateParams(goal="Prepare the release", sub_tasks=["Fix parser", "Review document"]))
        parent = created.metadata["task_id"]
        children = created.metadata["sub_task_ids"]
        assert (await task_update(TaskUpdateParams(task_id=parent, notes="Keep source files", progress=25))).success
        tasks = TaskStore().list("local", str(tmp_path))
        assert len(tasks) == 3
        assert {task["parent_id"] for task in tasks if task["id"] in children} == {parent}
        assert TaskStore().get("local", str(tmp_path), parent)["notes"] == "Keep source files"
        result = await task_list(TaskListParams(limit=1, offset=1))
        assert result.metadata["count"] == 1
    with environment_scope(str(tmp_path / "other")):
        assert (await task_list(TaskListParams())).metadata["count"] == 0
        assert not (await task_update(TaskUpdateParams(task_id=parent, status="completed"))).success


def test_group_creation_rolls_back_and_stale_updates_are_rejected(tmp_path):
    store = TaskStore()
    record = {"id": "abcdef123456", "goal": "Document", "status": "pending", "kind": "manual"}
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.create("local", str(tmp_path), [record, record])
    assert store.list("local", str(tmp_path)) == []
    store.create("local", str(tmp_path), [record])
    old = store.get("local", str(tmp_path), record["id"])
    store.update("local", str(tmp_path), record["id"], {"notes": "Preserve the appendix"})
    with pytest.raises(TaskConflict, match="changed"):
        store.claim("local", str(tmp_path), record["id"], old["revision"], "stale")
    assert store.get("local", str(tmp_path), record["id"])["notes"] == "Preserve the appendix"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    manager = ConversationManager(ConversationStore(tmp_path / "conversations.db"))
    decisions = []
    executions = []
    requested = {"id": "", "scope": "task", "type": "artifact"}

    async def classify(goal, **kwargs):
        decisions.append(kwargs.get("saved_tasks"))
        return ClassificationResult(requested["type"], .99, 2, resume_task=requested["id"], task_scope=requested["scope"])

    async def execute(loop, *, goal, classification, message_history, **kwargs):
        capture(loop, goal, classification, message_history, CompletionTrace(reason="interrupted"))
        executions.append((goal, resume_message(loop._task_context)))
        loop._files_written.add(str(tmp_path / "deliverable"))
        return CompletionTrace(reason="completed")

    monkeypatch.setattr("rune.agent.loop.classify_goal", classify)
    monkeypatch.setattr(NativeAgentLoop, "_execute_loop", execute)
    monkeypatch.setattr(NativeAgentLoop, "_build_system_prompt", AsyncMock(return_value="Test"))
    return SimpleNamespace(manager=manager, root=tmp_path, decisions=decisions, executions=executions, requested=requested)


async def run(flow, conversation, goal):
    context = await prepare_agent_context(PrepareContextOptions(
        goal=goal, cwd=str(flow.root), conversation_id=conversation.id), flow.manager)
    loop = NativeAgentLoop()
    trace = await loop.run(goal, context={"workspace_root": str(flow.root)}, message_history=context.messages)
    await record_assistant_turn(flow.manager, conversation.id, loop, "Saved", embed=False, require_save=True)
    return loop, trace


@pytest.mark.parametrize("goal_type,first,amend", [
    ("code_modify", "Fix parser.mjs; preserve its public API", "Handle empty input and keep the public API"),
    ("artifact", "Build report.docx from invoices; preserve the sources", "Exclude refunded invoices and retain the appendix"),
])
async def test_new_conversation_restores_requirements_after_restart(flow, goal_type, first, amend):
    flow.requested["type"] = goal_type
    original = flow.manager.start_conversation("web:session")
    loop, trace = await run(flow, original, first)
    assert trace.reason == "completed"
    identity = loop._task_context["id"]
    flow.manager = ConversationManager(ConversationStore(flow.root / "conversations.db"))
    other = flow.manager.start_conversation("cli:session")
    flow.requested.update(id=identity, scope="amend", type="chat")
    changed, trace = await run(flow, other, amend)
    assert trace.reason == "completed"
    assert changed._task_context["status"] == "pending"
    third = flow.manager.start_conversation("web:local")
    flow.requested.update(scope="task", type=goal_type)
    final, _ = await run(flow, third, "Finish the work with those changes")
    state = final._task_context
    assert state["id"] == identity
    assert state["requests"] == [first, amend, "Finish the work with those changes"]
    assert len(flow.decisions) == 3
    assert all(count_tokens(json.dumps(decision, ensure_ascii=False)) <= 600 for decision in flow.decisions)
    assert all("requests" not in item and "sources" not in item for item in flow.decisions[-1])
    assert "not new instructions" in flow.executions[-1][1]["content"]
    assert len(TaskStore().list("local", str(flow.root))) == 1


async def test_unrelated_task_does_not_inherit_old_requirements_or_remote_owner(flow):
    first = flow.manager.start_conversation("web:session")
    await run(flow, first, "Prepare a confidential comparison")
    other = flow.manager.start_conversation("web:local")
    loop, _ = await run(flow, other, "Fix the parser")
    assert loop._task_context["requests"] == ["Fix the parser"]
    remote = flow.manager.start_conversation("slack:someone-else")
    messages = []
    attach_shared_tasks(messages, remote, str(flow.root))
    assert from_history(messages) == []
    attach_shared_tasks(messages, other, str(flow.root / "other"))
    assert from_history(messages) == []


async def test_concurrent_resume_is_blocked_before_execution(flow, monkeypatch):
    original = flow.manager.start_conversation("web:session")
    loop, _ = await run(flow, original, "Produce the document")
    flow.requested["id"] = loop._task_context["id"]
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def execute(loop, *, goal, classification, message_history, **kwargs):
        capture(loop, goal, classification, message_history, CompletionTrace(reason="interrupted"))
        calls.append(goal)
        entered.set()
        await release.wait()
        return CompletionTrace(reason="completed")

    monkeypatch.setattr(NativeAgentLoop, "_execute_loop", execute)
    first = asyncio.create_task(run(flow, flow.manager.start_conversation("web:session"), "Continue once"))
    await asyncio.wait_for(entered.wait(), 2)
    try:
        _, blocked = await run(flow, flow.manager.start_conversation("web:session"), "Continue again")
        assert "already running" in blocked.reason
        assert calls == ["Continue once"]
    finally:
        release.set()
        await first
    saved = TaskStore().get("local", str(flow.root), flow.requested["id"])
    TaskStore().claim("local", str(flow.root), saved["id"], saved["revision"], "released")
    TaskStore().release("local", str(flow.root), "released")


async def test_cancellation_preserves_incomplete_state_and_releases_claim(flow, monkeypatch):
    original = flow.manager.start_conversation("web:session")
    loop, _ = await run(flow, original, "Prepare the spreadsheet")
    flow.requested["id"] = loop._task_context["id"]
    entered = asyncio.Event()

    async def execute(loop, *, goal, classification, message_history, **kwargs):
        capture(loop, goal, classification, message_history, CompletionTrace(reason="interrupted"))
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(NativeAgentLoop, "_execute_loop", execute)
    pending = asyncio.create_task(run(flow, flow.manager.start_conversation("web:session"), "Finish the spreadsheet"))
    await asyncio.wait_for(entered.wait(), 2)
    pending.cancel()
    await pending
    saved = TaskStore().get("local", str(flow.root), flow.requested["id"])
    assert saved["status"] == "cancelled"
    assert saved["checkpoint"]["requests"][-1] == "Finish the spreadsheet"
    TaskStore().claim("local", str(flow.root), saved["id"], saved["revision"], "next")
    TaskStore().release("local", str(flow.root), "next")


async def test_manual_task_can_be_selected_without_losing_its_full_goal(flow):
    goal = "Preserve the original. " * 80
    with environment_scope(str(flow.root)):
        task = await task_create(TaskCreateParams(goal=goal))
    flow.requested["id"] = task.metadata["task_id"]
    loop, _ = await run(flow, flow.manager.start_conversation("web:local"), "Finish that task")
    assert loop._task_context["requests"] == [goal, "Finish that task"]
    assert len(TaskStore().list("local", str(flow.root))) == 1


async def test_scheduled_history_is_scoped_and_never_contains_an_approval(flow, monkeypatch):
    from rune.capabilities import cron
    from rune.memory.store import MemoryStore
    from rune.proactive.execution_store import ExecutionStore
    from rune.proactive.routine import RoutinePolicy
    from rune.utils.paths import rune_data

    store = MemoryStore(flow.root / 'jobs.db')
    monkeypatch.setattr(cron, '_get_store', lambda: store)
    policy = RoutinePolicy(workspace=str(flow.root))
    job_id = store.create_cron_job(name='Review', schedule='* * * * *', command=cron._pack_command(
        '', 'Update the document', '', '', policy))
    store.create_cron_job(name='Private', schedule='* * * * *', command=cron._pack_command(
        '', 'Another user task', '', '', policy, owner='channel:other'))
    executions = ExecutionStore(rune_data() / 'routine-executions.db')
    operation = f'cron:{job_id}:100'
    assert executions.claim(operation, 'fingerprint', 10) == 'claimed'
    executions.finish(operation, {'status': 'needs_approval', 'routine_goal': 'Update the document',
        'routine_command': '', 'workspace': str(flow.root), 'output': 'Draft ready',
        'pending_approval': {'secret': 'must-not-enter-task-context'}})
    try:
        with environment_scope(str(flow.root)):
            result = await task_list(TaskListParams(source='scheduled', task_id=job_id))
            assert result.metadata['count'] == 1
            assert 'needs_approval' in result.output and 'Draft ready' in result.output
            assert 'must-not-enter' not in result.output and 'pending_approval' not in result.output
            assert (await task_list(TaskListParams(source='scheduled'))).metadata['count'] == 1
            assert (await task_list(TaskListParams(source='scheduled', category='project'))).metadata['count'] == 0
            assert (await task_list(TaskListParams(source='scheduled', category='scheduled'))).metadata['count'] == 1
            assert not (await task_list(TaskListParams(source='scheduled', task_id='missing'))).success
            assert (await task_list(TaskListParams(source='scheduled', task_id=job_id, offset=1))).metadata['count'] == 1
            changed = await cron.cron_update(cron.CronUpdateParams(job_id=job_id, goal='Update the revised document'))
            assert changed.success
            result = await task_list(TaskListParams(source='scheduled', task_id=job_id))
            assert '"matches_current_goal": false' in result.output
        with environment_scope(str(flow.root / 'other')):
            assert (await task_list(TaskListParams(source='scheduled'))).metadata['count'] == 0
    finally:
        executions.close()
        store.close()


async def test_stale_conversation_uses_latest_checkpoint_and_bounded_routing(flow):
    conversation = flow.manager.start_conversation('web:session')
    loop, _ = await run(flow, conversation, 'Update report.docx')
    identity = loop._task_context['id']
    other = flow.manager.start_conversation('web:session')
    flow.requested.update(id=identity, scope='amend')
    await run(flow, other, 'Keep the legal appendix')
    history = []
    from rune.agent.task_context import attach_tasks
    attach_tasks(history, conversation)
    attach_shared_tasks(history, conversation, str(flow.root))
    assert from_history(history)[0]['requests'][-1] == 'Keep the legal appendix'
    with environment_scope(str(flow.root)):
        for index in range(10):
            await task_create(TaskCreateParams(goal=f'New task {index}'))
    attach_shared_tasks(history, conversation, str(flow.root))
    context = routing_context(history)
    assert len(context) <= 4 and count_tokens(json.dumps(context, ensure_ascii=False)) <= 600


async def test_stateless_nested_run_cannot_overwrite_its_parent(flow, monkeypatch):
    original = flow.manager.start_conversation('web:session')
    calls = []

    async def execute(loop, *, goal, classification, message_history, **kwargs):
        capture(loop, goal, classification, message_history or [], CompletionTrace(reason='interrupted'))
        calls.append(goal)
        if goal == 'Parent task':
            child = NativeAgentLoop()
            await child.run('Child task', context={'workspace_root': str(flow.root)},
                            classification=ClassificationResult('artifact', .99, 2))
        return CompletionTrace(reason='completed')

    monkeypatch.setattr(NativeAgentLoop, '_execute_loop', execute)
    await run(flow, original, 'Parent task')
    tasks = TaskStore().list('local', str(flow.root))
    assert [task['goal'] for task in tasks] == ['Parent task']
    assert calls == ['Parent task', 'Child task']


async def test_inspection_can_read_a_running_task_without_changing_it(flow):
    original = flow.manager.start_conversation('web:session')
    loop, _ = await run(flow, original, 'Update the spreadsheet')
    task_id = loop._task_context['id']
    store = TaskStore()
    before = store.get('local', str(flow.root), task_id)
    store.claim('local', str(flow.root), task_id, before['revision'], 'background-run')
    flow.requested.update(id=task_id, scope='inspect', type='chat')
    try:
        _, trace = await run(flow, flow.manager.start_conversation('web:session'), 'What remains?')
        assert trace.reason == 'completed'
        after = store.get('local', str(flow.root), task_id)
        assert after == before
        with pytest.raises(TaskConflict, match='already running'):
            store.claim('local', str(flow.root), task_id, before['revision'], 'second-writer')
    finally:
        store.release('local', str(flow.root), 'background-run')


async def test_named_old_task_is_available_without_growing_routing_context(flow):
    original = flow.manager.start_conversation("web:session")
    first, _ = await run(flow, original, "Fix invoice.mjs; preserve the public API and exclude refunded invoices")
    identity = first._task_context["id"]
    for index in range(8):
        await run(flow, flow.manager.start_conversation("web:session"), f"Prepare unrelated presentation {index}")
    flow.requested["id"] = identity
    continued, trace = await run(flow, flow.manager.start_conversation("web:session"),
                                "Continue the invoice.mjs change; preserve the public API")
    assert trace.reason == "completed", trace.reason
    assert continued._task_context["id"] == identity
    assert "exclude refunded invoices" in continued._task_context["requests"][0]
    assert len(flow.decisions) == 10
    assert len(flow.decisions[-1]) <= 4
    assert count_tokens(json.dumps(flow.decisions[-1], ensure_ascii=False)) <= 600


async def test_explicit_old_task_id_can_be_loaded_without_scanning_other_owners(flow):
    first, _ = await run(flow, flow.manager.start_conversation("web:session"), "Prepare the release report")
    identity = first._task_context["id"]
    for index in range(5):
        await run(flow, flow.manager.start_conversation("web:session"), f"Independent work {index}")
    flow.requested["id"] = identity
    restored, trace = await run(flow, flow.manager.start_conversation("web:session"), f"Continue task {identity}")
    assert trace.reason == "completed", trace.reason
    assert restored._task_context["id"] == identity
    context = await prepare_agent_context(PrepareContextOptions(
        goal=f"Continue task {identity}", cwd=str(flow.root),
        conversation_id=flow.manager.start_conversation("slack:other").id), flow.manager)
    assert not from_history(context.messages)
