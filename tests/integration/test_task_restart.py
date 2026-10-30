"""Recover task records after a process exits while holding a task claim."""

import json
import os
import subprocess
import sys
from pathlib import Path

_WORKER = r'''
import asyncio, json, os, sys
from pathlib import Path
from rune.agent.task_continuity import attach_shared_tasks, stored_context
from rune.agent.task_context import from_history, resume_message
from rune.capabilities.task_ops import TaskCreateParams, task_create
from rune.conversation.types import Conversation
from rune.memory.task_store import TaskStore
from rune.safety.execution_environment import environment_scope
root = str(Path(sys.argv[2]).resolve())
store = TaskStore()
operation = sys.argv[1]
if operation == 'create':
    async def create():
        with environment_scope(root):
            office = await task_create(TaskCreateParams(goal='Update report.docx; preserve the source workbook'))
            code = await task_create(TaskCreateParams(goal='Fix invoice.mjs; preserve the public API'))
        return [office.metadata['task_id'], code.metadata['task_id']]
    print(json.dumps(asyncio.run(create())))
elif operation == 'interrupt':
    task = stored_context(store.get('local', root, sys.argv[3]))
    store.claim('local', root, task['id'], task['_revision'], 'interrupted-process')
    task.update(status='interrupted', blocker='Inspect existing output before retrying')
    store.checkpoint('local', root, task, lease='interrupted-process', conversation_id='old-chat')
    os._exit(9)
else:
    task = stored_context(store.get('local', root, sys.argv[3]))
    store.claim('local', root, task['id'], task['_revision'], 'new-process')
    task['requests'].append('Keep the appendix and exclude refunded invoices')
    task['status'] = 'pending'
    store.checkpoint('local', root, task, lease='new-process', conversation_id='new-chat')
    store.release('local', root, 'new-process')
    messages = []
    attach_shared_tasks(messages, Conversation(user_id='web:session'), root)
    restored = next(item for item in from_history(messages) if item['id'] == task['id'])
    private = []
    attach_shared_tasks(private, Conversation(user_id='slack:other'), root)
    print(json.dumps({'task':restored, 'private':from_history(private), 'resume':resume_message(restored)}))
'''


def test_task_claim_and_requirements_survive_process_restart(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    environment = {**os.environ, 'RUNE_HOME': str(tmp_path / 'state'), 'RUNE_WORKSPACE': str(workspace)}

    def worker(operation, task_id=''):
        return subprocess.run([sys.executable, '-c', _WORKER, operation, str(workspace), task_id],
                              cwd=repo, env=environment, capture_output=True, text=True, timeout=20)

    created = worker('create')
    assert created.returncode == 0, created.stderr
    office, code = json.loads(created.stdout)
    interrupted = worker('interrupt', office)
    assert interrupted.returncode == 9, interrupted.stderr
    recovered = worker('recover', office)
    assert recovered.returncode == 0, recovered.stderr
    result = json.loads(recovered.stdout)
    assert result['task']['id'] == office and result['task']['status'] == 'pending'
    assert result['task']['requests'] == ['Update report.docx; preserve the source workbook',
                                          'Keep the appendix and exclude refunded invoices']
    assert 'not new instructions' in result['resume']['content']
    assert result['private'] == []
    assert code != office
