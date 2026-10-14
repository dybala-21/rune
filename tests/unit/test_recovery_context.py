"""Check recovery prompt limits and preservation of stored records."""

import copy
from types import SimpleNamespace

from rune.agent.execution_journal import recovery_context
from rune.agent.timing import capture_timing
from rune.utils.tokenizer import count_tokens


def test_long_history_keeps_effects_and_bounds_read_previews():
    records = [{'tool': 'file_read', 'params': {'path': f'/tmp/input-{n}.txt'}, 'state': 'done',
                'effect': {'kind': 'read'}, 'result': {'output': 'long text ' * 300}} for n in range(100)]
    records.insert(0, {'tool': 'send_mail', 'params': {'recipient': 'fixture@example.test'}, 'state': 'done',
                       'effect': {'kind': 'opaque'}, 'result': {'output': 'receipt-42'}})
    original = copy.deepcopy(records)
    context = recovery_context({'runId': 'saved'}, records, max_tokens=1200)
    assert count_tokens(context) <= 1200
    assert 'receipt-42' in context and 'fixture@example.test' in context and 'omitted' in context
    assert records == original


def test_recovery_uses_remaining_parent_budget_and_keeps_file_changes():
    owner = SimpleNamespace(_token_budget=SimpleNamespace(total=16000, used=12000), _config=SimpleNamespace())
    records = [{'tool': 'file_read', 'params': {'path': '/tmp/source'}, 'state': 'done',
                'effect': {'kind': 'read', 'stale': True}, 'result': {'output': 'old data ' * 300}}]
    with capture_timing(owner):
        context = recovery_context({'runId': 'saved', 'fileChanges': {'inputs': {'modified': ['source.txt']}}}, records)
    assert count_tokens(context) <= 500
    assert 'source.txt' in context


def test_refreshing_read_revisions_does_not_allow_overwriting_changed_outputs(tmp_path):
    import pytest

    from rune.agent.execution_journal import RecoveryBlocked, fingerprint, reconcile
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.write_text('before')
    output.write_text('draft')
    records = [
        {'tool': 'file_read', 'state': 'done', 'effect': {'kind': 'read', 'path': str(source), 'after': fingerprint(str(source))}},
        {'tool': 'file_write', 'state': 'done', 'effect': {'kind': 'file', 'path': str(output), 'after': fingerprint(str(output))}},
    ]
    source.write_text('new input')
    refreshed = reconcile(records, refresh_reads=True)
    assert refreshed[0]['effect']['stale']
    output.write_text('edited by user')
    with pytest.raises(RecoveryBlocked):
        reconcile(records, refresh_reads=True)


def test_only_successful_completed_writes_count_as_recovery_evidence():
    from rune.agent.execution_journal import recovery_written_files

    records = [{"state": state, "tool": "file_write", "effect": {"kind": "file", "path": path},
                "result": {"success": success}}
               for state, success, path in [("done", True, "/saved"), ("done", False, "/failed"),
                                             ("unknown", True, "/uncertain"), ("started", True, "/started")]]
    assert recovery_written_files(records) == ["/saved"]


def test_recovery_distinguishes_changed_sources_outputs_and_completed_actions():
    from rune.agent.recovery_context import _task_state

    def record(tool, kind, path, *, stale=False, success=True):
        return {"tool": tool, "state": "done", "params": {},
                "effect": {"kind": kind, "path": path, "stale": stale},
                "result": {"success": success, "output": "receipt-42"}}
    records = [record("file_read", "read", "source.txt", stale=True),
               record("file_write", "file", "draft.txt"), record("file_write", "file", "failed.txt", success=False),
               record("file_read", "read", "source.txt"), record("web_fetch", "opaque", None)]
    state = _task_state({}, records)
    assert state["outputsToReview"] == state["outputsToVerify"] == ["draft.txt"]
    assert state["completedExternalActions"] == [{"tool": "web_fetch", "receipt": "receipt-42"}]
    records.append(record("file_write", "file", "draft.txt"))
    assert _task_state({}, records)["outputsToReview"] == []
    records.append(record("file_read", "read", "source.txt"))
    assert _task_state({}, records)["outputsToReview"] == []
    assert _task_state({}, records)["staleReadPaths"] == []
    assert _task_state({"fileChanges": {"inputs": {"modified": ["source.txt"]}}}, records)["outputsToReview"] == ["draft.txt"]
    context = recovery_context({"runId": "resume"}, records)
    assert '"success":false' in context and '"outputsToVerify":["draft.txt"]' in context


def test_completed_approval_is_identified_as_current_execution_not_a_pending_action():
    import json

    run = {'runId': 'resume', 'completedAction': {'tool': 'web_fetch', 'status': 'executed', 'verification': 'pending'}}
    records = [{'tool': 'web_fetch', 'params': {'method': 'POST'}, 'state': 'done',
                'effect': {'kind': 'opaque'}, 'result': {'success': True, 'output': 'receipt-42'}}]
    text = recovery_context(run, records)
    assert 'just succeeded in this run' in text and 'Do not resubmit' in text
    payload = json.loads(text[text.index('{'):])
    assert payload['completedAction']['verification'] == 'pending'
    assert 'receipt-42' in text
    assert 'just succeeded in this run' not in recovery_context({'runId': 'pending'}, records)
