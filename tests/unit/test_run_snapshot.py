from rune.api.run_snapshot import RunSnapshots
from rune.api.server import SseClientManager


def test_late_results_cannot_turn_a_cancelled_run_into_a_completed_run():
    runs = RunSnapshots()
    runs.start("r1", "s1", "report")
    runs.record("question", {"runId": "r1", "id": "q1"})
    runs.record("agent_aborted", {"runId": "r1", "trust": {"verified": False}})
    runs.record("question_closed", {"runId": "r1", "id": "q1"})
    assert runs.record("agent_complete", {"runId": "r1", "success": True}) is None
    runs.record("agent_aborted", {"runId": "r1", "trust": {"artifactReceipts": [{"revision": "a"}]}})
    run = runs.latest("s1")
    assert run["status"] == "cancelled" and run["question"] is None
    assert run["trust"]["artifactReceipts"] == [{"revision": "a"}]


def test_snapshot_joins_text_and_pairs_concurrent_tool_results():
    runs = RunSnapshots()
    runs.start("r1", "s1", "report")
    for call_id in ("a", "b"):
        runs.record("tool_call", {"runId": "r1", "callId": call_id, "toolName": "read"})
    runs.record("text_delta", {"runId": "r1", "delta": "one "})
    runs.record("text_delta", {"runId": "r1", "delta": "two"})
    runs.record("tool_result", {"runId": "r1", "callId": "b", "result": "second", "success": True})
    run = runs.latest("s1")
    assert run["text"] == "one two"
    assert "result" not in run["toolCalls"][0]
    assert run["toolCalls"][1]["result"] == "second"
    run["toolCalls"].clear()
    assert len(runs.latest("s1")["toolCalls"]) == 2


def test_slow_sse_client_is_told_to_restore_instead_of_silently_losing_text():
    manager = SseClientManager()
    queue = manager.add_client("slow")
    for _ in range(queue.maxsize + 1):
        manager.broadcast("text_delta", {"delta": "x"})
    assert queue.qsize() == 1
    assert "event: resync_required" in queue.get_nowait()


def test_an_agent_error_is_failed_even_when_delivered_in_the_final_event():
    runs = RunSnapshots()
    runs.start("r1", "s1", "report")
    runs.record("agent_complete", {"runId": "r1", "success": False,
                                  "trust": {"completionStatus": "failed", "reason": "error: invalid request"}})
    assert runs.latest("s1")["status"] == "failed"


def test_snapshot_preserves_check_failure_separately_from_process_exit():
    runs = RunSnapshots()
    runs.start("r1", "s1", "verify")
    runs.record("tool_call", {"runId": "r1", "callId": "a", "toolName": "bash_execute"})
    runs.record("tool_result", {"runId": "r1", "callId": "a", "success": True,
                               "result": "FAILED (failures=5)", "checkStatus": "fail", "outputTruncated": True})
    call = runs.latest("s1")["toolCalls"][0]
    assert call["success"] and call["checkStatus"] == "fail" and call["outputTruncated"]
    assert call["completedAt"] >= call["timestamp"]


def test_early_background_usage_is_combined_when_the_answer_arrives():
    runs = RunSnapshots()
    runs.start('r1', 's1', 'report')
    part = {'total': 2, 'input': 1, 'output': 1,
            'cost': {'usd': .01, 'knownUsd': .01, 'unpricedCalls': 0}}
    runs.record('usage_update', {'runId': 'r1', 'maintenance': {'status': 'completed', 'usage': part}})
    event = runs.record('agent_complete', {'runId': 'r1', 'usage': part, 'answer': 'done'})
    assert event['usage']['total'] == 4
    assert event['usage']['cost']['usd'] == .02
    assert runs.get('r1')['status'] == 'completed'


def test_interrupted_background_accounting_does_not_leave_a_finished_chat_updating():
    runs = RunSnapshots()
    runs.start('r1', 's1', 'report')
    usage = {'total': 2, 'input': 1, 'output': 1,
             'cost': {'usd': .01, 'knownUsd': .01, 'unpricedCalls': 0}}
    runs.record('agent_complete', {'runId': 'r1', 'usage': usage})
    runs.record('usage_update', {'runId': 'r1', 'maintenance': {'status': 'running'}})
    assert runs.get('r1')['usage']['cost']['pending']
    runs.record('usage_update', {'runId': 'r1', 'maintenance': {'status': 'interrupted'}})
    run = runs.get('r1')
    assert run['status'] == 'completed'
    assert not run['usage']['cost'].get('pending')
    assert run['usage']['cost']['incomplete']
    assert run['usage']['cost']['usd'] is None
    assert run['usage']['cost']['knownUsd'] == .01
