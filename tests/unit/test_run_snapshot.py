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
