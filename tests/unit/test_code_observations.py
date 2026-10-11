"""Intermediate successful changes must survive noisy follow-up tools."""

from rune.agent.code_observations import MAX_RECORDS, MAX_REVIEW_CHARS, review_observations
from rune.agent.test_claims import TestClaimGate as ClaimGate
from rune.agent.test_claims import code_observations
from rune.types import CapabilityResult


def test_successful_middle_edit_survives_reads_and_failed_retries():
    gate = ClaimGate()
    for path in ("money.py", "invoices.py", "test_invoices.py"):
        gate.observe("file_read", {"path": path}, CapabilityResult(success=True, output=f"Original {path}"))
    gate.observe("file_edit", {"path": "money.py"}, CapabilityResult(success=True, output="Applied HALF_UP"))
    gate.observe("file_edit", {"path": "invoices.py"}, CapabilityResult(success=True, output="Excluded cancelled"))
    for index in range(30):
        gate.observe("file_read", {"path": "test_invoices.py"}, CapabilityResult(success=True, output=f"Read {index}"))
        gate.observe("file_edit", {"path": "invoices.py"}, CapabilityResult(success=False, error=f"Failed retry {index}"))
    observations = review_observations(gate._observations)
    by_id = {entry["id"]: entry["text"] for entry in observations}
    assert "Applied HALF_UP" in by_id[4]
    assert "Excluded cancelled" in by_id[5] and "success=True" in by_id[5]
    assert "Original invoices.py" in by_id[2]
    assert "Failed retry 29" in observations[-1]["text"] and "success=False" in observations[-1]["text"]
    assert len(gate._observations) <= MAX_RECORDS
    assert sum(map(len, by_id.values())) <= MAX_REVIEW_CHARS


def test_applied_patch_and_execution_status_survive_large_requests():
    gate = ClaimGate()
    for index in range(20):
        gate.observe("file_edit", {"path": f"{index}.py", "replace": "x" * 10_000},
                     CapabilityResult(success=True, output="Edited", metadata={"fileChange": {
                         "path": f"{index}.py", "patch": "-HALF_EVEN\n+HALF_UP\n"}}))
    for record in review_observations(gate._observations):
        assert record["text"].startswith("success=True")
        assert "+HALF_UP" in record["text"]
        assert "x" * 100 not in record["text"]


def test_transcript_fallback_keeps_middle_file_changes_without_inventing_success():
    messages = []
    for index in range(20):
        name, path = ("file_edit", "invoices.py") if index == 5 else ("file_read", "money.py")
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": str(index), "function": {
                "name": name, "arguments": '{"path":"' + path + '"}'}}]},
            {"role": "tool", "tool_call_id": str(index), "content": f"Observation {index}"},
        ])
    records = code_observations(messages)
    middle = next(record for record in records if record["id"] == 6)
    assert "invoices.py" in middle["text"] and "Observation 5" in middle["text"]
    assert "success=True" not in middle["text"]


def test_review_excerpt_bound_includes_truncation_markers_and_status():
    gate = ClaimGate()
    for index in range(30):
        gate.observe("file_edit", {"path": f"{index}.py", "replace": "x" * 10_000},
                     CapabilityResult(success=index % 2 == 0, output="y" * 10_000))
    observations = review_observations(gate._observations)
    assert sum(len(record["text"]) for record in observations) <= MAX_REVIEW_CHARS
    assert all("[excerpt omitted]" in record["text"] for record in observations)
    assert all(record["text"].startswith("success=") for record in observations)
