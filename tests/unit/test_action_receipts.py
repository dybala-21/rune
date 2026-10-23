"""Each external change is graded by the evidence behind it, never by the tool's own claim alone."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rune.agent.action_receipts import ReceiptLedger, receipt_for
from rune.agent.file_outcomes import FileOutcomes
from rune.capabilities.file import (
    FileDeleteParams,
    FileEditParams,
    FileWriteParams,
    file_delete,
    file_edit,
    file_write,
)
from rune.types import CapabilityResult


@pytest.fixture(autouse=True)
def authorized_workspace(tmp_path, monkeypatch):
    def validate(path):
        return SimpleNamespace(allowed=Path(path).resolve().is_relative_to(tmp_path.resolve()),
                               requires_approval=False, reason="outside test workspace")

    monkeypatch.setattr("rune.capabilities.file.get_guardian", lambda: SimpleNamespace(
        validate_file_path=validate, validate_file_read_path=validate))
    monkeypatch.setattr("rune.safety.guardian.get_guardian", lambda: SimpleNamespace(
        validate_file_path=validate, validate_file_read_path=validate))


async def test_file_changes_are_verified_by_reading_the_disk(tmp_path):
    path = tmp_path / "보고서.txt"
    params = {"path": str(path), "content": "합계 0.6\n"}
    written = await file_write(FileWriteParams(**params))
    assert receipt_for("file_write", params, written)["tier"] == "verified"

    edit = {"path": str(path), "search": "0.6", "replace": "0.7"}
    expected = FileOutcomes(str(tmp_path)).expected_edit("file_edit", edit)
    edited = await file_edit(FileEditParams(**edit))
    assert receipt_for("file_edit", edit, edited, expected_edit=expected)["tier"] == "verified"
    # Without a precomputed expectation the edit is only the tool's report.
    assert receipt_for("file_edit", edit, edited)["tier"] == "acknowledged"

    deleted = await file_delete(FileDeleteParams(path=str(path)))
    assert receipt_for("file_delete", {"path": str(path)}, deleted)["tier"] == "verified"


async def test_content_that_differs_from_the_request_is_not_verified(tmp_path):
    path = tmp_path / "a.txt"
    params = {"path": str(path), "content": "requested"}
    written = await file_write(FileWriteParams(**params))
    path.write_text("something else")  # Another process changed it after the tool returned.
    assert receipt_for("file_write", params, written)["tier"] == "unknown"


async def test_unchanged_or_failed_file_operations_leave_no_receipt(tmp_path):
    path = tmp_path / "same.txt"
    path.write_text("x")
    unchanged = await file_write(FileWriteParams(path=str(path), content="x"))
    assert receipt_for("file_write", {"path": str(path), "content": "x"}, unchanged) is None
    assert receipt_for("file_write", {"path": str(path)}, CapabilityResult(success=False, error="denied")) is None


@pytest.mark.parametrize("status,tier", [(200, "acknowledged"), (201, "acknowledged"), (202, "unknown"),
                                         (409, "rejected"), (503, "rejected"), (500, "unknown"), (504, "unknown"), (0, "unknown")])
def test_api_writes_are_graded_by_response_status(status, tier):
    result = CapabilityResult(success=200 <= status < 300 and status != 202,
                              metadata={"status_code": status, "action_status": "completed" if status else "unknown"})
    receipt = receipt_for("web_fetch", {"url": "https://api.test/orders", "method": "post"}, result)
    assert receipt["tier"] == tier and receipt["target"] == "POST https://api.test/orders"


def test_reads_and_unsent_writes_are_not_changes():
    sent = CapabilityResult(success=True, metadata={"status_code": 200})
    assert receipt_for("web_fetch", {"url": "https://api.test/orders"}, sent) is None
    denied = CapabilityResult(success=False, metadata={"action_status": "not_executed"})
    assert receipt_for("web_fetch", {"url": "https://api.test/orders", "method": "DELETE"}, denied) is None


def browser_step(status, seqs=(3,)):
    return {"tool": "browser_act", "success": True, "metadata": {"write_receipt": {
        "status": status, "seqs": list(seqs),
        "requests": [{"method": "POST", "url": "https://app.test/save", "status": 200 if status == "accepted" else 503}]}}}


def test_every_call_keeps_its_own_receipt_and_only_the_read_back_save_is_verified():
    ledger = ReceiptLedger()
    ledger.record("browser_batch", {}, CapabilityResult(success=True, metadata={"browser_steps": [browser_step("rejected", (1,))]}),
                  call_id="c1")
    ledger.record("browser_act", {}, CapabilityResult(success=True, metadata=browser_step("accepted", (2,))["metadata"]),
                  call_id="c2")
    ledger.record("browser_act", {}, CapabilityResult(success=True, metadata=browser_step("accepted", (5,))["metadata"]),
                  call_id="c3")
    # The refused attempt stays visible next to the accepted ones.
    assert [r["tier"] for r in ledger.finalize({"status": "acknowledged"})] == ["rejected", "acknowledged", "acknowledged"]
    tiers = [r["tier"] for r in ledger.finalize({"status": "pass", "independentReadback": {"status": "pass"},
                                                  "savedWriteSeqs": [5]})]
    assert tiers == ["rejected", "acknowledged", "verified"]
    # A page-only pass is not a stored-record readback.
    page_only = ledger.finalize({"status": "pass", "readback": {"status": "pass"}, "savedWriteSeqs": [5]})
    assert [r["tier"] for r in page_only] == ["rejected", "acknowledged", "acknowledged"]
    assert all("seqs" not in r for r in page_only)


def test_a_later_failure_does_not_hide_an_earlier_success():
    ledger = ReceiptLedger()
    ledger.record("mcp.slack.send_message", {"channel": "a"}, CapabilityResult(success=True), call_id="c1")
    ledger.record("mcp.slack.send_message", {"channel": "b"}, CapabilityResult(success=False, metadata={"mcp_error": True}),
                  call_id="c2")
    for status, call in ((201, "c3"), (500, "c4")):
        ledger.record("web_fetch", {"url": "https://api.test/orders", "method": "POST"},
                      CapabilityResult(success=status < 300, metadata={"status_code": status}), call_id=call)
    assert [r["tier"] for r in ledger.finalize(None)] == ["acknowledged", "unknown", "acknowledged", "unknown"]


def test_a_call_cut_off_before_its_result_is_listed_as_unknown():
    ledger = ReceiptLedger()
    ledger.begin("c1", "web_fetch", {"url": "https://api.test/orders?token=SECRET", "method": "POST"})
    ledger.begin("c2", "web_fetch", {"url": "https://api.test/orders"})  # a read is not a change
    ledger.begin("c3", "bash_execute", {"command": "rm -rf build", "cwd": "/w"})
    ledger.begin("c4", "browser_act", {"action": "click"})
    ledger.discard("c4")  # replayed: nothing was sent again
    [post, shell] = ledger.finalize(None)
    assert post["tier"] == shell["tier"] == "unknown"
    assert post["target"] == "POST https://api.test/orders?…" and "SECRET" not in str(post)
    ledger.record("bash_execute", {"command": "rm -rf build", "cwd": "/w"},
                  CapabilityResult(success=True, metadata={"shell_changes": {"created": [], "modified": [], "removed": []}}),
                  call_id="c3")
    assert [r["target"] for r in ledger.finalize(None)] == [post["target"]]


def test_urls_never_carry_credentials_or_query_secrets():
    from rune.agent.action_receipts import redact_url

    assert redact_url("https://user:pw@api.test:8443/v1/c?token=S#frag") == "https://api.test:8443/v1/c?…"
    receipt = receipt_for("web_fetch", {"url": "https://api.test/hook?key=S", "method": "PUT"},
                          CapabilityResult(success=True, metadata={"status_code": 200}))
    assert receipt["target"] == "PUT https://api.test/hook?…"


def test_shell_without_a_snapshot_is_unknown_and_noise_is_summarized(tmp_path):
    ledger = ReceiptLedger()
    ledger.record("bash_execute", {"command": "make", "cwd": str(tmp_path)},
                  CapabilityResult(success=True, metadata={"shell_untracked": True}), call_id="c1")
    created = [f"pkg/__pycache__/m{i}.pyc" for i in range(5)] + [f"out/f{i}.txt" for i in range(25)]
    ledger.record("bash_execute", {"command": "build", "cwd": str(tmp_path)},
                  CapabilityResult(success=True, metadata={"shell_changes": {"created": created, "modified": [], "removed": []}}),
                  call_id="c2")
    receipts = ledger.finalize(None)
    assert receipts[0]["tier"] == "unknown" and "no snapshot" in receipts[0]["evidence"]
    assert not any("__pycache__" in r["target"] for r in receipts)
    assert len(receipts) == 1 + 20 + 1 and receipts[-1]["target"].startswith("5 more created")


def test_dropped_receipts_are_counted_and_shell_noise_goes_first():
    ledger = ReceiptLedger()
    ledger.record("web_fetch", {"url": "https://api.test/a", "method": "POST"},
                  CapabilityResult(success=True, metadata={"status_code": 200}), call_id="first")
    for i in range(70):
        ledger.record("bash_execute", {"command": "x", "cwd": "/w"}, CapabilityResult(success=True, metadata={
            "shell_changes": {"created": [f"f{i}"], "modified": [], "removed": []}}), call_id=f"s{i}")
    receipts = ledger.finalize(None)
    assert receipts[0]["target"] == "POST https://api.test/a"
    assert receipts[-1]["tool"] == "ledger" and receipts[-1]["target"].startswith("7 earlier")


@pytest.mark.parametrize("tool,params,metadata,tier", [
    ("connector_request", {"connector": "crm", "method": "POST", "path": "/contacts"}, {"status_code": 201}, "acknowledged"),
    ("connector_request", {"connector": "crm", "method": "GET", "path": "/contacts"}, {"status_code": 200}, None),
    ("document_bundle", {}, {"paths": ["/w/r.xlsx"], "verified": True}, "verified"),
    ("document_create", {}, {"path": "/w/r.docx"}, "acknowledged"),
    ("credential_save", {"key": "API_TOKEN", "value": "SECRET"}, {}, "acknowledged"),
    ("cron_create", {"name": "daily"}, {}, "acknowledged"),
])
def test_other_writing_tools_leave_receipts(tool, params, metadata, tier):
    receipt = receipt_for(tool, params, CapabilityResult(success=True, metadata=metadata))
    assert (receipt or {}).get("tier") == tier
    assert "SECRET" not in str(receipt)


def test_a_broken_result_never_breaks_the_run():
    ledger = ReceiptLedger()
    ledger.record("web_fetch", {"method": "POST", "url": object()}, CapabilityResult(success=True, metadata=None))
    assert isinstance(ledger.finalize(None), list)


def test_receipts_reach_the_trust_payload_on_completion_and_cancellation():
    from rune.api.trust import build_cancelled_trust, build_trust_payload

    change = {"tool": "file_write", "target": "/w/a.txt", "tier": "verified", "evidence": "hash"}
    assert build_trust_payload(SimpleNamespace(reason="completed", change_receipts=[change]))["changeReceipts"] == [change]
    # A stop before the loop returns still reports what was already changed.
    assert build_cancelled_trust(None, change_receipts=[change])["changeReceipts"] == [change]
    assert build_cancelled_trust(SimpleNamespace(reason="cancelled", change_receipts=[change]))["changeReceipts"] == [change]


@pytest.mark.parametrize("result,tier", [
    (CapabilityResult(success=True, metadata={"mcp_server": "crm"}), "acknowledged"),
    # A tool-level error may follow partial side effects.
    (CapabilityResult(success=False, metadata={"mcp_error": True}), "unknown"),
    (CapabilityResult(success=False, metadata={"action_status": "unknown"}), "unknown"),
])
def test_service_writes_are_graded_by_the_service_response(result, tier):
    assert receipt_for("mcp.crm.create_contact", {"name": "Kim"}, result)["tier"] == tier


def test_service_reads_and_declined_writes_are_not_changes():
    assert receipt_for("mcp.crm.list_contacts", {}, CapabilityResult(success=True)) is None
    declined = CapabilityResult(success=False, metadata={"action_status": "not_executed", "approval_decision": "denied"})
    assert receipt_for("mcp.crm.create_contact", {}, declined) is None


@pytest.mark.parametrize("outcome,tier", [("ok", "acknowledged"), ("refused", "unknown"), ("dropped", "unknown")])
async def test_mcp_bridge_results_carry_what_the_service_did(outcome, tier):
    from unittest.mock import AsyncMock

    from rune.mcp.bridge import _mcp_tool_to_capability
    from rune.mcp.client import MCPTool, MCPToolResult

    manager = AsyncMock()
    if outcome == "dropped":
        manager.call_tool.side_effect = ConnectionError("socket closed")
    else:
        manager.call_tool.return_value = MCPToolResult(content=[{"type": "text", "text": "done"}],
                                                       is_error=outcome == "refused")
    capability = _mcp_tool_to_capability("crm", MCPTool(name="create_contact", description="", input_schema={}), manager)
    result = await capability.execute({"name": "Kim"})
    assert receipt_for(capability.name, {"name": "Kim"}, result)["tier"] == tier


@pytest.mark.parametrize("name,changes", [
    ("get_address", False), ("list_posts", False), ("search_updates", False), ("getOrCreateUser", True),
    ("merge_pull_request", True), ("upload_file", True), ("comment_on_issue", True), ("createIssue", True),
    ("frobnicate", True),  # an unknown verb is reported rather than hidden
])
def test_mcp_change_detection_reads_tool_names_as_words(name, changes):
    from rune.safety.tool_policy import is_mcp_write_operation, mcp_changes_state

    assert mcp_changes_state(f"mcp.svc.{name}") is changes
    assert not mcp_changes_state(f"mcp.svc.{name}", read_only_hint=True)
    if changes and name != "frobnicate":
        assert is_mcp_write_operation(f"mcp.svc.{name}")  # approval never gets looser


def test_an_accepted_save_is_never_reported_as_verified_by_another_check():
    from rune.agent.run_outcome import run_outcome

    trace = {"reason": "completed", "requirement_acceptance": {"required": True, "status": "acknowledged"},
             "evidence_gate": {"has_check": True, "last_verdict": "pass"}, "mech_check": "pass"}
    outcome = run_outcome(trace)
    assert outcome.success and not outcome.verified and outcome.status == "completed"
