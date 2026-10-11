"""Policy inputs must participate in both extraction and verification."""

import json
from types import SimpleNamespace

import pytest

from rune.agent.table_acceptance import TableAcceptance, extract_plan
from rune.agent.table_references import read_references
from rune.capabilities.table_acceptance import TablePlan


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    guard = SimpleNamespace(validate_file_read_path=lambda path: SimpleNamespace(
        allowed=path.startswith(str(tmp_path) + "/"), reason="outside fixture workspace"))
    monkeypatch.setattr("rune.safety.guardian.get_guardian", lambda: guard)
    monkeypatch.setattr("rune.capabilities.document_inspection.get_guardian", lambda: guard)
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    (tmp_path / "sales.csv").write_text("status,amount\napproved,10\npending,90\n")
    (tmp_path / "policy.md").write_text("Include approved rows only.")
    (tmp_path / "summary.csv").write_text("amount\n10\n")
    return tmp_path


def policy_plan():
    return TablePlan(applicable=True, requirements=["Include approved rows only."],
                     filters=[{"column": "status", "operator": "eq", "values": ["approved"]}],
                     aggregates=[{"name": "amount", "column": "amount", "operation": "sum", "decimals": 2}],
                     output_names=["summary.csv"])


@pytest.mark.parametrize("change", ["content", "alias"])
async def test_policy_is_bound_before_extraction_and_checked_after_verification(workspace, monkeypatch, change):
    alias = workspace / "rules.md"
    alias.symlink_to(workspace / "policy.md")
    roles = {}
    state = TableAcceptance("Use sales.csv and rules.md to write summary.csv.",
                            workspace=str(workspace), input_roles=roles)
    roles.update({"rules.md": "input", "summary.csv": "output"})
    calls = []

    async def completion(system, payload, max_tokens):
        calls.append(json.loads(payload))
        return policy_plan().model_dump_json()

    monkeypatch.setattr("rune.agent.requirement_gate._completion", completion)
    response = await state.requirements(str(workspace / "sales.csv"), None)
    contract = response.metadata["tableContract"]
    assert calls[0]["reference_documents"][0]["text"] == "Include approved rows only."
    assert contract["reference_sources"][0]["locator"] == str(alias)
    assert json.loads(response.output)["computed_preview"]["rows"] == [["10.00"]]
    assert (await state.verify(contract["id"], str(workspace / "summary.csv"), None, 1)).success
    assert await state.blocker() is None
    assert (await state.requirements(str(workspace / "sales.csv"), None)).metadata["tableContract"] == contract
    assert len(calls) == 1
    if change == "content":
        (workspace / "policy.md").write_text("Include every row.")
    else:
        (workspace / "other.md").write_bytes((workspace / "policy.md").read_bytes())
        alias.unlink()
        alias.symlink_to(workspace / "other.md")
    assert await state.blocker()
    assert state.snapshot()["status"] != "pass"
    with pytest.raises(ValueError, match="Reference documents changed"):
        await state.requirements(str(workspace / "sales.csv"), None)
    assert not (await state.verify(contract["id"], str(workspace / "summary.csv"), None, 1)).success
    assert len(calls) == 1


def test_only_declared_input_documents_are_read(workspace):
    roles = {"policy.md": "input", "brief.md": "output", "untouched.md": "preserve", "unrelated.md": "input"}
    docs = read_references("Read policy.md. Write brief.md, preserve untouched.md.", [], str(workspace), roles)
    assert [doc["path"] for doc in docs] == [str(workspace / "policy.md")]


@pytest.mark.parametrize("condition", ["missing", "oversized", "unreadable", "outside"])
async def test_inaccessible_policy_cannot_freeze_a_contract(workspace, condition):
    policy = workspace / "policy.md"
    if condition == "missing":
        policy.unlink()
    elif condition == "oversized":
        policy.write_text("x" * 16_001)
    elif condition == "unreadable":
        policy.write_bytes(b"\xff")
    else:
        policy.unlink()
        policy.symlink_to(workspace.parent / "outside.md")
    state = TableAcceptance("Use policy.md and sales.csv. Write summary.csv.",
                            workspace=str(workspace), input_roles={"policy.md": "input"})
    with pytest.raises((ValueError, UnicodeError)):
        await state.requirements(str(workspace / "sales.csv"), None)
    assert not state.contracts


async def test_sample_values_are_not_requirement_evidence(monkeypatch):
    async def completion(*args):
        return policy_plan().model_dump_json()

    monkeypatch.setattr("rune.agent.requirement_gate._completion", completion)
    with pytest.raises(ValueError, match="not quoted"):
        await extract_plan("Sum sales.csv.", [], "sales.csv", ["amount"],
                           [{"amount": "Include approved rows only."}])


async def test_unsupported_policy_has_no_misleading_preview(workspace, monkeypatch):
    async def extract(*args):
        return policy_plan().model_copy(update={"unverified": ["Required currency rates are unavailable"]})

    monkeypatch.setattr("rune.agent.table_acceptance.extract_plan", extract)
    result = await TableAcceptance("Sum sales.csv.").requirements(str(workspace / "sales.csv"), None)
    assert "computed_preview" not in json.loads(result.output)
    assert "No complete calculation" in json.loads(result.output)["next"]
