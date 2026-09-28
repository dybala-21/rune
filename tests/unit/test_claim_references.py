"""Review references must resolve to actual, unmasked answer lines."""

import json
from unittest.mock import AsyncMock

import pytest

from rune.agent.test_claims import TestClaimGate as ClaimGate
from rune.agent.test_claims import answer_reference, comparison_evidence
from rune.agent.test_summary import recorded_tables
from tests.unit import test_test_evidence

state = test_test_evidence.state


@pytest.mark.parametrize("line", [True, 0, 2, 4, "3"])
def test_reference_rejects_invalid_or_blank_lines(line):
    with pytest.raises(ValueError, match="unavailable"):
        answer_reference({"source_line": line}, "First.\n\nThird.")


@pytest.mark.parametrize("count,blocked", [(3, False), (2, True)])
async def test_sparse_line_references_still_check_actual_metrics(monkeypatch, state, count, blocked):
    client = AsyncMock()
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({
        "claims": [{"source_line": 3, "run": 0, "test": -1, "metric": "failure_events", "value": count}],
        "explanation_issues": [],
    })}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    note = await ClaimGate().review(state, f"Summary\n\nThere were **{count} failure events** before the fix.")
    assert bool(note) == blocked
    schema = client.completion.call_args.kwargs["response_format"]["json_schema"]["schema"]["properties"]
    assert "quote" not in schema["claims"]["items"]["properties"]
    assert schema["claims"]["items"]["properties"]["source_line"]["enum"] == [1, 3]
    client.completion.assert_awaited_once()


async def test_masked_table_cannot_supply_a_new_claim(monkeypatch, state):
    table = recorded_tables(comparison_evidence(state))[0]
    client = AsyncMock()
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({
        "claims": [{"source_line": 2, "run": 0, "test": -1, "metric": "failure_events", "value": 3}],
        "explanation_issues": [],
    })}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    assert "could not be checked" in await ClaimGate().review(state, table + "\n\nChanged the code.")


async def test_explanation_uses_literal_answer_and_observation(monkeypatch, state):
    client = AsyncMock()
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({
        "claims": [], "explanation_issues": [{"source_line": 3, "evidence_id": "observation_1",
            "evidence_lines": [1], "reason": "Zero stays zero.", "needs_correction": True}],
    })}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    gate = ClaimGate(_observations=[{"id": 1, "text": "return value / 2"}])
    note = await gate.review(state, "Summary\n\nThe value **always decreases**.")
    assert "always decreases" in note and "Zero stays zero" in note
    client.completion.assert_awaited_once()
