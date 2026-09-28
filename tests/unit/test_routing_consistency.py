"""Reject incompatible execution requirements without excluding mixed work."""

from unittest.mock import AsyncMock

import pytest

from rune.agent.classification_response import InvalidClassification, validate_decision
from rune.agent.goal_classifier import classify_goal
from tests.unit.test_goal_classifier import response, verdict


@pytest.mark.parametrize("changes", [
    {"goal_type": "chat", "intent_categories": ["desktop"]},
    {"goal_type": "web", "requires_desktop_input": True},
    {"goal_type": "full", "requires_desktop_input": True},
    {"goal_type": "chat", "table_output": "csv"},
    {"goal_type": "web", "table_output": "xlsx"},
    {"goal_type": "code_modify", "calculation_expression": "1 + 2"},
])
def test_contradictory_requirements_are_not_accepted(changes):
    with pytest.raises(InvalidClassification, match="inconsistent"):
        validate_decision(verdict(**changes))


@pytest.mark.parametrize("changes", [
    {"goal_type": "code_modify", "table_output": "csv", "intent_categories": ["table", "calculation"]},
    {"goal_type": "full", "intent_categories": ["desktop", "table"], "requires_desktop_input": True,
     "requires_execution": True, "table_output": "xlsx"},
    {"goal_type": "execution", "requires_execution": True, "table_output": "csv"},
    {"goal_type": "research", "intent_categories": ["document"]},
    {"goal_type": "browser", "table_output": "csv"},
    {"goal_type": "chat", "calculation_expression": "1 + 2"},
])
def test_legitimate_mixed_requirements_remain_valid(changes):
    data = verdict(**changes)
    assert validate_decision(data) == data


async def test_connected_model_can_correct_conflicting_fields_once(monkeypatch):
    client = AsyncMock()
    client.completion.side_effect = [
        response(verdict(goal_type="chat", table_output="csv")),
        response(verdict(goal_type="code_modify", table_output="csv", intent_categories=["table", "calculation"])),
    ]
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    result = await classify_goal("Aggregate source.csv into summary.csv")
    assert result.available and result.goal_type == "code_modify"
    assert client.completion.await_count == 2
    assert "inconsistent" in client.completion.call_args.kwargs["messages"][0]["content"]
