"""Routing decisions must be complete before tools can be selected."""

import json
from unittest.mock import AsyncMock

import pytest

from rune.agent.classification_response import RESPONSE_FORMAT
from rune.agent.goal_classifier import classify_goal, from_wire, to_wire


def verdict(**changes):
    return {"goal_type": "full", "confidence": 0.9, "reason": "requested work",
            "requires_execution": False, "intent_categories": [],
            "requires_desktop_input": False, "is_related_to_previous": False,
            "table_output": "none", **changes}


def response(data, finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": {
        "content": data if isinstance(data, str) else json.dumps(data),
    }}]}


def client(monkeypatch, *responses):
    stub = AsyncMock()
    stub.completion.side_effect = responses
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: stub)
    return stub


async def test_valid_routing_needs_one_constrained_call(monkeypatch):
    stub = client(monkeypatch, response(verdict(goal_type="code_modify", requires_execution=True)))
    result = await classify_goal("Fix the test and summarize the result")
    assert result.available and result.requires_execution and result.is_complex_coding
    assert stub.completion.await_count == 1
    assert stub.completion.call_args.kwargs["response_format"] == RESPONSE_FORMAT
    assert from_wire(to_wire(result)) == result


async def test_extra_prose_is_rejected_and_retried_once(monkeypatch):
    malformed = '```json\n' + json.dumps(verdict()) + '\n```\nI will now implement the request.'
    stub = client(monkeypatch, response(malformed), response(verdict()))
    result = await classify_goal("Fix amounts.py and show before/after tests")
    assert result.available and stub.completion.await_count == 2
    assert "model" in stub.completion.call_args.kwargs


@pytest.mark.parametrize("change", [
    {"intent_categories": None}, {"requires_execution": "false"},
    {"requires_desktop_input": "false"}, {"confidence": float("nan")},
    {"goal_type": "invented"}, {"intent_categories": ["unknown"]},
])
async def test_invalid_decision_cannot_enable_tools(monkeypatch, change):
    bad = response(verdict(**change))
    stub = client(monkeypatch, bad, bad)
    result = await classify_goal("Operate the selected app")
    assert not result.available and stub.completion.await_count == 2
    assert from_wire(to_wire(result)).available is False


async def test_missing_field_is_not_silently_defaulted(monkeypatch):
    data = verdict()
    del data["requires_desktop_input"]
    client(monkeypatch, response(data), response(data))
    assert not (await classify_goal("Use the app")).available


@pytest.mark.parametrize("finish", ["length", "max_tokens", "content_filter"])
async def test_truncated_or_filtered_response_is_not_accepted(monkeypatch, finish):
    client(monkeypatch, response(verdict(), finish), response(verdict(), finish))
    assert not (await classify_goal("Summarize this")).available


@pytest.mark.parametrize("output,expected", [("none", False), ("csv", True), ("xlsx", True)])
async def test_table_gate_uses_deliverable_not_response_layout(monkeypatch, output, expected):
    client(monkeypatch, response(verdict(intent_categories=["table"], table_output=output)))
    result = await classify_goal("Summarize the result")
    assert result.available and ("table" in result.intent_categories) is expected


@pytest.mark.parametrize("needs_input", [True, False])
async def test_desktop_intent_keeps_explicit_input_requirement(monkeypatch, needs_input):
    client(monkeypatch, response(verdict(intent_categories=["desktop"], requires_desktop_input=needs_input)))
    result = await classify_goal("Inspect the selected app")
    assert result.requires_desktop_input is needs_input
    assert not result.requires_execution


async def test_unrelated_followup_does_not_inherit_desktop(monkeypatch):
    client(monkeypatch, response(verdict(goal_type="chat")))
    result = await classify_goal("Summarize these notes", previous_goal="Use TextEdit", previous_goal_type="full")
    assert result.is_domain_change and not result.intent_categories


async def test_provider_error_fails_closed_without_unbounded_retry(monkeypatch):
    stub = client(monkeypatch, RuntimeError("provider unavailable"))
    result = await classify_goal("Save a document")
    assert not result.available and stub.completion.await_count == 1
    assert "RuntimeError" in result.reason


async def test_routing_uses_selected_model_without_a_weaker_intermediate(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("rune.llm.model_selection.get_effective_model_selection", lambda: SimpleNamespace(
        provider="anthropic", model="claude-opus-5"))
    stub = client(monkeypatch, response(verdict(intent_categories=["desktop"], requires_desktop_input=True)))
    result = await classify_goal("Use Calculator to compute the answer")
    assert result.available and result.requires_desktop_input
    assert stub.completion.await_count == 1
    assert stub.completion.call_args.kwargs["model"] == "claude-opus-5"
    assert stub.completion.call_args.kwargs["provider"] == "anthropic"
