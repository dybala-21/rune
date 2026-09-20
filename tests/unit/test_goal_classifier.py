"""Routing decisions must be complete before tools can be selected."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from rune.agent.classification_response import RESPONSE_FORMAT
from rune.agent.goal_classifier import classify_goal, from_wire, to_wire


def verdict(**changes):
    return {"goal_type": "full", "confidence": 0.9, "reason": "requested work",
            "requires_execution": False, "intent_categories": [],
            "requires_desktop_input": False, "is_related_to_previous": False,
            "table_output": "none", "calculation_expression": "", **changes}


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


async def test_calculation_extraction_reuses_the_routing_call(monkeypatch):
    from rune.agent.calculation import calculation_context

    stub = client(monkeypatch, response(verdict(goal_type="chat", calculation_expression="0.1 + 0.2")))
    result = await classify_goal("Evaluate 0.1 + 0.2")
    assert '"result": "0.3"' in calculation_context("Evaluate 0.1 + 0.2", result)
    assert stub.completion.await_count == 1
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


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (403, 1), (400, 1), (429, 2), (503, 2)])
async def test_only_transient_rejections_retry_within_the_same_budget(monkeypatch, status, expected_calls):
    import httpx

    reply = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"),
                           headers={"Retry-After": "0"})
    error = httpx.HTTPStatusError("private provider details", request=reply.request, response=reply)
    stub = client(monkeypatch, error, response(verdict(goal_type="web")))
    result = await classify_goal("Find the source")
    assert stub.completion.await_count == expected_calls
    assert result.available == (expected_calls == 2)
    requests = [call.kwargs for call in stub.completion.call_args_list]
    assert all(call["max_retries"] == 0 for call in requests)
    assert 30 < requests[0]["timeout"] <= 35
    if expected_calls == 2:
        assert requests[1]["timeout"] <= requests[0]["timeout"]


async def test_provider_retry_after_cannot_extend_routing_deadline(monkeypatch):
    import httpx

    reply = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"),
                           headers={"Retry-After": "120"})
    stub = client(monkeypatch, httpx.HTTPStatusError("limited", request=reply.request, response=reply))
    result = await classify_goal("Read the page")
    assert not result.available and stub.completion.await_count == 1


@pytest.mark.parametrize("failure", [TimeoutError, httpx.ReadTimeout])
async def test_ambiguous_timeout_is_not_sent_again(monkeypatch, failure):
    stub = client(monkeypatch, failure("secret response body"))
    result = await classify_goal("Save the file")
    assert not result.available and stub.completion.await_count == 1
    assert "secret" not in result.reason and "timeout" in result.reason


async def test_deadline_cancels_one_request_without_running_another(monkeypatch):
    import asyncio

    monkeypatch.setattr("rune.agent.classification_response.ROUTING_TIMEOUT", 0.02)
    cancelled = []

    async def wait(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    stub = client(monkeypatch)
    stub.completion.side_effect = wait
    result = await classify_goal("Open the app")
    assert not result.available and cancelled == [True]
    assert stub.completion.await_count == 1


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


@pytest.mark.parametrize("provider,model,effort", [
    ("xai", "grok-4.6", "low"), ("gemini", "gemini-2.5-flash", "none"),
    ("anthropic", "claude-opus-5", "low"), ("openai", "gpt-6-astra", "low"),
    ("xai", "grok-4.3", "none"), ("xai", "unknown-grok", None),
])
async def test_routing_uses_supported_small_effort_without_changing_preferences(monkeypatch, provider, model, effort):
    from types import SimpleNamespace

    from rune.config import get_config

    cfg = get_config()
    original = dict(cfg.llm.reasoning_efforts)
    monkeypatch.setattr("rune.llm.model_selection.get_effective_model_selection", lambda: SimpleNamespace(
        provider=provider, model=model))
    stub = client(monkeypatch, response(verdict()))
    assert (await classify_goal("Read the source file")).available
    assert stub.completion.call_args.kwargs.get("reasoning_effort") == effort
    assert cfg.llm.reasoning_efforts == original


async def test_failed_routing_clears_prior_domain_and_never_selects_tools(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    monkeypatch.setenv('RUNE_HOME', str(tmp_path))
    loop = NativeAgentLoop()
    loop._last_goal_type = 'code_modify'
    select = Mock()
    monkeypatch.setattr(loop, '_select_tools', select)
    bad = ClassificationResult(goal_type='full', confidence=.5, tier=2,
                               available=False, reason='Classification unavailable: read_timeout')
    trace = await loop.run('Open the app', classification=bad)
    assert 'read_timeout' in trace.reason
    assert loop._last_goal_type == ''
    select.assert_not_called()
