"""Final replies under run and shared budget limits."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rune.agent.request_budget import BudgetExceeded
from rune.agent.timing import capture_timing
from tests.unit.test_live_streaming import _TOOLS, _chunk, _make, _tc
from tests.unit.test_request_budget import owner

MODELS = ["openai/gpt-5.4", "openai/gpt-6-astra", "azure/gpt-6-astra", "xai/grok-4.7",
          "vertex_ai/gemini-2.5-flash", "anthropic/claude-opus-5", "ollama/qwen3"]


def has_work_tools(params):
    return "noop" in str(params.get("tools") or params.get("response_format") or "")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("budget", ["tokens", "cost"])
async def test_checked_work_gets_one_final_reply_without_another_tool_call(monkeypatch, model, limited, budget):
    report = {"type": "function", "function": {"name": "task_blocked", "parameters": {
        "type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"],
    }}}
    stream = _make(model, tools=[*_TOOLS, report])
    requests, executed = [], []
    monkeypatch.setattr("rune.llm.pricing.reserve_request_cost", lambda p, i, o: i * .0005 + o * .00005)
    monkeypatch.setattr("rune.llm.pricing.estimate_request_cost", lambda model, usage, request=None:
                        usage["input_tokens"] * .0005 + usage["output_tokens"] * .00005)
    monkeypatch.setattr("rune.agent.spend_budget.enabled", lambda: False)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda p: 400 if has_work_tools(p) else 100)

    async def complete(**params):
        requests.append(deepcopy(params))
        assert len(requests) <= 2
        async def response():
            if len(requests) == 1:
                yield _chunk(json.dumps({"tool": "noop", "arguments": {}}), finish="stop") if stream._guided else _chunk(tool_calls=_tc(), finish="tool_calls")
            else:
                text = "The recorded value is 42."
                yield _chunk(json.dumps({"final": text}) if stream._guided else text, finish="stop")
            yield SimpleNamespace(choices=[], usage={"prompt_tokens": 400 if has_work_tools(params) else 100, "completion_tokens": 10})
        return response()

    async def execute(calls):
        executed.extend(calls)
        stream._action_ok = True
        stream._messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "Checked result: 42"})

    monkeypatch.setattr("litellm.acompletion", complete)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    limit = owner(tokens=700 if limited and budget == "tokens" else 10000,
                  cost=.30 if limited and budget == "cost" else None)
    with capture_timing(limit) as run:
        text = "".join([part async for part in stream.stream_text()])
        assert run["usage"]["calls"] == 2 and not run["budget_blocked"]
        assert not run["reservations"]
        if limit._config.cost_budget_usd is not None:
            assert run["usage"]["cost_usd"] <= limit._config.cost_budget_usd
    assert text == "The recorded value is 42."
    assert len(executed) == 1 and not stream.tool_budget_exhausted
    assert has_work_tools(requests[-1]) is not limited
    assert "Checked result: 42" in str(requests[-1]["messages"])
    assert stream.answer_reserve_reason == ("budget" if limited else "")
    if limited:
        assert "task_blocked" in str(requests[-1].get("tools") or requests[-1].get("response_format"))
    if not limited:
        assert requests[0].get("tools") == requests[1].get("tools")


@pytest.mark.parametrize("pending", ["verification", "recovery", "edited", "postcondition", "artifact", "removed", "repeated_removal"])
async def test_unfinished_checks_cannot_be_replaced_with_a_final_reply(monkeypatch, pending):
    from rune.agent.verification_state import VerificationState
    state = VerificationState()
    stream = _make("openai/gpt-5.4")
    stream._verification_state = lambda: state
    stream._require_verification = pending == "verification"
    stream._tool_recovery = lambda: {"noop"} if pending == "recovery" and state.pending else None
    calls = []
    monkeypatch.setattr("rune.agent.spend_budget.enabled", lambda: False)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda p: 400 if has_work_tools(p) else 100)

    async def complete(**params):
        calls.append(params)
        async def response():
            yield _chunk(tool_calls=_tc(), finish="tool_calls")
            yield SimpleNamespace(choices=[], usage={"prompt_tokens": 400, "completion_tokens": 10})
        return response()

    async def execute(batch):
        state.changed()
        stream._vos_edited = pending == "edited"
        if pending == "removed":
            stream._reobs_dirs.add("uninspected-directory")
        if pending == "repeated_removal":
            stream._reobs_removed["uninspected-directory"] = 5
        if pending == "postcondition":
            monkeypatch.setattr(stream, "_unmet_postconditions", lambda: ["Missing output"])
        if pending == "artifact":
            monkeypatch.setattr(stream, "_unresolved_artifacts", lambda: ["Missing input"])
        stream._messages.append({"role": "tool", "tool_call_id": batch[0]["id"], "content": "Written; not checked"})

    monkeypatch.setattr("litellm.acompletion", complete)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    with capture_timing(owner(tokens=700)), pytest.raises(BudgetExceeded):
        _ = [part async for part in stream.stream_text()]
    assert len(calls) == 1 and not stream.answer_reserve_reason
    assert state.pending


@pytest.mark.parametrize("block", ["cost", "calls", "unknown", "parent", "closed"])
def test_preview_cannot_clear_limits_or_reserve_budget(block):
    from rune.agent.timing import request_fits
    parent = owner(tokens=0 if block == "parent" else 10000)
    with capture_timing(parent), capture_timing(owner(tokens=1000, calls=0 if block == "calls" else 10,
                                                      cost=0 if block == "cost" else 1)) as run:
        if block == "unknown":
            run["unpriced_finished"] = True
        if block == "closed":
            run["closed"] = True
        before = deepcopy({k:run[k] for k in ("usage", "reservations", "cost_reservations", "budget_blocked")})
        assert not request_fits({"model":"openai/gpt-5.4", "messages":[], "max_tokens":32})
        assert {k:run[k] for k in before} == before


@pytest.mark.parametrize("reply", ["tool", "truncated", "empty", "shared_limit"])
async def test_budget_reply_cannot_restart_work_or_bypass_shared_limits(monkeypatch, reply):
    stream = _make("openai/gpt-5.4")
    requests, executed = [], []
    monkeypatch.setattr("rune.agent.spend_budget.enabled", lambda: False)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda p: 400 if has_work_tools(p) else 100)

    def shared_reserve(params, estimated):
        if requests and reply == "shared_limit":
            raise BudgetExceeded("The shared model budget cannot cover this request.")
        return params, None
    monkeypatch.setattr("rune.agent.spend_budget.reserve", shared_reserve)

    async def complete(**params):
        requests.append(params)
        async def response():
            if len(requests) == 1 or reply == "tool":
                yield _chunk(tool_calls=_tc(), finish="tool_calls")
            else:
                yield _chunk("Partial" if reply == "truncated" else "", finish="length" if reply == "truncated" else "stop")
            yield SimpleNamespace(choices=[], usage={"prompt_tokens":400 if has_work_tools(params) else 100, "completion_tokens":10})
        return response()

    async def execute(batch):
        executed.extend(batch)
        stream._messages.append({"role":"tool", "tool_call_id":batch[0]["id"], "content":"Result recorded"})

    monkeypatch.setattr("litellm.acompletion", complete)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    with capture_timing(owner(tokens=700)):
        if reply == "shared_limit":
            with pytest.raises(BudgetExceeded, match="shared"):
                _ = [part async for part in stream.stream_text()]
        else:
            _ = [part async for part in stream.stream_text()]
            assert stream.tool_budget_exhausted
    assert len(executed) == 1
    assert len(requests) == (1 if reply == "shared_limit" else 2)


async def test_compatibility_retry_keeps_the_final_reply_and_evidence(monkeypatch):
    import litellm

    stream = _make("openai/budget-compatibility-test")
    requests = []
    monkeypatch.setattr("rune.agent.spend_budget.enabled", lambda: False)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda p: 400 if has_work_tools(p) else 100)

    async def complete(**params):
        requests.append(deepcopy(params))
        assert len(requests) <= 3
        if len(requests) == 2:
            raise litellm.BadRequestError("Unsupported value: temperature does not support 0", model=params["model"], llm_provider="openai")
        async def response():
            yield _chunk(tool_calls=_tc(), finish="tool_calls") if len(requests) == 1 else _chunk("Checked: 42", finish="stop")
            yield SimpleNamespace(choices=[], usage={"prompt_tokens":400 if has_work_tools(params) else 100, "completion_tokens":10})
        return response()

    async def execute(batch):
        stream._messages.append({"role":"tool", "tool_call_id":batch[0]["id"], "content":"Checked: 42"})

    monkeypatch.setattr(litellm, "acompletion", complete)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    with capture_timing(owner(tokens=700)) as run:
        assert "".join([part async for part in stream.stream_text()]) == "Checked: 42"
        assert run["usage"]["calls"] == 3 and not run["budget_blocked"] and not run["reservations"]
    assert len(requests) == 3 and all(not has_work_tools(p) for p in requests[1:])
    assert requests[1]["messages"] == requests[2]["messages"]
    assert "Checked: 42" in str(requests[2]["messages"])
    assert "temperature" not in requests[2]
