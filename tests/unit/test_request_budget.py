"""Budget checks cover inner rounds, parallel requests, retries and missing usage."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.request_budget import BudgetExceeded
from rune.agent.timing import capture_timing, timed_completion
from tests.unit.test_live_streaming import _chunk, _make, _tc


def owner(tokens=1000, calls=10, cost=None):
    return SimpleNamespace(_token_budget=SimpleNamespace(total=tokens, used=0),
                           _config=SimpleNamespace(model_request_limit=calls, cost_budget_usd=cost))


def test_input_estimate_accepts_schema_fields_named_type():
    from rune.agent.request_budget import input_estimate

    schema = {"type": "object", "properties": {"type": {"type": "string"}, "value": {"type": ["string", "null"]}}}
    assert input_estimate({"messages": [], "tools": [{"type": "function", "function": {"parameters": schema}}]}) > 0
    assert input_estimate({"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:private"}}]}]}) > 4000


async def test_stream_rounds_stop_before_the_next_request(monkeypatch):
    async def completion(**kwargs):
        async def chunks():
            yield _chunk(tool_calls=_tc(), finish="tool_calls")
            yield SimpleNamespace(choices=[], usage={"prompt_tokens": 70, "completion_tokens": 30})
        return chunks()

    request = AsyncMock(side_effect=completion)
    monkeypatch.setattr("litellm.acompletion", request)
    stream = _make("openai/gpt-5.4")
    stream._request_tokens_limit = 100
    monkeypatch.setattr(stream, "_execute_tool_batch", AsyncMock())
    with pytest.raises(BudgetExceeded):
        async for _ in stream.stream_text():
            pass
    assert request.await_count == 1


@pytest.mark.parametrize("field", ["request_tokens_limit", "response_tokens_limit"])
async def test_zero_stream_limit_never_sends_a_request(monkeypatch, field):
    from rune.agent.litellm_adapter import LiteLLMAgent, UsageLimits

    request = AsyncMock()
    monkeypatch.setattr("litellm.acompletion", request)
    agent = LiteLLMAgent(model="openai/gpt-5.4", tools=[], system_prompt="test")
    limits = UsageLimits(**{field: 0})
    with pytest.raises(BudgetExceeded):
        async with agent.run_stream("test", usage_limits=limits) as stream:
            async for _ in stream.stream_text():
                pass
    request.assert_not_called()


async def test_parallel_requests_reserve_budget_before_await(monkeypatch):
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 100)
    started, finish = asyncio.Event(), asyncio.Event()
    async def completion(**kwargs):
        started.set()
        await finish.wait()
        return {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    runner = owner(tokens=300)
    with capture_timing(runner):
        first = asyncio.create_task(timed_completion(completion, {"model": "test", "messages": [], "max_tokens": 200}))
        await started.wait()
        with pytest.raises(BudgetExceeded):
            await timed_completion(completion, {"model": "test", "messages": []})
        finish.set()
        await first
        assert runner._token_budget.used == 150


async def test_delegated_run_shares_parent_limits_and_accounts_usage_once():
    parent, child = owner(tokens=10000, calls=1), owner(tokens=10000, calls=10)
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    with capture_timing(parent) as outer:
        with capture_timing(child) as inner:
            await timed_completion(request, {"model": "xai/grok-4.6", "messages": [], "max_tokens": 100})
            assert parent._token_budget.used == child._token_budget.used == 110
            assert outer["usage"]["calls"] == inner["usage"]["calls"] == 1
            assert outer["usage"]["cost_usd"] == inner["usage"]["cost_usd"] > 0
            with pytest.raises(BudgetExceeded, match="request limit"):
                await timed_completion(request, {"model": "xai/grok-4.6", "messages": []})
        with pytest.raises(BudgetExceeded):
            await timed_completion(request, {"model": "xai/grok-4.6", "messages": []})
    request.assert_awaited_once()


async def test_detached_child_cannot_bill_an_ended_parent():
    ready, proceed = asyncio.Event(), asyncio.Event()
    request = AsyncMock()
    async def child():
        with capture_timing(owner()):
            ready.set()
            await proceed.wait()
            with pytest.raises(BudgetExceeded, match="parent run ended"):
                await timed_completion(request, {"model": "test", "messages": []})
    with capture_timing(owner()):
        task = asyncio.create_task(child())
        await ready.wait()
    proceed.set()
    await task
    request.assert_not_called()


async def test_missing_usage_keeps_reservation_and_request_limit_counts_failed_attempts(monkeypatch):
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 100)
    runner = owner(tokens=300)
    request = AsyncMock(return_value={})
    with capture_timing(runner):
        await timed_completion(request, {"model": "test", "messages": [], "max_tokens": 200})
        with pytest.raises(BudgetExceeded):
            await timed_completion(request, {"model": "test", "messages": []})
    assert request.await_count == 1 and runner._token_budget.used == 0
    import httpx
    request = AsyncMock(side_effect=httpx.ConnectTimeout("offline"))
    with capture_timing(owner(calls=1)):
        with pytest.raises(httpx.ConnectTimeout):
            await timed_completion(request, {"model": "test", "messages": []})
        with pytest.raises(BudgetExceeded, match="request limit"):
            await timed_completion(request, {"model": "test", "messages": []})
    assert request.await_count == 1


@pytest.mark.parametrize("known", [True, False])
async def test_recorded_cost_or_unknown_cost_stops_further_requests(known):
    model = "xai/grok-4.6" if known else "unknown"
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 1000, "completion_tokens": 100}})
    with capture_timing(owner(tokens=10000, cost=.001)):
        if known:
            await timed_completion(request, {"model": model, "messages": [], "max_tokens": 100})
        with pytest.raises(BudgetExceeded, match="cost"):
            await timed_completion(request, {"model": model, "messages": []})
    assert request.await_count == int(known)


async def test_budget_stop_does_not_restart_or_escalate_goal(tmp_path):
    from rune.agent.goal_loop import GoalLoop, GoalLoopConfig, GoalSpec
    from rune.types import CompletionTrace

    request = AsyncMock(return_value=CompletionTrace(reason="request_budget_exhausted"))
    escalate = AsyncMock()
    loop = GoalLoop(GoalLoopConfig(), run_fn=request, escalate_fn=escalate, workspace=tmp_path)
    result = await loop.run(GoalSpec(goal="Create a report"))
    assert result.stop_cause == "budget" and not result.success
    request.assert_awaited_once()
    escalate.assert_not_called()


async def test_partial_stream_usage_keeps_unreported_allowance(monkeypatch):
    from rune.agent.timing import current_usage
    from rune.llm.pricing import usage_payload

    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 100)
    async def completion(**kwargs):
        async def chunks():
            yield {"usage": {"prompt_tokens": 100, "completion_tokens": 10}}
            raise TimeoutError("stream interrupted")
        return chunks()

    runner = owner(tokens=300)
    with capture_timing(runner):
        stream = await timed_completion(completion, {"model": "xai/grok-4.6", "stream": True,
                                                     "messages": [], "max_tokens": 200})
        with pytest.raises(TimeoutError):
            async for _ in stream:
                pass
        assert runner._token_budget.used == 110
        assert current_usage()["incomplete_calls"] == 1
        cost = usage_payload(SimpleNamespace(timings={})) ["cost"]
        assert cost["usd"] is None and cost["knownUsd"] > 0 and cost["incomplete"]
        with pytest.raises(BudgetExceeded):
            await timed_completion(completion, {"model": "test", "messages": []})


async def test_zero_native_budget_applies_before_classification(monkeypatch, tmp_path):
    from rune.agent.loop import NativeAgentLoop
    from rune.api.trust import build_trust_payload
    from rune.types import AgentConfig

    request = AsyncMock()
    monkeypatch.setattr("litellm.acompletion", request)
    agent = NativeAgentLoop(AgentConfig(token_budget_override=0))
    trace = await agent.run("Create report.docx", context={"workspace_root": str(tmp_path)})
    assert trace.reason == "request_budget_exhausted"
    assert trace.total_tokens_used == 0
    request.assert_not_called()
    trust = build_trust_payload(trace)
    assert trust["budgetExhausted"] and trust["completionCheck"]


async def test_early_exit_trace_keeps_billed_auxiliary_usage():
    from rune.agent.timing import timed_run
    from rune.types import CompletionTrace

    class Runner:
        def __init__(self):
            self._token_budget = owner()._token_budget

        @timed_run
        async def run(self):
            request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
            await timed_completion(request, {"model": "test", "messages": [], "max_tokens": 100})
            return CompletionTrace(reason="cancelled")

    trace = await Runner().run()
    assert trace.total_tokens_used == trace.timings["usage"]["total_tokens"] == 110


@pytest.mark.parametrize("caller", ["advisor", "fastpath", "path_oracle"])
async def test_auxiliary_calls_share_the_run_limit_and_usage(monkeypatch, caller):
    from rune.agent.advisor.protocol import AdvisorRequest
    from rune.agent.advisor.service import AdvisorConfig, AdvisorService
    from rune.agent.fastpath import _complete
    from rune.agent.path_intent_oracle import _call_path_intent_oracle

    request = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"requested_paths": [], "confidence": 1}'))],
        usage={"prompt_tokens": 100, "completion_tokens": 10},
    ))
    monkeypatch.setattr("litellm.acompletion", request)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    monkeypatch.setattr("rune.agent.litellm_adapter._resolve_litellm_model", lambda _: ("xai/grok-4.6", {}))
    async def invoke():
        try:
            if caller == "advisor":
                service = AdvisorService(AdvisorConfig(enabled=True, provider="xai", model="grok-4.6", timeout_ms=1000))
                await service._invoke_llm(AdvisorRequest(
                    trigger="stuck", goal="test", classification_summary="", activity_phase="execution",
                    step=1, token_budget_frac=.5, evidence_snapshot={}, gate_state=None,
                    stall_state={}, recent_messages=[], files_written=[],
                ))
            elif caller == "fastpath":
                await _complete("grok-4.6", "xai", "test", n=2)
            else:
                await _call_path_intent_oracle("test", ["/tmp"], "xai/grok-4.6", 0.8)
        except BudgetExceeded:
            pass

    with capture_timing(owner(tokens=10000, calls=1)) as measured:
        await invoke()
        await invoke()
    assert request.await_count == 1
    assert measured["usage"]["calls"] == 1
    assert measured["usage"]["total_tokens"] == 110
    assert measured["usage"]["cost_usd"] > 0
    assert request.call_args.kwargs["model"] == "xai/grok-4.6"


@pytest.mark.parametrize("consume", [False, True])
async def test_closing_an_unfinished_stream_closes_the_provider(monkeypatch, consume):
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    response = SimpleNamespace(aclose=AsyncMock())
    class Response:
        def __aiter__(self):
            return self
        async def __anext__(self):
            return {"usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        async def aclose(self):
            await response.aclose()
    with capture_timing(owner(tokens=10000, cost=.1)) as measured:
        stream = await timed_completion(AsyncMock(return_value=Response()),
                                       {"model": "xai/grok-4.6", "messages": [], "stream": True})
        if consume:
            await anext(stream)
        await stream.aclose()
        await stream.aclose()
        response.aclose.assert_awaited_once()
        assert measured["unpriced_finished"]
        with pytest.raises(BudgetExceeded):
            await timed_completion(AsyncMock(), {"model": "xai/grok-4.6", "messages": []})


@pytest.mark.parametrize("cancel", [False, True])
async def test_agent_stream_exit_closes_the_provider(monkeypatch, cancel):
    from rune.agent.litellm_adapter import LiteLLMAgent

    entered = asyncio.Event()
    closed = AsyncMock()
    class Response:
        def __aiter__(self):
            return self
        async def __anext__(self):
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            return _chunk("Partial answer")
        async def aclose(self):
            await closed()

    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=Response()))
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    agent = LiteLLMAgent(model="xai/grok-4.6", tools=[], system_prompt="test")
    async def consume():
        async with agent.run_stream("test") as stream:
            async for _ in stream.stream_text():
                break

    with capture_timing(owner(tokens=10000, cost=.1)) as measured:
        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), timeout=2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        closed.assert_awaited_once()
        assert measured["unpriced_finished"]
        assert measured["usage"]["calls"] == 1


@pytest.mark.parametrize("wave", [False, True])
async def test_isolated_workers_cannot_bypass_parent_budget(monkeypatch, wave):
    from rune.agent.parallel_isolated import WorkerSpec, run_isolated_worker, run_wave_and_merge

    create = AsyncMock()
    spawn = AsyncMock()
    monkeypatch.setattr("rune.agent.worktree.create", create)
    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    spec = WorkerSpec("test", "Create a report")
    with capture_timing(owner()) as measured:
        with pytest.raises(BudgetExceeded, match="shared run budget"):
            if wave:
                await run_wave_and_merge("/unused", [spec])
            else:
                await run_isolated_worker("/unused", spec)
        assert measured["budget_blocked"]
    create.assert_not_called()
    spawn.assert_not_called()
