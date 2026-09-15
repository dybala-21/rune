"""Measure model requests through stream completion and isolate concurrent runs."""

import asyncio
from types import SimpleNamespace

import pytest

from rune.agent.timing import timed_completion, timed_run, timing_phase
from rune.types import CompletionTrace


async def test_stream_timing_includes_generation_and_preserves_events(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("rune.agent.timing.time.monotonic", lambda: now[0])
    events = [SimpleNamespace(choices=[]), SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))])]

    async def complete(**kwargs):
        now[0] += .2

        async def stream():
            for event in events:
                now[0] += .3
                yield event
        return stream()

    @timed_run
    async def run():
        with timing_phase("execution"):
            stream = await timed_completion(complete, {"model": "test", "stream": True,
                "messages": [{"content": "private"}], "api_key": "secret", "reasoning_effort": "max"})
            assert [event async for event in stream] == events
        return CompletionTrace(reason="completed")

    trace = await run()
    request, = [row for row in trace.timings["spans"] if row["kind"] == "model"]
    assert request["durationMs"] == 800
    assert request["firstEventMs"] == 500 and request["firstTextMs"] == 800
    assert request["phase"] == "execution" and request["reasoningEffort"] == "max"
    assert "secret" not in str(trace.timings) and "private" not in str(trace.timings)


async def test_concurrent_runs_keep_model_events_separate():
    @timed_run
    async def run(model):
        async def complete(**kwargs):
            await asyncio.sleep(0)
            return {"response": model}
        result = await timed_completion(complete, {"model": model})
        assert result["response"] == model
        return CompletionTrace(reason="completed")

    traces = await asyncio.gather(run("first"), run("second"))
    for name, trace in zip(("first", "second"), traces, strict=True):
        assert [row["model"] for row in trace.timings["spans"]] == [name]


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_failure_is_recorded_and_propagated(streaming):
    async def complete(**kwargs):
        if not streaming:
            raise ConnectionError("provider unavailable")

        async def stream():
            yield SimpleNamespace(choices=[])
            raise ConnectionError("stream disconnected")
        return stream()

    @timed_run
    async def run():
        with pytest.raises(ConnectionError):
            response = await timed_completion(complete, {"model": "test", "stream": streaming})
            if streaming:
                async for _ in response:
                    pass
        return CompletionTrace(reason="error")

    trace = await run()
    assert trace.timings["spans"][0]["status"] == "interrupted"
    assert "durationMs" in trace.timings["spans"][0]


async def test_auxiliary_and_stream_usage_are_counted_once_and_missing_usage_is_visible():
    class Runner:
        def __init__(self):
            self._token_budget = SimpleNamespace(used=0)

        @timed_run
        async def run(self):
            async def auxiliary(**_):
                return {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}

            async def completion(**_):
                async def stream():
                    for _ in range(2):
                        yield SimpleNamespace(choices=[], usage={
                            "prompt_tokens": 1000, "completion_tokens": 50,
                            "prompt_tokens_details": {"cached_tokens": 600, "cache_creation_tokens": 300},
                            "completion_tokens_details": {"reasoning_tokens": 30}})
                return stream()

            async def missing(**_):
                return {"choices": []}

            with timing_phase("classification"):
                await timed_completion(auxiliary, {"model": "router"})
            assert self._token_budget.used == 120
            stream = await timed_completion(completion, {"model": "main", "stream": True})
            assert len([chunk async for chunk in stream]) == 2
            await timed_completion(missing, {"model": "main"})
            assert self._token_budget.used == 120  # The main loop accounts for streamed tokens.
            return CompletionTrace(reason="completed")

    trace = await Runner().run()
    usage = trace.timings["usage"]
    assert usage["calls"] == 3 and usage["reported_calls"] == 2
    assert usage["total_tokens"] == 1170 and usage["reasoning_tokens"] == 30
    assert usage["cache_write_tokens"] == 300 and usage["cache_write_unreported_calls"] == 1
    assert usage["by_model"]["router"]["total_tokens"] == 120
    assert usage["by_model"]["main"]["total_tokens"] == 1050
