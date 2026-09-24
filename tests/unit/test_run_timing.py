"""Measure model requests through stream completion and isolate concurrent runs."""

import asyncio
from types import SimpleNamespace

import pytest

from rune.agent.timing import timed_completion, timed_run, timing_phase
from rune.types import CompletionTrace


async def test_catalog_fingerprint_detects_changes_without_recording_definitions():
    from rune.agent.timing import capture_timing, timing_snapshot

    async def complete(**kwargs):
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 1}}

    tool = {"type": "function", "function": {"name": "read", "description": "private description"}}
    with capture_timing() as run:
        for tools in ([tool], [tool], [tool, {"type": "function", "function": {"name": "write"}}]):
            await timed_completion(complete, {"model": "test", "tools": tools})
    snapshot = timing_snapshot(run)
    rows = snapshot["spans"]
    assert rows[0]["toolCatalogHash"] == rows[1]["toolCatalogHash"] != rows[2]["toolCatalogHash"]
    assert [row["toolCount"] for row in rows] == [1, 1, 2]
    assert "private description" not in str(snapshot)


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
    assert request["lastEventMs"] == 800 and request["eventCount"] == 2
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


async def test_interrupted_run_keeps_usage_already_reported():
    class Runner:
        @timed_run
        async def run(self):
            async def complete(**kwargs):
                return {"usage": {"prompt_tokens": 20, "completion_tokens": 3}}
            await timed_completion(complete, {"model": "test"})
            raise TimeoutError("later request timed out")
    runner = Runner()
    with pytest.raises(TimeoutError):
        await runner.run()
    assert runner._last_run_timings["usage"]["total_tokens"] == 23


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


async def test_cost_is_summed_per_request_without_repricing_or_stream_duplicates():
    from rune.llm.pricing import usage_payload

    @timed_run
    async def run():
        async def complete(**_):
            async def stream():
                for _ in range(2):
                    yield {"usage": {"prompt_tokens": 150000, "completion_tokens": 100}}
            return stream()

        # Combined input exceeds Grok's long-context threshold, but each request does not.
        for _ in range(2):
            response = await timed_completion(complete, {"model": "xai/grok-4.6", "stream": True})
            async for _ in response:
                pass

        async def auxiliary(**_):
            return {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
        await timed_completion(auxiliary, {"model": "vertex_ai/gemini-2.5-flash"})
        in_flight = usage_payload(SimpleNamespace(timings={}))
        assert in_flight["cost"]["usd"] == pytest.approx(.60175)
        return CompletionTrace(reason="completed")

    trace = await run()
    payload = usage_payload(trace)
    assert payload["cost"]["usd"] == pytest.approx(.60175)
    assert payload["total"] == 301300
    assert trace.timings["usage"]["calls"] == 3


async def test_partial_usage_reports_unknown_total_and_preserves_known_cost():
    from rune.llm.pricing import usage_payload

    @timed_run
    async def run():
        async def complete(**_):
            return {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
        async def missing(**_):
            return {}
        for model in ("xai/grok-4.6", "unknown"):
            await timed_completion(complete, {"model": model})
        await timed_completion(missing, {"model": "xai/grok-4.6"})
        return CompletionTrace(reason="completed")

    payload = usage_payload(await run())
    assert payload["cost"]["usd"] is None
    assert payload["cost"]["knownUsd"] == pytest.approx(.0026)
    assert payload["cost"]["unpricedCalls"] == 2


async def test_request_failure_records_stage_and_status_without_provider_text():
    import httpx

    @timed_run
    async def run():
        async def fail(**_):
            try:
                raise httpx.ReadTimeout("private request content")
            except httpx.ReadTimeout as cause:
                raise RuntimeError("private provider response") from cause
        with pytest.raises(RuntimeError):
            await timed_completion(fail, {"model": "xai/grok-4.6"})
        return CompletionTrace(reason="failed")

    trace = await run()
    row = trace.timings["spans"][0]
    assert row["error"]["kind"] == "read_timeout"
    assert row["error"]["retryable"] is False
    assert "private" not in str(trace.timings)
    assert trace.timings["usage"]["calls"] == 1
    assert trace.timings["usage"]["reported_calls"] == 0
