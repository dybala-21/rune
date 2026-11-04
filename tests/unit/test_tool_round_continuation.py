"""A stream boundary must not turn speculative progress into a completed run."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.agent.run_outcome import run_outcome
from rune.types import CapabilityResult
from tests.unit.test_live_streaming import _chunk, _make, _tc


@pytest.mark.parametrize("reserve", [False, True])
async def test_stream_reports_resumable_boundary_and_keeps_executed_results(monkeypatch, reserve):
    stream = _make("anthropic/claude-opus-5")
    stream._max_tool_rounds = 1
    stream._reserve_answer = reserve
    stream._reserve_answer_at_round_limit = False
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)

        async def chunks():
            yield _chunk("Looking up the next result.")
            yield _chunk(tool_calls=_tc(), finish="tool_calls")
        return chunks()

    async def execute(batch):
        stream._messages.append({"role": "tool", "tool_call_id": batch[0]["id"],
                                 "content": f"Checked offer {len(calls)}"})

    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    _ = [text async for text in stream.stream_text()]
    assert stream.tool_round_limit_reached and stream.tool_budget_exhausted
    assert await stream.get_output() == ""
    assert stream.all_messages()[-1]["content"] == "Checked offer 2"
    assert not stream.answer_reserve_reason


@pytest.mark.parametrize("stop", ["complete", "iterations", "cancelled", "budget", "limited_answer", "timed_answer"])
async def test_real_loop_resumes_history_but_obeys_run_limits(monkeypatch, tmp_path, stop):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    options, turns = [], []
    history = [{"role": "assistant", "tool_calls": [
        {"id": "observed", "type": "function", "function": {"name": "browser_extract", "arguments": "{}"}},
    ]}, {"role": "tool", "tool_call_id": "observed", "content": "Offer A: 51000"}]

    class Stream:
        tool_budget_exhausted = False
        tool_round_limit_reached = False
        answer_reserve_reason = ""

        async def stream_text(self, **kwargs):
            if len(turns) == 2 and stop == "budget":
                from rune.agent.request_budget import BudgetExceeded
                raise BudgetExceeded("The model request limit was reached.")
            await options[-1].on_tool_start("browser_extract", {})
            await options[-1].on_tool_end("browser_extract", CapabilityResult(success=True, output="Offer A: 51000"))
            if len(turns) == 1:
                self.tool_budget_exhausted = self.tool_round_limit_reached = True
                if stop == "cancelled":
                    loop._cancel_event.set()
                yield "Still retrieving the remaining offers."
            else:
                if stop in {"limited_answer", "timed_answer"}:
                    self.answer_reserve_reason = "budget" if stop == "limited_answer" else "time"
                yield "The verified offer is 51000."

        async def get_output(self):
            return "" if self.tool_round_limit_reached else "The verified offer is 51000."

        def usage(self): return SimpleNamespace(input_tokens=10, output_tokens=10)
        def get_failure_state(self): return {}, set()
        def all_messages(self): return list(history)

    class Agent:
        def __init__(self, *args, **kwargs):
            assert kwargs["reserve_answer"] is True
            assert kwargs["reserve_answer_at_round_limit"] is False

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            turns.append(kwargs)
            yield Stream()

    def build(opts):
        options.append(opts)
        return {}

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    classification = ClassificationResult("browser", .99, 2, browser_effect="view")
    trace = await loop._execute_loop("Retrieve the offer price", "", [], 1 if stop == "iterations" else 2, classification)
    assert run_outcome(trace).success is (stop == "complete")
    assert "Still retrieving" not in loop._last_answer_text
    assert len(turns) == (1 if stop in {"iterations", "cancelled"} else 2)
    if len(turns) == 2:
        assert turns[1]["message_history"] == history
    if stop == "complete":
        assert loop._last_answer_text == "The verified offer is 51000."
        assert not trace.tool_budget_exhausted
    elif stop in {"limited_answer", "timed_answer"}:
        assert loop._last_answer_text == "The verified offer is 51000."
        assert trace.tool_budget_exhausted and trace.reason == "task_incomplete"
        assert trace.completion_check["name"] == "Browser retrieval coverage"


async def test_resumable_browser_work_reserves_cost_for_a_final_answer(monkeypatch):
    from rune.agent.timing import capture_timing
    from tests.unit.test_request_budget import owner

    stream = _make("anthropic/claude-opus-5")
    stream._reserve_answer = True
    stream._reserve_answer_at_round_limit = False
    requests = []
    monkeypatch.setattr("rune.agent.spend_budget.enabled", lambda: False)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 100)
    monkeypatch.setattr("rune.llm.pricing.reserve_request_cost", lambda p, i, o: (i + o) * .0001)
    monkeypatch.setattr("rune.llm.pricing.estimate_request_cost", lambda *args, **kwargs: .03)

    async def completion(**kwargs):
        requests.append(kwargs)
        assert len(requests) <= 2

        async def chunks():
            yield (_chunk(tool_calls=_tc(), finish="tool_calls") if len(requests) == 1
                   else _chunk('{"offers":[],"limitations":["Unverified conditions"]}', finish="stop"))
            yield SimpleNamespace(choices=[], usage={"prompt_tokens": 100, "completion_tokens": 10})
        return chunks()

    async def execute(batch):
        stream._messages.append({"role": "tool", "tool_call_id": batch[0]["id"], "content": "Observed room details"})

    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    with capture_timing(owner(tokens=100000, cost=.15)):
        _ = [text async for text in stream.stream_text()]
    assert stream.answer_reserve_reason == "budget"
    assert not requests[-1].get("tools")
    assert "Observed room details" in str(requests[-1]["messages"])
    assert '"limitations"' in await stream.get_output()
    assert not stream.tool_round_limit_reached
