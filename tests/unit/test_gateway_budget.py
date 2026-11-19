"""The hosted gateway refusing for the month's budget: the owner is told when it starts again, nothing
is retried, and a long task waits for it instead of ending."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest

from rune.agent.background import BackgroundTask, run_background
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.llm import gateway_budget
from rune.proactive import long_task
from rune.proactive.routine import RoutinePolicy, should_notify
from rune.types import CompletionTrace
from tests.unit.test_background_runs import Loop
from tests.unit.test_long_task import job_of, start, step, store, told  # noqa: F401  (fixtures)

NOV = datetime(2026, 11, 18, 5, 0, tzinfo=UTC)


def refused(code: str | None = "budget_exhausted", status: int = 402) -> litellm.APIError:
    body = f'{{"error":{{"code":"{code}","message":"monthly budget reached"}}}}' if code else "payment required"
    return litellm.APIError(status_code=status, message=body, llm_provider="openai", model="gpt-5.4")


@pytest.fixture
def behind_gateway(monkeypatch):
    monkeypatch.setattr("rune.llm.gateway.gateway_active", lambda: True)


def test_only_the_gateway_s_budget_refusal_is_one(behind_gateway, monkeypatch):
    assert gateway_budget.refusal(refused(), NOV).code == "budget_exhausted"
    assert gateway_budget.refusal(refused("host_budget_exhausted"), NOV).code == "host_budget_exhausted"
    assert gateway_budget.refusal(refused(None), NOV) is None, "a provider's own 402 is the operator's to fix"
    import httpx
    response = httpx.Response(402, request=httpx.Request("GET", "https://example.test"))
    assert gateway_budget.refusal(httpx.HTTPStatusError("402", request=response.request, response=response), NOV) is None
    double = litellm.APIConnectionError(message='{\\"error\\": {\\"code\\": \\"budget_exhausted\\"}}',
                                        llm_provider="anthropic", model="claude-sonnet-5-5")
    assert gateway_budget.refusal(double, NOV).code == "budget_exhausted", "its body encoded once more"
    assert gateway_budget.refusal(ValueError("boom"), NOV) is None
    try:
        try:
            raise refused("host_budget_exhausted")
        except litellm.APIError as inner:
            raise RuntimeError("the call failed") from inner
    except RuntimeError as outer:
        assert gateway_budget.refusal(outer, NOV).code == "host_budget_exhausted", "behind another error"
    monkeypatch.setattr("rune.llm.gateway.gateway_active", lambda: False)
    assert gateway_budget.refusal(refused(), NOV) is None, "a provider's own 402 is its business"


def test_it_starts_again_on_the_1st_and_says_so_in_the_owner_s_language():
    found = gateway_budget.BudgetRefusal("budget_exhausted", gateway_budget.next_month(NOV))
    assert found.resumes == datetime(2026, 12, 1, tzinfo=UTC)
    assert gateway_budget.next_month(datetime(2026, 12, 31, 23, 59, tzinfo=UTC)) == datetime(2027, 1, 1, tzinfo=UTC)
    assert found.message() == ("You've used this month's budget for your Rune's models, so I can't answer right "
                               "now. It starts again on December 1 (UTC).")
    assert found.message("ko-KR") == "이번 달 모델 사용 예산을 다 써서 지금은 답할 수 없어요. 12월 1일 09:00(한국 시간)부터 다시 쓸 수 있어요."
    host = gateway_budget.BudgetRefusal("host_budget_exhausted", found.resumes)
    assert "This server has reached its shared model budget" in host.message()
    assert "이 서버의 이번 달 전체" in host.message("ko")
    assert found.waiting().endswith("This task waits until then and goes on from where it was.")
    assert found.resume_minute * 60 == int(found.resumes.timestamp())


async def test_a_run_refused_stops_with_the_message_and_tries_nothing_else(behind_gateway, monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    calls = []

    class Stream:
        tool_budget_exhausted = tool_round_limit_reached = False
        answer_reserve_reason = ""

        async def stream_text(self, **kwargs):
            calls.append(kwargs)
            raise refused()
            yield ""

        async def get_output(self):
            return ""

        def usage(self): return SimpleNamespace(input_tokens=0, output_tokens=0)
        def get_failure_state(self): return {}, set()
        def all_messages(self): return []

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            yield Stream()

    monkeypatch.setattr("rune.agent.loop.build_tool_set", lambda opts: {})
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    said = []
    loop.on("text_delta", lambda text: said.append(text))
    trace = await loop._execute_loop("What's on my calendar?", "", [], 5, ClassificationResult("chat", .99, 1))
    assert trace.reason == gateway_budget.REASON
    assert len(calls) == 1, "not retried, and no other model tried"
    assert loop._last_answer_text.startswith("You've used this month's budget")
    assert said == [loop._last_answer_text], "streamed, so every channel answers with it"
    assert loop._gateway_budget.code == "budget_exhausted"


class Refused(Loop):
    """A background run whose model calls the gateway refused, after `calls` tool calls."""

    def __init__(self, calls=()):
        super().__init__(CompletionTrace(reason=gateway_budget.REASON))
        self._gateway_budget = gateway_budget.BudgetRefusal("budget_exhausted", datetime(2026, 12, 1, tzinfo=UTC))
        self.calls, self.listeners = calls, {}

    def on(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    async def run(self, goal, **kwargs):
        for call in self.calls:
            for callback in self.listeners.get("tool_call", []):
                callback(call)
        return self.trace


async def test_a_background_run_refused_before_changing_anything_waits():
    loop = Refused([{"name": "web_fetch", "params": {"url": "https://example.test"}}])
    result = await run_background(BackgroundTask("Report", "test"), loop_factory=lambda **kw: loop)
    assert result["status"] == "model_budget" and not result["success"]
    assert result["not_started"], "the step that goes on reads what the one before it left"
    assert result["resume_at"] == loop._gateway_budget.resume_minute
    assert "goes on from where it was" in result["error"]


async def test_a_background_run_refused_after_changing_something_is_checked_first():
    loop = Refused([{"name": "file_write", "params": {"path": "notes.md", "content": "x"}}])
    result = await run_background(BackgroundTask("Report", "test"), loop_factory=lambda **kw: loop)
    assert result["status"] == "interrupted" and result["execution_unknown"]
    assert "Check what it did" in result["error"]
    assert long_task.after_step(None, result, 2, 20) == ({"wake_at": None, "ended": "interrupted"}, result["error"])


async def test_a_long_task_s_last_allowed_step_refused_ends_it(store, told, monkeypatch):  # noqa: F811
    job_id = await start(store, steps=1)
    await step(store, job_id, monkeypatch, {"status": "model_budget", "success": False, "verified": False,
                                            "not_started": True, "output": "", "error": "Used up.",
                                            "resume_at": 99_999_999})
    job = job_of(store, job_id)
    assert job.status == "paused" and job.task["ended"] == "step_limit", "not left waiting for a step it can't take"


async def test_a_long_task_waits_for_the_month_instead_of_ending(store, told, monkeypatch):  # noqa: F811
    job_id = await start(store)
    resume = int(datetime(2026, 12, 1, tzinfo=UTC).timestamp()) // 60
    await step(store, job_id, monkeypatch, {"status": "model_budget", "success": False, "verified": False,
                                            "output": "", "error": "Used up. This task waits until then.",
                                            "resume_at": resume})
    job = job_of(store, job_id)
    assert job.status == "active" and job.wake_at == resume, "not ended"
    assert told.await_args.args[2] == "Used up. This task waits until then."


def test_a_scheduled_run_refused_is_told_once():
    policy = RoutinePolicy(notify="failures")
    result = {"status": "model_budget", "output": "", "error": "Used up."}
    assert should_notify(policy, result, None)
    assert not should_notify(policy, result, dict(result)), "the same again isn't news"
    assert not should_notify(RoutinePolicy(notify="always"), result, dict(result)), "not even to always"
    assert long_task.after_step(None, result, 1, 20) == ({"wake_at": None, "ended": "model_budget"},
                                                        "Stopped (model_budget): Used up."), \
        "without a time to wait for, it can only end"


async def test_the_step_after_the_month_goes_on_from_the_last_note(store, told, monkeypatch):  # noqa: F811
    from tests.unit.test_long_task import now_minute, parked
    job_id = await start(store)
    await step(store, job_id, monkeypatch, parked("Read 1-100. Next: 101-200.", minutes=1))
    second_at, resume = (now_minute() + 1) * 60 + 5, now_minute() + 60
    monkeypatch.setattr("time.time", lambda: second_at)
    await step(store, job_id, monkeypatch, {"status": "model_budget", "success": False, "verified": False,
                                            "not_started": True, "output": "", "error": "Used up.",
                                            "resume_at": resume})
    monkeypatch.setattr("time.time", lambda: resume * 60 + 5)
    third = await step(store, job_id, monkeypatch, parked("Read 101-200."))
    assert "Read 1-100. Next: 101-200." in third.await_args.args[0].continuation


def wrapped(code: str = "budget_exhausted") -> Exception:
    """The refusal wrapped in a routing backend's own error, as it hands it on."""
    from rune.agent.classification_response import RoutingUnavailable
    try:
        raise refused(code)
    except litellm.APIError as exc:
        try:
            raise RoutingUnavailable("Connected routing unavailable") from exc
        except RoutingUnavailable as outer:
            return outer


@pytest.mark.parametrize("raised", [refused, wrapped])
async def test_a_refusal_while_routing_is_said_as_such(behind_gateway, monkeypatch, tmp_path, raised):
    """A run's first model call is the task router's: its refusal must not read as "routing unavailable"."""
    from rune.agent import goal_classifier

    monkeypatch.setattr("rune.agent.decision_router.classify_request", AsyncMock(side_effect=raised()))
    with pytest.raises(Exception) as caught:
        await goal_classifier.classify_tier2("What's on my calendar?")
    assert gateway_budget.refusal(caught.value) is not None
    monkeypatch.setattr("rune.llm.gateway.gateway_active", lambda: False)
    assert not (await goal_classifier.classify_tier2("What's on my calendar?")).available, "other failures as before"

    # A whole run with only the router's call refused, as on a VM whose budget is used up.
    monkeypatch.setattr("rune.llm.gateway.gateway_active", lambda: True)
    monkeypatch.setattr("rune.agent.decision_router.classify_request",
                        AsyncMock(side_effect=raised("host_budget_exhausted")))
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    said, done = [], []
    loop.on("text_delta", lambda text: said.append(text))
    loop.on("completed", lambda trace: done.append(trace))
    trace = await loop.run("What's on my calendar?")
    assert trace.reason == gateway_budget.REASON
    assert said == [loop._last_answer_text] and "This server has reached its shared model budget" in said[0]
    assert done == [trace], "listeners see the run end"
