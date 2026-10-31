"""Final replies preserve pending checks in the agent loop."""

import json
from types import SimpleNamespace

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.agent.run_outcome import run_outcome
from rune.agent.timing import capture_timing
from rune.types import AgentConfig, CapabilityResult
from tests.unit.test_task_blocked import _astream, _delta


@pytest.mark.parametrize("kind,checked", [("code", False), ("code", True), ("document", True)])
async def test_final_reply_does_not_replace_missing_work_checks(monkeypatch, tmp_path, checked, kind):
    for name, value in {"RUNE_HOME":str(tmp_path), "RUNE_IN_BEST_OF":"1", "RUNE_AUTO_VERIFY":"0",
                        "RUNE_REQUIREMENT_GATE":"0", "RUNE_ADVISOR":"0", "RUNE_AUTO_SKILL":"0"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("rune.agent.spend_budget.enabled", lambda: False)
    from rune.config import get_config
    get_config().llm.route_simple_queries = False
    options = None

    def build_tools(opts):
        nonlocal options
        options = opts
        async def file_write(path: str, content: str):
            return content
        async def bash_execute(command: str):
            return command
        return {"file_write":file_write, "bash_execute":bash_execute}

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build_tools)
    monkeypatch.setattr("rune.agent.litellm_adapter.StreamResult._start_artifact_role_classification", lambda self: None)

    async def execute(stream, calls):
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            await options.on_tool_start(name, args)
            output = f"Saved {args['path']}" if name == "file_write" else "3 passed" if checked else "1 failed, 2 passed"
            success = name == "file_write" or checked
            await options.on_tool_end(name, CapabilityResult(success=success, output=output))
            stream._messages.append({"role":"tool", "tool_call_id":call["id"], "content":output})

    monkeypatch.setattr("rune.agent.litellm_adapter.StreamResult._execute_tool_batch", execute)
    def work_tools(params):
        return any(t.get("function", {}).get("name") in {"file_write", "bash_execute"} for t in params.get("tools") or [])
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda p: 4000 if work_tools(p) else 1000)
    requests = []
    async def completion(**params):
        requests.append(params)
        assert len(requests) <= 2
        usage = SimpleNamespace(choices=[], usage={"prompt_tokens":4000 if work_tools(params) else 1000, "completion_tokens":100})
        if len(requests) == 1:
            path = str(tmp_path / ("main.py" if kind == "code" else "report.txt"))
            actions = [{"index":0, "name":"file_write", "id":"write", "arguments":json.dumps({"path":path,"content":"Result"})}]
            if kind == "code":
                actions.append({"index":1, "name":"bash_execute", "id":"check", "arguments":'{"command":"pytest -q"}'})
            return _astream([_delta(tool_calls=actions, finish_reason="tool_calls"), usage])
        assert not work_tools(params)
        return _astream([_delta(content="The requested result is saved; the recorded checks are shown above.", finish_reason="stop"), usage])

    monkeypatch.setattr("litellm.acompletion", completion)
    loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=1, max_tokens=1024))
    loop._token_budget.total = 7000
    loop._auto_skill = False
    classification = ClassificationResult(goal_type="code_modify" if kind == "code" else "artifact", confidence=.99,
        tier=2, intent_categories=frozenset({"coding"} if kind == "code" else {"document"}),
        requires_code=kind == "code", output_expectation="file")
    with capture_timing(loop) as run:
        trace = await loop._execute_loop(goal="Fix main.py and run the tests" if kind == "code" else "Save report.txt",
            system_prompt="Carry out the request.", tools=["file_write", "bash_execute"], max_iterations=1,
            classification=classification, context={"workspace_root":str(tmp_path),"ephemeral":True})
    if kind == "code" and not checked:
        assert len(requests) == 1 and not run_outcome(trace).success
        assert trace.reason == "request_budget_exhausted"
        assert loop._verification.pending
    else:
        assert len(requests) == 2 and run_outcome(trace).success
        assert trace.answer_reserve_reason == "budget" and not run["budget_blocked"]
    assert run["usage"]["calls"] == len(requests)
