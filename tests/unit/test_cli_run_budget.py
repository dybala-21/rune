"""CLI preflight, agent work and learning consume the same allowance."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.timing import capture_execution, capture_timing, timed_completion


@pytest.fixture
def cli_run(tmp_path, monkeypatch):
    from rune.agent import agent_context
    from rune.cli import main
    from rune.config import get_config

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("RUNE_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("RUNE_IN_BEST_OF", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "model_request_limit", 0)
    monkeypatch.setattr(cfg.llm, "cost_budget_usd", None)
    monkeypatch.setattr(cfg.llm, "active_provider", "openai")
    monkeypatch.setattr(cfg.llm, "active_model", "gpt-4o")
    monkeypatch.setattr(cfg.llm.decision_routing, "backend", "connected")
    decision = dict(goal_type="chat", confidence=.99, reason="Simple answer", requires_execution=False,
        intent_categories=[], requires_desktop_input=False, is_related_to_previous=False,
        table_output="none", calculation_expression="")
    request = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(decision)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    monkeypatch.setattr("litellm.acompletion", request)
    monkeypatch.setattr(main, "_ensure_llm_key", lambda: True)
    monkeypatch.setattr(main, "_wire_cli_approval", lambda _: None)
    monkeypatch.setattr(main, "_defer_memory_work", lambda: None)
    monkeypatch.setattr("rune.mcp.config.load_mcp_config", lambda: [])
    monkeypatch.setattr("rune.mcp.client.get_mcp_client_manager", lambda: SimpleNamespace(disconnect_all=AsyncMock()))
    monkeypatch.setattr("rune.memory.manager.get_memory_manager", lambda: SimpleNamespace(
        build_memory_context=AsyncMock(return_value=""), promote_memories=AsyncMock()))
    monkeypatch.setattr(agent_context, "prepare_agent_context", AsyncMock(return_value=agent_context.AgentContext(
        goal="Hello", workspace_root=str(tmp_path))))
    monkeypatch.setattr(agent_context, "post_process_agent_result", AsyncMock(return_value=[]))
    return SimpleNamespace(run=main._handle_non_interactive, request=request, config=cfg.llm)


def test_zero_limit_blocks_cli_classification(cli_run, capsys):
    cli_run.run("Hello")
    cli_run.request.assert_not_called()
    assert "request_budget_exhausted" in capsys.readouterr().err


@pytest.mark.parametrize("limit", [1, 2])
def test_cli_preflight_and_learning_share_the_limit(cli_run, monkeypatch, limit):
    from rune.agent import agent_context
    from rune.agent.loop import NativeAgentLoop
    from rune.agent.timing import timed_run
    from rune.types import CompletionTrace

    monkeypatch.setattr(cli_run.config, "model_request_limit", limit)
    seen = []
    @timed_run
    async def run(self, *args, **kwargs):
        self._reset_run_state()
        trace = CompletionTrace(reason="completed")
        seen.append(trace)
        return trace
    monkeypatch.setattr(NativeAgentLoop, "run", run)
    async def learn(inp):
        for _ in range(2):
            await timed_completion(cli_run.request, {"model": "openai/gpt-4o", "messages": [], "max_tokens": 100})
        return []
    monkeypatch.setattr(agent_context, "post_process_agent_result", learn)
    cli_run.run("Hello")
    assert cli_run.request.await_count == limit
    assert seen[0].timings["usage"]["calls"] == limit
    assert seen[0].total_tokens_used == 110 * limit
    assert seen[0].reason == "completed"


async def test_execution_tokens_follow_loop_limits_without_resetting_usage(monkeypatch):
    from rune.agent.request_budget import BudgetExceeded
    from rune.types import AgentConfig

    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    loop = SimpleNamespace(_config=AgentConfig(), _token_budget=SimpleNamespace(total=500000, used=0))
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    params = {"model": "openai/gpt-4o", "messages": [], "max_tokens": 100}
    with capture_execution(loop) as measured:
        await timed_completion(request, params)
        loop._token_budget = SimpleNamespace(total=150, used=0)
        with capture_timing(loop):
            with pytest.raises(BudgetExceeded):
                await timed_completion(request, params)
        assert measured["usage"]["total_tokens"] == 110
        with pytest.raises(BudgetExceeded):
            await timed_completion(request, params)
    assert request.await_count == 1
