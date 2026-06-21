"""Agent-loop factory for orchestrator workers.

The orchestrator runs each subtask through an agent_loop_factory. Without one it
falls back to a stub string, so delegate_orchestrate wires this factory to run
every subtask in a real NativeAgentLoop. Each worker gets a slice of the total
token budget so one cannot consume it all.
"""

from __future__ import annotations

import copy
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_MIN_WORKER_BUDGET = 40_000


def make_worker_factory(base_config: Any, total_budget: int, n_workers: int):
    """Return an agent_loop_factory: ``factory(role_id) -> runner(goal, context)``.

    The runner executes the subgoal in a real NativeAgentLoop. The per-worker
    budget slice is applied via ``token_budget_override`` because ``run()``
    otherwise resets the budget from the intent map.
    """
    from rune.agent.loop import NativeAgentLoop
    from rune.agent.orchestrator import WorkerResult
    from rune.types import AgentConfig

    worker_budget = max(total_budget // max(n_workers, 1), _MIN_WORKER_BUDGET)

    async def factory(role_id: Any):
        async def runner(
            goal: str, context: dict[str, Any] | None = None
        ) -> WorkerResult:
            cfg = copy.copy(base_config) if base_config is not None else AgentConfig(
                model="", provider="")
            cfg.token_budget_override = worker_budget
            loop = NativeAgentLoop(config=cfg)
            # Count tool calls as actions for the quality gate. A read-only
            # verify or research worker does real work without writing files, so
            # counting only file writes would reject it.
            action_count = {"n": 0}

            async def _count_tool_call(_info: dict[str, Any]) -> None:
                action_count["n"] += 1

            loop.on("tool_call", _count_tool_call)
            trace = await loop.run(goal, context=context)
            # final_step is the iteration count; tool calls are the actions.
            iterations = int(getattr(trace, "final_step", 0) or 0)
            actions = action_count["n"]
            out = (getattr(loop, "_last_answer_text", "") or "").strip()
            if not out:
                out = f"(worker produced no answer: {getattr(trace, 'reason', '?')})"
            return WorkerResult(answer=out, iterations=iterations, actions=actions)
        return runner

    return factory
