"""Multi-Agent Orchestrator v2 - plan, dispatch, retry, merge.

Ported from src/agent/orchestrator.ts (930 lines).
Coordinates multiple specialised agent loops via a shared task board,
with dependency-aware scheduling, failure classification, corrective
retries, and strategic re-planning.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from rune.agent.quality_gate import (
    AgentResult as QAResult,
)
from rune.agent.quality_gate import (
    TaskInfo as QATaskInfo,
)
from rune.agent.quality_gate import (
    check_task_quality,
)
from rune.agent.roles import AgentRoleId, get_role
from rune.agent.task_board import (
    SharedTaskBoard,
    SubTask,
    SubTaskResult,
)
from rune.utils.events import EventEmitter
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Data classes

# Re-export SubTask and SubTaskResult from task_board so callers can import
# from orchestrator directly.
__all__ = [
    "SubTask",
    "SubTaskResult",
    "OrchestrationPlan",
    "OrchestrationResult",
    "OrchestratorConfig",
    "ExecutionMode",
    "Orchestrator",
    "TaskFailureType",
]


@dataclass(slots=True)
class OrchestrationPlan:
    """A decomposed plan ready for parallel execution."""

    tasks: list[SubTask] = field(default_factory=list)
    description: str = ""
    risk_level: str = "low"


@dataclass(slots=True)
class OrchestrationResult:
    """Aggregate result of an orchestrated execution."""

    success: bool = False
    results: list[SubTaskResult] = field(default_factory=list)
    merged_output: str = ""
    duration_ms: float = 0.0


ExecutionMode = Literal["parallel", "sequential"]


@dataclass(slots=True)
class OrchestratorConfig:
    max_workers: int = 3
    max_retries: int = 2
    quality_threshold: float = 0.3
    risk_gate_enabled: bool = True
    execution_mode: ExecutionMode = "parallel"
    subtask_timeout_s: float = 300.0  # 5 minutes default
    abort_on_failure: bool = False  # If True, stop remaining tasks on first failure


TaskFailureType = Literal["transient", "correctable", "systemic", "critical", "unknown"]

# Type alias for the factory that creates per-role agent loops.
# The factory receives the role id and returns an async callable that
# accepts (goal, context) and returns a string output.
AgentLoopFactory = Callable[
    [AgentRoleId],
    Coroutine[Any, Any, Callable[..., Coroutine[Any, Any, str]]],
]


# Orchestrator

class Orchestrator(EventEmitter):
    """Multi-agent orchestrator with dependency-aware parallel execution,
    failure recovery, and result merging.

    Parameters:
        config: Tuning knobs for concurrency, retries, and quality gates.
        agent_loop_factory: Async factory that produces a per-role agent
            runner.  Signature: ``async (role_id) -> async (goal, context) -> str``.
        guardian: Optional guardian instance for risk-gating plans.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        agent_loop_factory: AgentLoopFactory | None = None,
        guardian: Any | None = None,
    ) -> None:
        super().__init__()
        self._config = config or OrchestratorConfig()
        self._agent_loop_factory = agent_loop_factory
        self._guardian = guardian

    # -- Public entry point -------------------------------------------------

    async def execute(
        self,
        goal: str,
        plan: OrchestrationPlan | None = None,
    ) -> OrchestrationResult:
        """Execute an orchestrated multi-agent plan for *goal*.

        If *plan* is not provided, the orchestrator will generate one via
        :meth:`_parse_plan`.
        """
        t0 = time.monotonic()

        if plan is None:
            plan = await self._parse_plan(goal)

        # Risk gate
        if self._config.risk_gate_enabled:
            risk = self._assess_plan_risk(plan)
            if risk == "critical":
                log.warning("orchestrator_risk_gate_blocked", risk=risk)
                return OrchestrationResult(
                    success=False,
                    merged_output="Plan rejected: risk level is critical.",
                    duration_ms=(time.monotonic() - t0) * 1000,
                )
            log.info("orchestrator_risk_assessment", risk=risk)

        await self.emit("plan_ready", plan)

        # Execute the worker pool
        results = await self._execute_worker_pool(plan)

        # Check for failures - attempt strategic replan once
        failed = [r for r in results if not r.success]
        succeeded = [r for r in results if r.success]
        if failed and self._config.max_retries > 0:
            log.info(
                "orchestrator_replan_attempt",
                failed_count=len(failed),
                succeeded_count=len(succeeded),
            )
            new_plan = await self._strategic_replan(goal, failed, succeeded)
            if new_plan is not None and new_plan.tasks:
                retry_results = await self._execute_worker_pool(new_plan)
                # Merge: replace failed results with retried ones where possible
                retry_map = {r.task_id: r for r in retry_results}
                merged: list[SubTaskResult] = []
                for r in results:
                    if not r.success and r.task_id in retry_map:
                        merged.append(retry_map[r.task_id])
                    else:
                        merged.append(r)
                results = merged

        merged_output = self._merge_results(results)
        success_count = sum(1 for r in results if r.success)
        success = success_count >= math.ceil(len(results) / 2)
        duration_ms = (time.monotonic() - t0) * 1000

        result = OrchestrationResult(
            success=success,
            results=results,
            merged_output=merged_output,
            duration_ms=duration_ms,
        )
        await self.emit("completed", result)
        log.info(
            "orchestrator_done",
            success=success,
            task_count=len(results),
            duration_ms=round(duration_ms, 1),
        )
        return result

    # -- Plan generation ----------------------------------------------------

    async def _parse_plan(self, goal: str) -> OrchestrationPlan:
        """Generate an orchestration plan for *goal*.

        In a production system this would use the LLM planner agent.
        The default implementation creates a simple single-task plan so
        that the orchestrator is functional without an LLM connection.
        """
        # Attempt to use the planner role via the agent loop factory
        if self._agent_loop_factory is not None:
            try:
                planner_run = await self._agent_loop_factory("planner")
                raw_plan = await planner_run(
                    f"Break the following goal into sub-tasks with dependencies. "
                    f"Return JSON with tasks list. Goal: {goal}",
                    {},
                )
                parsed = self._try_parse_plan_json(raw_plan)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                log.warning("orchestrator_plan_llm_failed", error=str(exc))

        # Fallback: single executor task
        task = SubTask(
            id=uuid4().hex[:8],
            description=goal,
            role="executor",
        )
        return OrchestrationPlan(
            tasks=[task],
            description=f"Single-task plan for: {goal}",
            risk_level="low",
        )

    @staticmethod
    def _try_parse_plan_json(raw: str) -> OrchestrationPlan | None:
        """Attempt to extract an OrchestrationPlan from LLM JSON output."""
        from rune.utils.fast_serde import json_decode

        try:
            # Find JSON object in the raw output
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json_decode(raw[start:end])
        except ValueError:
            return None

        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return None

        tasks: list[SubTask] = []
        for t in raw_tasks:
            tasks.append(
                SubTask(
                    id=t.get("id", uuid4().hex[:8]),
                    description=t.get("description", ""),
                    role=t.get("role", "executor"),
                    dependencies=t.get("dependencies", []),
                    params=t.get("params", {}),
                    timeout_ms=t.get("timeout_ms", 60_000),
                )
            )
        return OrchestrationPlan(
            tasks=tasks,
            description=data.get("description", ""),
            risk_level=data.get("risk_level", "low"),
        )

    # -- Worker pool --------------------------------------------------------

    async def _execute_worker_pool(
        self,
        plan: OrchestrationPlan,
    ) -> list[SubTaskResult]:
        """Execute all tasks in *plan*.

        Dispatches to :meth:`_execute_parallel` or
        :meth:`_execute_sequential` based on ``execution_mode``.
        """
        if self._config.execution_mode == "sequential":
            return await self._execute_sequential(plan)
        return await self._execute_parallel(plan)

    # -- Parallel execution (default) --------------------------------------

    async def _execute_parallel(
        self,
        plan: OrchestrationPlan,
    ) -> list[SubTaskResult]:
        """Execute tasks in *plan* using a pool of concurrent workers
        coordinated by a :class:`SharedTaskBoard`."""
        board = SharedTaskBoard(plan.tasks)
        workers: list[asyncio.Task[None]] = []
        abort_event = asyncio.Event()
        completed_count = 0
        total_count = len(plan.tasks)

        async def worker(worker_id: str) -> None:
            nonlocal completed_count, total_count
            while not board.is_done() and not abort_event.is_set():
                ready = board.get_ready_tasks()
                claimed = False
                for ready_task in ready:
                    ct = await board.claim(ready_task.id, worker_id)
                    if ct is None:
                        continue
                    claimed = True
                    dep_context = board.get_dependency_context(ct.id)
                    result = await self._execute_subtask_with_retry(
                        SubTask(
                            id=ct.id,
                            description=ct.description,
                            role=ct.role,
                            dependencies=ct.dependencies,
                            params=ct.params,
                            timeout_ms=ct.timeout_ms,
                        ),
                        dep_context,
                    )
                    if result.success:
                        await board.complete(ct.id, result)
                        # Dynamic expansion: if output contains follow-up
                        # tasks, add them to the board.
                        expanded = self._parse_follow_up_tasks(
                            result.output, ct.id,
                        )
                        for new_task in expanded:
                            try:
                                await board.add_task(new_task)
                                total_count += 1
                                log.info(
                                    "dynamic_task_added",
                                    parent=ct.id,
                                    new_task=new_task.id,
                                )
                            except Exception as exc:
                                log.debug(
                                    "dynamic_task_add_failed",
                                    parent=ct.id,
                                    error=str(exc),
                                )
                    else:
                        await board.fail(ct.id, result.error or "unknown error")
                        if self._config.abort_on_failure:
                            log.warning(
                                "orchestrator_abort_on_failure",
                                task_id=ct.id,
                                error=result.error,
                            )
                            abort_event.set()

                    completed_count += 1
                    await self.emit(
                        "progress",
                        completed_count,
                        total_count,
                        ct.id,
                        result.success,
                        ct.description,
                        ct.role,
                    )
                    break  # Re-check ready tasks after each completion

                if not claimed and not board.is_done() and not abort_event.is_set():
                    await board.wait_for_change(timeout=1.0)

        # Start max_workers workers. Even if fewer initial tasks exist,
        # dynamic expansion may add more work during execution.
        for i in range(self._config.max_workers):
            t = asyncio.create_task(worker(f"worker-{i}"), name=f"orch-worker-{i}")
            workers.append(t)

        await asyncio.gather(*workers, return_exceptions=True)

        # Collect results, including dynamically added tasks
        results_map = board.get_results()
        plan_ids = {task.id for task in plan.tasks}
        ordered: list[SubTaskResult] = []
        for task in plan.tasks:
            if task.id in results_map:
                ordered.append(results_map[task.id])
            else:
                aborted = abort_event.is_set()
                ordered.append(
                    SubTaskResult(
                        task_id=task.id,
                        success=False,
                        error="Aborted due to earlier failure" if aborted else "Task did not complete",
                    )
                )
        # Append results from dynamically expanded tasks
        for tid, result in results_map.items():
            if tid not in plan_ids:
                ordered.append(result)
        return ordered

    # -- Sequential execution -----------------------------------------------

    async def _execute_sequential(
        self,
        plan: OrchestrationPlan,
    ) -> list[SubTaskResult]:
        """Execute tasks one-by-one in dependency order.

        Earlier task outputs are passed as dependency context to later
        tasks.  Respects ``abort_on_failure`` - when set, remaining
        tasks are skipped on first failure.
        """
        results: list[SubTaskResult] = []
        results_by_id: dict[str, SubTaskResult] = {}
        total = len(plan.tasks)

        for idx, task in enumerate(plan.tasks):
            # Build dependency context from completed tasks
            dep_context: dict[str, Any] = {}
            for dep_id in task.dependencies:
                if dep_id in results_by_id and results_by_id[dep_id].success:
                    dep_context[dep_id] = results_by_id[dep_id].output

            result = await self._execute_subtask_with_retry(task, dep_context)
            results.append(result)
            results_by_id[task.id] = result

            await self.emit(
                "progress",
                idx + 1,
                total,
                task.id,
                result.success,
            )

            if not result.success and self._config.abort_on_failure:
                log.warning(
                    "orchestrator_sequential_abort",
                    task_id=task.id,
                    error=result.error,
                )
                # Mark remaining tasks as aborted
                for remaining_task in plan.tasks[idx + 1:]:
                    results.append(
                        SubTaskResult(
                            task_id=remaining_task.id,
                            success=False,
                            error="Aborted due to earlier failure",
                        )
                    )
                break

        return results

    # -- Single sub-task execution ------------------------------------------

    async def _execute_subtask_with_retry(
        self,
        task: SubTask,
        dep_context: dict[str, Any],
    ) -> SubTaskResult:
        """Execute *task* with corrective retry on correctable failures."""
        for attempt in range(1 + self._config.max_retries):
            result = await self._execute_subtask(task, dep_context)
            if result.success:
                return result

            failure_type = self._classify_failure(result.error or "")
            log.info(
                "orchestrator_subtask_failed",
                task_id=task.id,
                attempt=attempt + 1,
                failure_type=failure_type,
            )

            if failure_type in ("critical", "systemic"):
                return result

            if failure_type == "correctable" and attempt < self._config.max_retries:
                await self.emit(
                    "subtask_retry", task.id, failure_type,
                    attempt + 1, result.error or "",
                )
                result = await self._corrective_retry(
                    task, result.error or "", dep_context
                )
                if result.success:
                    return result
            elif failure_type == "unknown" and attempt == 0:
                # Unknown errors get a single corrective retry
                log.warning(
                    "orchestrator_unknown_error_retry",
                    task_id=task.id,
                    error=result.error,
                )
                await self.emit(
                    "subtask_retry", task.id, failure_type,
                    1, result.error or "",
                )
                result = await self._corrective_retry(
                    task, result.error or "", dep_context
                )
                if result.success:
                    return result
                # Do not retry unknown errors more than once
                break
            elif failure_type == "transient" and attempt < self._config.max_retries:
                await self.emit(
                    "subtask_retry", task.id, failure_type,
                    attempt + 1, result.error or "",
                )
                await asyncio.sleep(min(2 ** attempt, 8))
                continue
            else:
                break

        return result  # type: ignore[possibly-undefined]

    async def _execute_subtask(
        self,
        task: SubTask,
        dep_context: dict[str, Any],
    ) -> SubTaskResult:
        """Execute a single sub-task via the agent loop factory.

        Uses the per-task ``timeout_ms`` if set, otherwise falls back to
        the orchestrator-level ``subtask_timeout_s`` config.  When no
        ``agent_loop_factory`` is configured the stub path is preserved
        for testing.
        """
        t0 = time.monotonic()
        role_id: AgentRoleId = task.role  # type: ignore[assignment]

        try:
            get_role(role_id)
        except KeyError:
            role_id = "executor"
            get_role(role_id)

        goal = task.description
        context: dict[str, Any] = {
            **task.params,
            "dependencies": dep_context,
        }

        # Determine effective timeout: task-level (ms) takes precedence,
        # then orchestrator config (seconds).
        if task.timeout_ms > 0:
            timeout_s = task.timeout_ms / 1000
        else:
            timeout_s = self._config.subtask_timeout_s

        try:
            if self._agent_loop_factory is not None:
                runner = await self._agent_loop_factory(role_id)
                output = await asyncio.wait_for(
                    runner(goal, context),
                    timeout=timeout_s,
                )
            else:
                # No factory - stub execution for testing
                output = f"[stub] Executed: {goal}"

            duration_ms = (time.monotonic() - t0) * 1000

            # Quality gate: catch hollow success before accepting.
            # Skip for stub outputs (no agent_loop_factory) to avoid
            # false positives on short synthetic responses.
            qc = check_task_quality(
                QATaskInfo(id=task.id, role=role_id, goal=goal),
                QAResult(success=True, answer=str(output), duration_ms=duration_ms),
                threshold=self._config.quality_threshold,
            ) if self._agent_loop_factory is not None else None
            if qc is not None and not qc.passed:
                log.warning(
                    "orchestrator_quality_gate_failed",
                    task_id=task.id,
                    score=qc.score,
                    issues=qc.issues,
                )
                await self.emit("subtask_error", task.id, qc.suggestion or "Quality check failed")
                return SubTaskResult(
                    task_id=task.id,
                    success=False,
                    error=qc.suggestion or "Quality check failed",
                    duration_ms=duration_ms,
                )

            await self.emit("subtask_complete", task.id, output)
            log.info(
                "orchestrator_subtask_done",
                task_id=task.id,
                role=role_id,
                duration_ms=round(duration_ms, 1),
            )
            return SubTaskResult(
                task_id=task.id,
                success=True,
                output=str(output),
                duration_ms=duration_ms,
            )

        except TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000
            error_msg = f"Timeout after {timeout_s:.0f}s"
            log.warning(
                "orchestrator_subtask_timeout",
                task_id=task.id,
                timeout_s=timeout_s,
            )
            await self.emit("subtask_error", task.id, error_msg)
            return SubTaskResult(
                task_id=task.id,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            log.error(
                "orchestrator_subtask_error",
                task_id=task.id,
                error=str(exc),
            )
            await self.emit("subtask_error", task.id, str(exc))
            return SubTaskResult(
                task_id=task.id,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

    # -- Failure handling ---------------------------------------------------

    @staticmethod
    def _classify_failure(error: str) -> TaskFailureType:
        """Classify an error string into a failure type.

        Returns one of: critical, systemic, transient, correctable, unknown.
        ``correctable`` matches errors that are clearly fixable by retrying
        with a different approach.  ``unknown`` is used when no pattern
        matches — it still allows a single retry but is more conservative
        than ``correctable``.
        """
        lower = error.lower()

        # Critical: permission, security, OOM
        critical_patterns = [
            "permission denied", "access denied", "out of memory",
            "segmentation fault", "kill signal", "oom",
        ]
        if any(p in lower for p in critical_patterns):
            return "critical"

        # Systemic: missing dependencies, config errors
        systemic_patterns = [
            "module not found", "import error", "no such file",
            "command not found", "configuration error", "not installed",
        ]
        if any(p in lower for p in systemic_patterns):
            return "systemic"

        # Transient: network, rate limit, timeout
        transient_patterns = [
            "timeout", "rate limit", "connection", "network",
            "temporarily unavailable", "503", "429", "econnreset",
            "econnrefused", "etimedout",
        ]
        if any(p in lower for p in transient_patterns):
            return "transient"

        # Correctable: errors that hint at a fixable approach
        correctable_patterns = [
            "invalid argument", "invalid parameter", "invalid value",
            "type error", "typeerror", "valueerror", "value error",
            "key error", "keyerror", "index error", "indexerror",
            "syntax error", "syntaxerror", "parse error",
            "missing required", "expected", "unexpected token",
            "wrong number", "invalid format", "malformed",
        ]
        if any(p in lower for p in correctable_patterns):
            return "correctable"

        # Default: unknown error — allow limited retry
        return "unknown"

    def _assess_plan_risk(self, plan: OrchestrationPlan) -> str:
        """Assess the aggregate risk level of a plan.

        Returns one of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        """
        if not plan.tasks:
            return "low"

        risk_scores = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        max_risk = 0

        for task in plan.tasks:
            try:
                role = get_role(task.role)  # type: ignore[arg-type]
                role_risk = risk_scores.get(role.risk_level, 0)
            except KeyError:
                role_risk = 2  # Unknown role → high
            max_risk = max(max_risk, role_risk)

        # Escalate if many tasks
        if len(plan.tasks) > 5:
            max_risk = min(max_risk + 1, 3)

        # Use plan-level risk if explicitly set and higher
        plan_risk = risk_scores.get(plan.risk_level, 0)
        max_risk = max(max_risk, plan_risk)

        reverse_map = {v: k for k, v in risk_scores.items()}
        return reverse_map.get(max_risk, "low")

    async def _corrective_retry(
        self,
        task: SubTask,
        error: str,
        dep_context: dict[str, Any],
    ) -> SubTaskResult:
        """Attempt a corrective retry by augmenting the task description
        with the error context."""
        corrected_task = SubTask(
            id=task.id,
            description=(
                f"{task.description}\n\n"
                f"Previous attempt failed with error: {error}\n"
                f"Please try a different approach to accomplish the same goal."
            ),
            role=task.role,
            dependencies=task.dependencies,
            params=task.params,
            timeout_ms=task.timeout_ms,
        )
        return await self._execute_subtask(corrected_task, dep_context)

    async def _strategic_replan(
        self,
        goal: str,
        failed_results: list[SubTaskResult],
        succeeded_results: list[SubTaskResult] | None = None,
    ) -> OrchestrationPlan | None:
        """Attempt to create a new plan accounting for the failures.

        When *succeeded_results* is provided the planner also sees what
        has already been accomplished, so it can avoid duplicate work and
        build on existing outputs.
        """
        if self._agent_loop_factory is None:
            return None

        try:
            planner_run = await self._agent_loop_factory("planner")
            failure_summary = "\n".join(
                f"- Task {r.task_id}: {r.error}" for r in failed_results
            )
            success_summary = ""
            if succeeded_results:
                success_summary = (
                    "\n\nAlready completed tasks (do NOT redo these):\n"
                    + "\n".join(
                        f"- Task {r.task_id}: {r.output[:200]}"
                        for r in succeeded_results
                    )
                )
            raw_plan = await planner_run(
                f"The following sub-tasks failed during execution of the goal: "
                f"{goal}\n\nFailures:\n{failure_summary}"
                f"{success_summary}\n\n"
                f"Create an alternative plan that works around these failures. "
                f"Do not repeat already completed tasks. "
                f"Return JSON with tasks list.",
                {},
            )
            parsed = self._try_parse_plan_json(raw_plan)
            if parsed is not None:
                log.info("orchestrator_replan_success", tasks=len(parsed.tasks))
                return parsed
        except Exception as exc:
            log.warning("orchestrator_replan_failed", error=str(exc))

        return None

    # Dynamic task expansion

    _MAX_FOLLOW_UP_TASKS: int = 5

    @staticmethod
    def _parse_follow_up_tasks(
        output: str, parent_id: str,
    ) -> list[SubTask]:
        """Extract follow-up tasks from a completed subtask's output.

        Looks for a JSON array inside a ```follow_up code fence:

            ```follow_up
            [{"description": "...", "role": "executor"}, ...]
            ```

        Returns parsed SubTasks with automatic dependency on *parent_id*.
        Silently returns [] on any parse failure (graceful degradation).
        """
        import re

        pattern = re.compile(
            r"```follow_up\s*\n(.*?)\n\s*```",
            re.DOTALL,
        )
        match = pattern.search(output)
        if not match:
            return []

        try:
            from rune.utils.fast_serde import json_decode

            raw = json_decode(match.group(1))
            if not isinstance(raw, list):
                return []

            tasks: list[SubTask] = []
            for item in raw[: Orchestrator._MAX_FOLLOW_UP_TASKS]:
                if not isinstance(item, dict) or "description" not in item:
                    continue
                tasks.append(SubTask(
                    id=uuid4().hex[:8],
                    description=item["description"],
                    role=item.get("role", "executor"),
                    dependencies=[parent_id],
                ))
            return tasks
        except Exception:
            return []

    @staticmethod
    def _merge_results(results: list[SubTaskResult]) -> str:
        """Merge outputs from all sub-task results into a single string."""
        if not results:
            return ""

        parts: list[str] = []
        for r in results:
            if r.success and r.output:
                parts.append(r.output)
            elif not r.success:
                parts.append(f"[FAILED: {r.task_id}] {r.error or 'unknown error'}")

        return "\n\n".join(parts)
