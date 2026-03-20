"""Delegation capabilities for RUNE.

Ported from src/capabilities/delegate.ts - creates sub-agents with
specified roles and orchestrates multi-task decomposition.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schemas

class DelegateTaskParams(BaseModel):
    goal: str = Field(description="Goal for the sub-agent to accomplish")
    role: str = Field(
        default="executor",
        description="Sub-agent role: researcher/planner/executor/communicator",
    )


class DelegateOrchestrateParams(BaseModel):
    goal: str = Field(description="Goal to decompose and orchestrate")
    max_workers: int = Field(default=3, alias="maxWorkers",
                             description="Maximum concurrent sub-agents")


# Implementations

async def delegate_task(params: DelegateTaskParams) -> CapabilityResult:
    """Create and run a sub-agent with the specified role."""
    log.info("delegate_task", goal=params.goal[:100], role=params.role)

    try:
        from rune.agent.roles import AgentRoleId, get_role

        # Validate role
        valid_roles = ("researcher", "planner", "executor", "communicator")
        role_id: AgentRoleId = params.role  # type: ignore[assignment]
        if params.role not in valid_roles:
            role_id = "executor"
            log.warning("delegate_unknown_role", role=params.role, fallback="executor")

        role = get_role(role_id)
        log.debug(
            "delegate_role_resolved",
            role=role.name,
            capabilities=len(role.capabilities),
        )

        # Attempt to use the agent loop for execution
        try:
            from rune.agent.loop import create_agent_loop

            agent = create_agent_loop(
                role=role_id,
                max_iterations=role.max_iterations,
                timeout_seconds=role.timeout_seconds,
            )
            result = await agent.run(params.goal)

            return CapabilityResult(
                success=True,
                output=result,
                metadata={
                    "role": role_id,
                    "goal": params.goal,
                },
            )

        except ImportError:
            log.debug("delegate_no_agent_loop", reason="agent loop not available")

        # Fallback 2: Try the orchestrator for single-task execution
        try:
            from rune.agent.orchestrator import Orchestrator, OrchestratorConfig

            guardian: Any = None
            try:
                from rune.safety.guardian import get_guardian
                guardian = get_guardian()
            except Exception:
                pass

            orchestrator = Orchestrator(
                config=OrchestratorConfig(max_workers=1),
                guardian=guardian,
            )
            orch_result = await orchestrator.execute(params.goal)

            return CapabilityResult(
                success=orch_result.success,
                output=orch_result.merged_output,
                metadata={
                    "role": role_id,
                    "goal": params.goal,
                    "via": "orchestrator_fallback",
                    "duration_ms": round(orch_result.duration_ms, 1),
                },
            )
        except Exception as orch_exc:
            log.debug(
                "delegate_orchestrator_fallback_failed",
                error=str(orch_exc),
            )

        # Fallback 3: Last resort - return a descriptive stub message
        return CapabilityResult(
            success=True,
            output=(
                f"[Delegated to {role.name}]\n"
                f"Goal: {params.goal}\n"
                f"Role capabilities: {', '.join(role.capabilities[:10])}\n"
                f"Max iterations: {role.max_iterations}\n\n"
                f"Note: Neither the agent loop nor the orchestrator could "
                f"be initialised. Install pydantic-ai or configure an LLM "
                f"provider to enable full execution."
            ),
            metadata={
                "role": role_id,
                "goal": params.goal,
                "stub": True,
            },
        )

    except Exception as exc:
        log.error("delegate_task_failed", error=str(exc))
        return CapabilityResult(
            success=False,
            error=f"Delegation failed: {exc}",
        )


async def delegate_orchestrate(params: DelegateOrchestrateParams) -> CapabilityResult:
    """Orchestrate a multi-task decomposition using the Orchestrator."""
    log.info(
        "delegate_orchestrate",
        goal=params.goal[:100],
        max_workers=params.max_workers,
    )

    try:
        from rune.agent.orchestrator import (
            Orchestrator,
            OrchestratorConfig,
        )

        config = OrchestratorConfig(max_workers=params.max_workers)

        # Try to get a guardian for risk gating
        guardian: Any = None
        try:
            from rune.safety.guardian import get_guardian
            guardian = get_guardian()
        except Exception:
            pass

        orchestrator = Orchestrator(
            config=config,
            guardian=guardian,
        )

        result = await orchestrator.execute(params.goal)

        if result.success:
            return CapabilityResult(
                success=True,
                output=result.merged_output,
                metadata={
                    "goal": params.goal,
                    "task_count": len(result.results),
                    "duration_ms": round(result.duration_ms, 1),
                    "all_succeeded": all(r.success for r in result.results),
                },
            )
        else:
            # Partial or full failure
            failed = [r for r in result.results if not r.success]
            return CapabilityResult(
                success=False,
                output=result.merged_output,
                error=f"{len(failed)}/{len(result.results)} sub-tasks failed",
                metadata={
                    "goal": params.goal,
                    "task_count": len(result.results),
                    "failed_count": len(failed),
                    "duration_ms": round(result.duration_ms, 1),
                },
            )

    except Exception as exc:
        log.error("delegate_orchestrate_failed", error=str(exc))
        return CapabilityResult(
            success=False,
            error=f"Orchestration failed: {exc}",
        )


# Registration

def register_delegate_capabilities(registry: CapabilityRegistry) -> None:
    """Register delegation capabilities."""
    registry.register(CapabilityDefinition(
        name="delegate_task",
        description="Delegate a task to a specialised sub-agent",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.MEDIUM,
        group="delegate",
        parameters_model=DelegateTaskParams,
        execute=delegate_task,
    ))
    registry.register(CapabilityDefinition(
        name="delegate_orchestrate",
        description="Orchestrate multi-task decomposition with parallel sub-agents",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.MEDIUM,
        group="delegate",
        parameters_model=DelegateOrchestrateParams,
        execute=delegate_orchestrate,
    ))
