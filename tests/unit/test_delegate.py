"""Tests for delegation capabilities.

Regression guard: delegate_orchestrate must wire a real agent loop factory into
the orchestrator. Without one the orchestrator runs the stub path and returns
"[stub] Executed: ..." as success, which fabricates completion.
"""

from __future__ import annotations

import pytest

import rune.agent.orchestrator as orch_mod
from rune.capabilities.delegate import (
    DelegateOrchestrateParams,
    _build_base_agent_config,
    delegate_orchestrate,
)


def test_build_base_agent_config_uses_session_model():
    cfg = _build_base_agent_config()
    # Provider/model resolve to non-empty values from config defaults.
    assert cfg.provider
    assert cfg.model


@pytest.mark.asyncio
async def test_delegate_orchestrate_wires_real_factory(monkeypatch):
    captured = {}

    class _FakeOrchestrator:
        def __init__(self, *, config=None, agent_loop_factory=None, guardian=None):
            captured["factory"] = agent_loop_factory
            captured["execution_mode"] = getattr(config, "execution_mode", None)
            captured["risk_gate_enabled"] = getattr(config, "risk_gate_enabled", None)

        async def execute(self, goal):
            from rune.agent.orchestrator import OrchestrationResult, SubTaskResult
            return OrchestrationResult(
                success=True,
                results=[SubTaskResult(task_id="t1", success=True, output="real")],
                merged_output="real",
                duration_ms=1.0,
            )

    monkeypatch.setattr(orch_mod, "Orchestrator", _FakeOrchestrator)

    result = await delegate_orchestrate(
        DelegateOrchestrateParams(goal="build a thing", max_workers=2)
    )

    assert result.success is True
    # A real factory is wired, so workers run instead of returning stubs.
    assert captured["factory"] is not None
    assert captured["execution_mode"] == "sequential"
    assert captured["risk_gate_enabled"] is False


@pytest.mark.asyncio
async def test_delegate_orchestrate_reports_failure_honestly(monkeypatch):
    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            pass

        async def execute(self, goal):
            from rune.agent.orchestrator import OrchestrationResult, SubTaskResult
            return OrchestrationResult(
                success=False,
                results=[SubTaskResult(task_id="t1", success=False, error="boom")],
                merged_output="[FAILED: 1] boom",
                duration_ms=1.0,
            )

    monkeypatch.setattr(orch_mod, "Orchestrator", _FakeOrchestrator)

    result = await delegate_orchestrate(
        DelegateOrchestrateParams(goal="x", max_workers=1)
    )
    # A failing orchestration surfaces as failure, not fabricated success.
    assert result.success is False
    assert "failed" in (result.error or "").lower()
