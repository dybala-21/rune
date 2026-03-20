"""Tests for core types."""

from __future__ import annotations

from rune.types import (
    AgentConfig,
    AppState,
    AppStatus,
    Domain,
    Message,
    MessageRole,
    Plan,
    RiskLevel,
    Step,
    Task,
    TaskStatus,
    ToolResult,
)


class TestTypes:
    def test_domain_enum(self):
        assert Domain.FILE == "file"
        assert Domain.BROWSER == "browser"

    def test_risk_level_enum(self):
        assert RiskLevel.LOW == "low"
        assert RiskLevel.CRITICAL == "critical"

    def test_task_defaults(self):
        task = Task(goal="test")
        assert task.status == TaskStatus.PENDING
        assert task.id  # auto-generated

    def test_tool_result(self):
        result = ToolResult(success=True, output="hello")
        assert result.success
        assert result.error is None

    def test_plan(self):
        plan = Plan(
            steps=[Step(description="step 1"), Step(description="step 2")],
            risk_level=RiskLevel.MEDIUM,
        )
        assert len(plan.steps) == 2
        assert plan.risk_level == "medium"

    def test_app_state(self):
        state = AppState()
        assert state.status == AppStatus.IDLE
        assert state.messages == []

    def test_message(self):
        msg = Message(role=MessageRole.USER, content="hello")
        assert msg.role == "user"
        assert msg.timestamp is not None

    def test_agent_config_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 200
        assert config.timeout_seconds == 1800
