"""Tests for the agent roles module."""

from __future__ import annotations

import pytest

from rune.agent.roles import get_role, list_role_capabilities, suggest_role


class TestAgentRoles:
    def test_get_role_researcher(self):
        role = get_role("researcher")
        assert role.id == "researcher"
        assert role.name == "Researcher"
        assert "file_read" in role.capabilities
        assert "web_search" in role.capabilities
        assert role.max_iterations == 15
        assert role.risk_level == "low"

    def test_get_role_executor(self):
        role = get_role("executor")
        assert role.id == "executor"
        assert "file_write" in role.capabilities
        assert "bash_execute" in role.capabilities
        assert "file_read" in role.capabilities
        assert role.max_iterations == 20

    def test_suggest_role_research(self):
        assert suggest_role("analyze the codebase") == "researcher"

    def test_suggest_role_execute(self):
        assert suggest_role("run tests and fix failures") == "executor"
        assert suggest_role("build the project") == "executor"

    def test_suggest_role_chat(self):
        assert suggest_role("explain how this works") == "communicator"
        assert suggest_role("summarize the findings") == "communicator"

    def test_list_role_capabilities(self):
        caps = list_role_capabilities("researcher")
        assert isinstance(caps, list)
        assert all(isinstance(c, str) for c in caps)
        assert len(caps) > 0

    def test_unknown_role_raises(self):
        with pytest.raises(KeyError, match="Unknown agent role"):
            get_role("nonexistent")  # type: ignore[arg-type]
