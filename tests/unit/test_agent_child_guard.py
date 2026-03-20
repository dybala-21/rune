"""Tests for rune.utils.agent_child_guard — ported from agent-child-guard.test.ts."""

import pytest

from rune.utils.agent_child_guard import (
    agent_child_guard,
    get_agent_child_guard_depth,
    is_agent_child_guard_active,
    pop_agent_child_guard,
    push_agent_child_guard,
    with_agent_child_guard_async,
)


class TestAgentChildGuard:
    """Tests for agent child guard env helpers."""

    def test_increments_and_decrements_depth(self):
        env: dict[str, str] = {}

        assert get_agent_child_guard_depth(env) == 0
        assert is_agent_child_guard_active(env) is False

        assert push_agent_child_guard(env) == 1
        assert get_agent_child_guard_depth(env) == 1
        assert is_agent_child_guard_active(env) is True

        assert push_agent_child_guard(env) == 2
        assert get_agent_child_guard_depth(env) == 2

        assert pop_agent_child_guard(env) == 1
        assert get_agent_child_guard_depth(env) == 1
        assert is_agent_child_guard_active(env) is True

        assert pop_agent_child_guard(env) == 0
        assert get_agent_child_guard_depth(env) == 0
        assert is_agent_child_guard_active(env) is False

    @pytest.mark.asyncio
    async def test_restores_depth_when_function_completes(self):
        env: dict[str, str] = {}

        async def inner():
            assert get_agent_child_guard_depth(env) == 1
            return 42

        result = await with_agent_child_guard_async(inner, env)

        assert result == 42
        assert get_agent_child_guard_depth(env) == 0

    @pytest.mark.asyncio
    async def test_restores_depth_even_when_function_throws(self):
        env: dict[str, str] = {}

        async def inner():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await with_agent_child_guard_async(inner, env)

        assert get_agent_child_guard_depth(env) == 0
        assert is_agent_child_guard_active(env) is False

    def test_context_manager_pushes_and_pops(self):
        env: dict[str, str] = {}

        with agent_child_guard(env) as depth:
            assert depth == 1
            assert get_agent_child_guard_depth(env) == 1

        assert get_agent_child_guard_depth(env) == 0

    def test_context_manager_restores_on_exception(self):
        env: dict[str, str] = {}

        with pytest.raises(ValueError):
            with agent_child_guard(env):
                raise ValueError("test error")

        assert get_agent_child_guard_depth(env) == 0

    def test_nested_context_managers(self):
        env: dict[str, str] = {}

        with agent_child_guard(env) as d1:
            assert d1 == 1
            with agent_child_guard(env) as d2:
                assert d2 == 2
                assert get_agent_child_guard_depth(env) == 2
            assert get_agent_child_guard_depth(env) == 1

        assert get_agent_child_guard_depth(env) == 0

    def test_pop_does_not_go_below_zero(self):
        env: dict[str, str] = {}
        result = pop_agent_child_guard(env)
        assert result == 0
        assert get_agent_child_guard_depth(env) == 0

    def test_handles_invalid_env_values(self):
        env: dict[str, str] = {"RUNE_AGENT_CHILD_GUARD_DEPTH": "not_a_number"}
        assert get_agent_child_guard_depth(env) == 0

    def test_handles_negative_env_values(self):
        env: dict[str, str] = {"RUNE_AGENT_CHILD_GUARD_DEPTH": "-5"}
        assert get_agent_child_guard_depth(env) == 0
