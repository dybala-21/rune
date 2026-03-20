"""Tests for rune.agent.agent_context — shared pre/post-processing layer."""

from unittest.mock import AsyncMock, patch

import pytest

from rune.agent.agent_context import (
    ASSISTANT_TURN_SAVE_MAX_ATTEMPTS,
    ASSISTANT_TURN_SAVE_RETRY_DELAY_S,
    AgentContext,
    PostProcessInput,
    PrepareContextOptions,
    _sanitize_goal_input,
    post_process_agent_result,
)

# ---------------------------------------------------------------------------
# _sanitize_goal_input
# ---------------------------------------------------------------------------

class TestSanitizeGoalInput:
    def test_strips_invisible_control_chars(self):
        # \x7f is DEL control character
        result = _sanitize_goal_input("\x7fhello world")
        assert "\x7f" not in result
        assert "hello world" in result

    def test_preserves_newlines_and_tabs(self):
        result = _sanitize_goal_input("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_strips_leading_trailing_whitespace(self):
        result = _sanitize_goal_input("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        result = _sanitize_goal_input("")
        assert result == ""

    def test_preserves_normal_unicode(self):
        result = _sanitize_goal_input("fix the bug please")
        assert result == "fix the bug please"

    def test_preserves_korean_text(self):
        result = _sanitize_goal_input("로그인 버그 수정해줘")
        assert result == "로그인 버그 수정해줘"


# ---------------------------------------------------------------------------
# PrepareContextOptions / AgentContext dataclasses
# ---------------------------------------------------------------------------

class TestAgentContextDataclasses:
    def test_prepare_context_options_defaults(self):
        opts = PrepareContextOptions(goal="test")
        assert opts.channel == "tui"
        assert opts.cwd == ""
        assert opts.pinned_cwd is None

    def test_agent_context_defaults(self):
        ctx = AgentContext()
        assert ctx.goal == ""
        assert ctx.channel == "tui"
        assert ctx.token_budget == 500_000
        assert ctx.messages == []
        assert ctx.at_references == []

    def test_agent_context_with_values(self):
        ctx = AgentContext(
            goal="fix bug",
            workspace_root="/workspace",
            conversation_id="conv_123",
        )
        assert ctx.goal == "fix bug"
        assert ctx.workspace_root == "/workspace"
        assert ctx.conversation_id == "conv_123"


# ---------------------------------------------------------------------------
# post_process_agent_result
# ---------------------------------------------------------------------------

class TestPostProcessAgentResult:
    @pytest.mark.asyncio
    async def test_skips_when_no_answer(self):
        """Should return without attempting save when answer is empty."""
        inp = PostProcessInput(
            context=AgentContext(conversation_id="conv_1"),
            success=True,
            answer="",
        )
        # Should not raise
        with patch("rune.agent.memory_bridge.save_agent_result_to_memory", new_callable=AsyncMock) as mock_save:
            await post_process_agent_result(inp)
            mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Should retry save up to MAX_ATTEMPTS times."""
        mock_save = AsyncMock(side_effect=Exception("save failed"))
        inp = PostProcessInput(
            context=AgentContext(conversation_id="conv_1"),
            success=True,
            answer="result text",
        )
        with patch("rune.agent.memory_bridge.save_agent_result_to_memory", mock_save):
            await post_process_agent_result(inp)
            assert mock_save.call_count == ASSISTANT_TURN_SAVE_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        mock_save = AsyncMock()
        inp = PostProcessInput(
            context=AgentContext(conversation_id="conv_1"),
            success=True,
            answer="result text",
        )
        with patch("rune.agent.memory_bridge.save_agent_result_to_memory", mock_save):
            await post_process_agent_result(inp)
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_attempts_is_positive(self):
        assert ASSISTANT_TURN_SAVE_MAX_ATTEMPTS >= 1

    def test_retry_delay_is_small(self):
        assert ASSISTANT_TURN_SAVE_RETRY_DELAY_S > 0
        assert ASSISTANT_TURN_SAVE_RETRY_DELAY_S < 1.0
