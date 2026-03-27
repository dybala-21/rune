"""Tests for tool_call_policy — weak model guardrails."""

from __future__ import annotations

from rune.agent.tool_call_policy import ToolCallPolicy


class TestShouldForceTool:
    """Test tool_choice='required' retry logic."""

    def test_force_on_text_without_tools(self):
        p = ToolCallPolicy()
        assert p.should_force_tool(has_tool_calls=False, has_text=True) is True

    def test_no_force_when_tools_called(self):
        p = ToolCallPolicy()
        assert p.should_force_tool(has_tool_calls=True, has_text=True) is False

    def test_no_force_without_text(self):
        p = ToolCallPolicy()
        assert p.should_force_tool(has_tool_calls=False, has_text=False) is False

    def test_max_one_retry(self):
        p = ToolCallPolicy()
        assert p.should_force_tool(has_tool_calls=False, has_text=True) is True
        assert p.should_force_tool(has_tool_calls=False, has_text=True) is False

    def test_disabled(self):
        p = ToolCallPolicy(force_tool_on_empty=False)
        assert p.should_force_tool(has_tool_calls=False, has_text=True) is False

    def test_reset_clears_counter(self):
        p = ToolCallPolicy()
        p.should_force_tool(has_tool_calls=False, has_text=True)  # count=1
        p.reset()
        assert p.should_force_tool(has_tool_calls=False, has_text=True) is True


class TestRecordToolCall:
    """Test consecutive tool call loop detection."""

    def test_no_nudge_for_varied_tools(self):
        p = ToolCallPolicy()
        for tool in ["file_read", "file_edit", "bash", "file_read"]:
            assert p.record_tool_call(tool) is None

    def test_nudge_after_consecutive(self):
        p = ToolCallPolicy(max_consecutive_same_tool=3)
        assert p.record_tool_call("file_search") is None
        assert p.record_tool_call("file_search") is None
        nudge = p.record_tool_call("file_search")
        assert nudge is not None
        assert "file_search" in nudge
        assert "3" in nudge

    def test_default_threshold_is_5(self):
        p = ToolCallPolicy()
        for _ in range(4):
            assert p.record_tool_call("file_read") is None
        assert p.record_tool_call("file_read") is not None

    def test_reset_clears_history(self):
        p = ToolCallPolicy(max_consecutive_same_tool=3)
        p.record_tool_call("x")
        p.record_tool_call("x")
        p.reset()
        assert p.record_tool_call("x") is None  # history cleared

    def test_different_tool_breaks_streak(self):
        p = ToolCallPolicy(max_consecutive_same_tool=3)
        p.record_tool_call("file_search")
        p.record_tool_call("file_search")
        p.record_tool_call("file_read")  # breaks streak
        assert p.record_tool_call("file_search") is None  # count resets


class TestGetExtraParams:

    def test_default_empty(self):
        p = ToolCallPolicy()
        assert p.get_extra_params() == {}

    def test_disable_parallel(self):
        p = ToolCallPolicy(disable_parallel=True)
        assert p.get_extra_params() == {"parallel_tool_calls": False}
