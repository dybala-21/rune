"""Unit tests for turn-atomic grouping and tool pair validation."""

from rune.agent.message_utils import group_into_turns, msg_role, validate_tool_pairs


def _asst(*tool_calls):
    """Assistant message with tool_calls."""
    return {"role": "assistant", "content": "", "tool_calls": list(tool_calls)}


def _tool(name="t"):
    return {"role": "tool", "content": f"result of {name}", "tool_call_id": name}


def _user(text="hi"):
    return {"role": "user", "content": text}


def _sys(text="system"):
    return {"role": "system", "content": text}


class TestGroupIntoTurns:
    def test_single_tool_call(self):
        msgs = [_user(), _asst("call1"), _tool("call1")]
        turns = group_into_turns(msgs)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"
        assert len(turns[1].messages) == 2  # assistant + 1 tool

    def test_multi_tool_call(self):
        msgs = [_user(), _asst("a", "b", "c"), _tool("a"), _tool("b"), _tool("c")]
        turns = group_into_turns(msgs)
        assert len(turns) == 2
        assert len(turns[1].messages) == 4  # assistant + 3 tools

    def test_consecutive_assistant_turns(self):
        msgs = [
            _user(),
            _asst("a"), _tool("a"),
            _asst("b", "c"), _tool("b"), _tool("c"),
        ]
        turns = group_into_turns(msgs)
        assert len(turns) == 3  # user, asst+1tool, asst+2tools
        assert len(turns[2].messages) == 3

    def test_system_messages_are_own_turn(self):
        msgs = [_sys(), _user(), _asst("a"), _tool("a"), _sys("nudge")]
        turns = group_into_turns(msgs)
        assert len(turns) == 4
        assert turns[0].role == "system"
        assert turns[3].role == "system"


class TestValidateToolPairs:
    def test_valid_pairs_unchanged(self):
        msgs = [_user(), _asst("a", "b"), _tool("a"), _tool("b")]
        assert validate_tool_pairs(msgs) == msgs

    def test_orphaned_tool_dropped(self):
        msgs = [_sys("summary"), _tool("orphan1"), _tool("orphan2")]
        result = validate_tool_pairs(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_mixed_valid_and_orphaned(self):
        msgs = [
            _tool("orphan"),  # orphan at start
            _asst("a"), _tool("a"),  # valid pair
            _tool("orphan2"),  # orphan after pair consumed
        ]
        result = validate_tool_pairs(msgs)
        assert len(result) == 2  # orphan dropped, [asst, tool] kept
        assert result[0]["role"] == "assistant"
        assert result[1]["tool_call_id"] == "a"

    def test_missing_tool_result_synthesized(self):
        msgs = [
            _asst("a", "b"),
            _tool("a"),
            _user("next turn"),
        ]
        result = validate_tool_pairs(msgs)
        assert [msg_role(m) for m in result] == ["assistant", "tool", "tool", "user"]
        assert result[1]["tool_call_id"] == "a"
        assert result[2]["tool_call_id"] == "b"
        assert "omitted" in result[2]["content"]

    def test_empty_list(self):
        assert validate_tool_pairs([]) == []

    def test_no_tools(self):
        msgs = [_user(), _asst(), _user("follow up")]
        # assistant without tool_calls
        result = validate_tool_pairs(msgs)
        assert len(result) == 3


class TestCompactMessagesAtomic:
    """Test via NativeAgentLoop._compact_messages_atomic."""

    def test_multi_tool_preserved(self):
        from rune.agent.loop import NativeAgentLoop
        msgs = [
            _user(),
            _asst("a", "b", "c"), _tool("a"), _tool("b"), _tool("c"),
            _user("next"),
        ]
        # keep_last=2 should keep [asst+3tools, user] = 2 turns
        result = NativeAgentLoop._compact_messages_atomic(msgs, keep_last=2)
        roles = [msg_role(m) for m in result]
        assert roles == ["assistant", "tool", "tool", "tool", "user"]

    def test_keep_all_when_small(self):
        from rune.agent.loop import NativeAgentLoop
        msgs = [_user(), _asst("a"), _tool("a")]
        result = NativeAgentLoop._compact_messages_atomic(msgs, keep_last=10)
        assert len(result) == 3

    def test_compaction_keeps_turn_boundaries_when_only_latest_turn_survives(self):
        from rune.agent.loop import NativeAgentLoop
        msgs = [
            _user("old"),
            _asst("x", "y"), _tool("x"), _tool("y"),
            _user("recent"),
        ]
        result = NativeAgentLoop._compact_messages_atomic(msgs, keep_last=1)
        assert len(result) == 1
        assert result[0]["content"] == "recent"

    def test_token_trim_preserves_recent_multi_tool_turn(self, monkeypatch):
        from rune.agent.loop import NativeAgentLoop

        loop = NativeAgentLoop()
        loop._activity_phase = "exploration"
        monkeypatch.setitem(loop._TOKEN_CAPS, "exploration", 220)
        monkeypatch.setattr(loop, "_get_budget_proportional_cap", lambda: 220)
        msgs = [
            _user("goal"),
            _user("old " * 2000),
            _user("mid"),
            _user("recent"),
            _asst("a", "b", "c"),
            _tool("a"),
            _tool("b"),
            _tool("c"),
        ]

        result = loop._trim_to_token_cap(msgs)
        roles = [msg_role(m) for m in result[-4:]]
        assert roles == ["assistant", "tool", "tool", "tool"]
        assert [m.get("tool_call_id") for m in result[-3:]] == ["a", "b", "c"]
