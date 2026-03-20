"""Tests for rune.ui.sessions — save, load, list, delete sessions."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from rune.ui.sessions import (
    SerializedMessage,
    SerializedToolCallBlock,
    delete_session,
    list_sessions,
    load_session,
    save_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(**kwargs) -> SerializedMessage:
    defaults = dict(id="msg-1", role="user", content="hello", timestamp="2026-01-15T10:00:00Z")
    defaults.update(kwargs)
    return SerializedMessage(**defaults)


def _tc(**kwargs) -> SerializedToolCallBlock:
    defaults = dict(id="tc-1", action="file.read foo.ts", observation="contents", success=True, timestamp="2026-01-15T10:02:00Z", capability="file.read")
    defaults.update(kwargs)
    return SerializedToolCallBlock(**defaults)


# ---------------------------------------------------------------------------
# save_session
# ---------------------------------------------------------------------------


class TestSaveSession:
    @pytest.mark.asyncio
    async def test_creates_session_file(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            messages = [
                _msg(id="u1", role="user", content="fix the bug"),
                _msg(id="a1", role="assistant", content="Done!"),
            ]
            tool_blocks = [_tc()]

            result = await save_session("sess-abc", "My Session", messages, tool_blocks)

            assert result.exists()
            data = json.loads(result.read_text())
            assert data["id"] == "sess-abc"
            assert data["name"] == "My Session"
            assert data["message_count"] == 2  # user + assistant
            assert data["tool_call_count"] == 1
            assert len(data["messages"]) == 2
            assert len(data["tool_call_blocks"]) == 1

    @pytest.mark.asyncio
    async def test_counts_only_user_and_assistant(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            messages = [
                _msg(id="s1", role="system", content="sys"),
                _msg(id="u1", role="user", content="q"),
                _msg(id="t1", role="thinking", content="..."),
                _msg(id="a1", role="assistant", content="a"),
            ]
            result = await save_session("sess-count", "Count", messages, [])
            data = json.loads(result.read_text())
            assert data["message_count"] == 2
            assert len(data["messages"]) == 4

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            result = await save_session("sess-empty", "Empty", [], [])
            data = json.loads(result.read_text())
            assert data["message_count"] == 0
            assert data["messages"] == []


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_lists_saved_sessions(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            await save_session("s1", "First", [_msg(role="user", content="hello")], [])
            items = await list_sessions()
            assert len(items) == 1
            assert items[0].id == "s1"
            assert items[0].name == "First"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_sessions(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            items = await list_sessions()
            assert items == []


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------


class TestLoadSession:
    @pytest.mark.asyncio
    async def test_loads_existing_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            await save_session("s1", "Test", [_msg()], [_tc()])
            session = await load_session("s1")
            assert session is not None
            assert session.id == "s1"
            assert len(session.messages) == 1
            assert len(session.tool_call_blocks) == 1

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            sessions_dir.mkdir(parents=True, exist_ok=True)
            assert await load_session("nonexistent") is None


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_deletes_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            await save_session("s1", "Test", [_msg()], [])
            assert await delete_session("s1") is True
            assert await load_session("s1") is None

    @pytest.mark.asyncio
    async def test_returns_false_for_missing(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        with patch("rune.ui.sessions.SESSIONS_DIR", sessions_dir):
            assert await delete_session("nonexistent") is False
