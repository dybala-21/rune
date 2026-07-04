"""Server-side slash-command actions for the web app (TUI feature parity)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rune.api import command_actions
from rune.api.command_actions import ActionContext


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    command_actions._reset_for_tests()
    from rune.api import conversation_wiring

    conversation_wiring._reset_for_tests()

    broadcasts: list[tuple[str, dict]] = []

    async def _broadcast(event: str, data: dict) -> None:
        broadcasts.append((event, data))

    c = ActionContext(broadcast=_broadcast, workspace=tmp_path, session_id="s1")
    c.extra["broadcasts"] = broadcasts
    yield c
    command_actions._reset_for_tests()
    conversation_wiring._reset_for_tests()


# direct commands


async def test_help_lists_commands(ctx):
    out = await command_actions.handle_direct_command("/help", "")
    assert out and "/goal" in out and "/escalate" in out


async def test_model_shows_and_switches(ctx):
    out = await command_actions.handle_direct_command("/model", "")
    assert out and "Current model:" in out
    out = await command_actions.handle_direct_command("/model", "ollama:qwen2.5-coder:7b")
    assert out is not None and "Switched to ollama:" in out
    from rune.config import get_config

    assert get_config().llm.active_provider == "ollama"


async def test_non_direct_command_returns_none(ctx):
    assert await command_actions.handle_direct_command("/diff", "") is None


# memory


async def test_memory_add_show_clear_roundtrip(ctx):
    out = await command_actions.execute_action("memory:add:remember the port is 18789", ctx)
    assert "Added" in out
    out = await command_actions.execute_action("memory:show", ctx)
    assert "18789" in out
    out = await command_actions.execute_action("memory:clear", ctx)
    assert "cleared" in out.lower()


async def test_memory_add_requires_text(ctx):
    out = await command_actions.execute_action("memory:add", ctx)
    assert "Usage" in out


# git diff


async def test_git_diff_reports_changes(ctx, tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    out = await command_actions.execute_action("toggle_git_diff", ctx)
    assert "No uncommitted changes" in out
    f.write_text("two\n")
    out = await command_actions.execute_action("toggle_git_diff", ctx)
    assert "diff" in out and "two" in out


# undo / files (stubbed tracker — the real one is a filesystem watcher)


class _StubTracker:
    def __init__(self, workspace: Path):
        self._workspace = str(workspace)
        self._changed_files = {"b.txt"}
        self._snapshots = {"b.txt": "original\n"}

    def get_changed_files(self):
        return sorted(self._changed_files)


async def test_undo_restores_snapshot(ctx, tmp_path):
    (tmp_path / "b.txt").write_text("modified\n")
    command_actions._file_tracker = _StubTracker(tmp_path)
    out = await command_actions.execute_action("undo", ctx)
    assert "Reverted b.txt" in out
    assert (tmp_path / "b.txt").read_text() == "original\n"
    out = await command_actions.execute_action("undo", ctx)
    assert "Nothing to undo" in out


async def test_files_lists_changes(ctx, tmp_path):
    command_actions._file_tracker = _StubTracker(tmp_path)
    out = await command_actions.execute_action("toggle_files", ctx)
    assert "b.txt" in out


# search / sessions / load (against the real conversation store)


async def _seed_conversation(session_id: str) -> None:
    from rune.api import conversation_wiring

    manager = conversation_wiring.get_conv_manager()
    conv_id = await conversation_wiring.resolve_conversation(
        manager, session_id, sticky=False,
    )
    manager.add_turn(conv_id, "user", "the launch code is heron")
    manager.add_turn(conv_id, "assistant", "Noted.")
    await manager._store.save(manager._active[conv_id])


async def test_search_finds_turn(ctx):
    await _seed_conversation("s1")
    out = await command_actions.execute_action("search:heron", ctx)
    assert "heron" in out and "user" in out
    out = await command_actions.execute_action("search:zebra", ctx)
    assert "No matches" in out


async def test_sessions_lists_conversations(ctx):
    await _seed_conversation("s1")
    out = await command_actions.execute_action("show_sessions", ctx)
    assert "s1" in out and "(current)" in out


async def test_load_broadcasts_turns(ctx):
    await _seed_conversation("s2")
    out = await command_actions.execute_action("load:s2", ctx)
    assert out == ""  # structured result goes over the broadcast
    events = ctx.extra["broadcasts"]
    assert events and events[-1][0] == "command_result"
    data = events[-1][1]["data"]
    assert data["action"] == "load_session" and data["sessionId"] == "s2"
    assert any("heron" in t["content"] for t in data["turns"])


async def test_load_missing_conversation(ctx):
    out = await command_actions.execute_action("load:nope", ctx)
    assert "not found" in out.lower()


# status / config / escalate / goal guards


async def test_status_and_config(ctx):
    out = command_actions._action_status(ctx)
    assert "Agent status" in out
    out = command_actions._action_config()
    assert "Model:" in out


async def test_escalate_without_provider_is_helpful(ctx, monkeypatch):
    from rune.config import get_config

    monkeypatch.setattr(get_config().llm, "escalation_provider", None)
    out = await command_actions.execute_action("escalate:do a thing", ctx)
    assert "No escalation model configured" in out


async def test_goal_disabled_message(ctx, monkeypatch):
    from rune.config import get_config

    monkeypatch.setattr(get_config().goal_loop, "enabled", False)
    out = await command_actions.execute_action("goal_loop:make tests pass", ctx)
    assert "disabled" in out


async def test_unknown_client_action_degrades_gracefully(ctx):
    out = await command_actions.execute_action("cycle_style", ctx)
    assert "not available here" in out
