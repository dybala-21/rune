"""Multi-turn conversation wiring on the API/web server paths.

Regression tests for the defect where every web message ran stateless:
POST /api/message and the execute endpoints never recorded turns nor passed
message_history to the agent loop, so a turn-2 follow-up had no context.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


# Helpers


class FakeLoop:
    """Stands in for NativeAgentLoop; records what run() receives."""

    captured: list[dict] = []

    def __init__(self, config=None, **kwargs) -> None:
        self._last_answer_text = ""
        self._last_goal_type = ""

    def set_approval_callback(self, cb) -> None:
        pass

    def set_ask_user_callback(self, cb) -> None:
        pass

    def on(self, event, cb) -> None:
        pass

    async def cancel(self) -> None:
        pass

    async def run(self, goal, context=None, message_history=None, **kwargs):
        FakeLoop.captured.append({
            "goal": goal,
            "history": list(message_history or []),
        })
        self._last_answer_text = f"answer to: {goal}"
        self._last_goal_type = "chat"
        return SimpleNamespace(
            reason="completed", final_step=1, evidence_gate=None,
        )


async def _noop_post_process(_input):
    return None


@pytest.fixture
def isolated_wiring(tmp_path, monkeypatch):
    """Point RUNE_HOME at a tmp dir and reset the wiring singletons."""
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    from rune.api import conversation_wiring

    conversation_wiring._reset_for_tests()
    FakeLoop.captured = []
    yield conversation_wiring
    conversation_wiring._reset_for_tests()


def _turn_rows(db_path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT role, content FROM turns ORDER BY created_order"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# conversation_wiring unit tests


async def test_sticky_conversation_reused_before_first_save(isolated_wiring):
    """A second fast message must reuse the in-flight sticky conversation even
    though nothing has been persisted yet (start_conversation is in-memory)."""
    cw = isolated_wiring
    manager = cw.get_conv_manager()
    assert manager is not None

    conv_id_1 = await cw.resolve_conversation(manager, None, sticky=True)
    conv_id_2 = await cw.resolve_conversation(manager, None, sticky=True)
    assert conv_id_1 and conv_id_1 == conv_id_2


async def test_sticky_conversation_rotates_after_idle(isolated_wiring):
    cw = isolated_wiring
    manager = cw.get_conv_manager()

    conv_id_1 = await cw.resolve_conversation(manager, None, sticky=True)
    manager._active[conv_id_1].updated_at = datetime.now() - timedelta(hours=2)

    conv_id_2 = await cw.resolve_conversation(manager, None, sticky=True)
    assert conv_id_2 != conv_id_1


async def test_non_sticky_without_session_stays_stateless(isolated_wiring):
    """Headless execute callers must not fall into the human's web chat."""
    cw = isolated_wiring
    manager = cw.get_conv_manager()
    assert await cw.resolve_conversation(manager, None, sticky=False) is None


async def test_explicit_session_loads_persisted_turns(isolated_wiring, tmp_path):
    cw = isolated_wiring
    manager = cw.get_conv_manager()

    conv_id = await cw.resolve_conversation(manager, "web_s1", sticky=False)
    assert conv_id == "web_s1"
    cw.record_user_turn(manager, conv_id, "my number is 47")
    fake = FakeLoop()
    fake._last_answer_text = "noted"
    await cw.record_assistant_turn(manager, conv_id, fake, "")

    # Fresh manager (server restart): explicit session resolves from disk.
    cw._reset_for_tests()
    manager2 = cw.get_conv_manager()
    conv_id2 = await cw.resolve_conversation(manager2, "web_s1", sticky=False)
    turns = manager2._active[conv_id2].turns
    assert [t.role for t in turns] == ["user", "assistant"]
    assert "47" in turns[0].content


async def test_resolve_does_not_clobber_active_conversation(isolated_wiring):
    """DB reload must not replace the in-memory object mid-run (a concurrent
    run's freshly added turns live only there)."""
    cw = isolated_wiring
    manager = cw.get_conv_manager()

    conv_id = await cw.resolve_conversation(manager, "web_s2", sticky=False)
    cw.record_user_turn(manager, conv_id, "unsaved turn")
    obj_before = manager._active[conv_id]

    await cw.resolve_conversation(manager, "web_s2", sticky=False)
    assert manager._active[conv_id] is obj_before
    assert len(manager._active[conv_id].turns) == 1


# /api/message end-to-end through the FastAPI app


@pytest.fixture
def client(isolated_wiring, monkeypatch, tmp_path):
    import rune.agent.agent_context as agent_context
    import rune.agent.loop as agent_loop_mod

    monkeypatch.setattr(agent_loop_mod, "NativeAgentLoop", FakeLoop)
    monkeypatch.setattr(
        agent_context, "post_process_agent_result", _noop_post_process,
    )

    async def _fixed_cwd(goal, requested_cwd, pinned_cwd=None, path_intent_model=None):
        return str(tmp_path)

    monkeypatch.setattr(
        agent_context, "_resolve_workspace_cwd_for_turn", _fixed_cwd,
    )

    from starlette.testclient import TestClient

    from rune.api.server import create_app

    app = create_app()
    # Loopback client address so the localhost auth bypass applies.
    with TestClient(app, client=("127.0.0.1", 50000)) as tc:
        yield tc


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_api_message_threads_history_across_turns(client, tmp_path):
    r1 = client.post(
        "/api/message",
        json={"text": "my favorite number is 47", "sessionId": "web_t1"},
    )
    assert r1.status_code == 200
    db = tmp_path / "conversations.db"
    assert _wait_for(lambda: len(_turn_rows(db)) >= 2), (
        "turns were not persisted after turn 1"
    )

    r2 = client.post(
        "/api/message",
        json={"text": "what is my favorite number?", "sessionId": "web_t1"},
    )
    assert r2.status_code == 200
    assert _wait_for(lambda: len(FakeLoop.captured) >= 2)

    history = FakeLoop.captured[1]["history"]
    assert history, "turn 2 received no message history"
    joined = str(history)
    assert "47" in joined
    assert "answer to:" in joined

    assert _wait_for(lambda: len(_turn_rows(db)) >= 4)
    roles = [r for r, _ in _turn_rows(db)]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_api_message_sticky_without_session_id(client, tmp_path):
    r1 = client.post("/api/message", json={"text": "sticky turn one"})
    assert r1.status_code == 200
    assert _wait_for(lambda: len(FakeLoop.captured) >= 1)

    r2 = client.post("/api/message", json={"text": "sticky turn two"})
    assert r2.status_code == 200
    assert _wait_for(lambda: len(FakeLoop.captured) >= 2)
    assert _wait_for(
        lambda: any(
            "sticky turn one" in str(c["history"]) for c in FakeLoop.captured
        ),
        timeout=5.0,
    ), "sticky web conversation did not carry turn-1 context"


def test_execute_without_session_is_stateless(client, tmp_path):
    r1 = client.post(
        "/api/v1/agent/execute",
        json={"goal": "headless one", "stream": False},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/api/v1/agent/execute",
        json={"goal": "headless two", "stream": False},
    )
    assert r2.status_code == 200
    assert len(FakeLoop.captured) == 2
    assert FakeLoop.captured[1]["history"] == []
    # Nothing persisted for session-less headless calls
    assert len(_turn_rows(tmp_path / "conversations.db")) == 0


def test_execute_with_session_threads_history(client, tmp_path):
    r1 = client.post(
        "/api/v1/agent/execute",
        json={"goal": "the code word is heron", "stream": False,
              "session_id": "web_x1"},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/api/v1/agent/execute",
        json={"goal": "what is the code word?", "stream": False,
              "session_id": "web_x1"},
    )
    assert r2.status_code == 200
    assert "heron" in str(FakeLoop.captured[1]["history"])


# /agent/run handler: server-generated sessionId must record turn 1.
# (The router is not mounted on create_app today, so this calls the handler
# directly rather than going through the app.)


async def test_agent_run_generated_session_records_first_turn(
    isolated_wiring, monkeypatch, tmp_path,
):
    import rune.agent.agent_context as agent_context
    import rune.agent.loop as agent_loop_mod

    monkeypatch.setattr(agent_loop_mod, "NativeAgentLoop", FakeLoop)
    monkeypatch.setattr(
        agent_context, "post_process_agent_result", _noop_post_process,
    )

    async def _fixed_cwd(goal, requested_cwd, pinned_cwd=None, path_intent_model=None):
        return str(tmp_path)

    monkeypatch.setattr(
        agent_context, "_resolve_workspace_cwd_for_turn", _fixed_cwd,
    )

    from rune.api.handlers.agent import AgentRunRequest, agent_run

    resp = await agent_run(AgentRunRequest(goal="hello there"))
    session_id = resp.session_id
    assert session_id

    db = tmp_path / "conversations.db"

    def _session_turns():
        conn = sqlite3.connect(str(db))
        try:
            return conn.execute(
                "SELECT role FROM turns WHERE conversation_id = ?",
                (session_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and len(_session_turns()) < 2:
        await asyncio.sleep(0.05)

    assert len(_session_turns()) >= 2, (
        "first turn was not recorded under the server-generated sessionId"
    )
