"""Multi-turn conversation wiring on the API/web server paths.

Regression tests for the defect where every web message ran stateless:
POST /api/message and the execute endpoints never recorded turns nor passed
message_history to the agent loop, so a turn-2 follow-up had no context.
"""

from __future__ import annotations

import asyncio
import json
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


@pytest.fixture
def question_flow(client, monkeypatch):
    import rune.agent.loop as loop_module
    from rune.api.server import SseClientManager
    from rune.capabilities.ask_user import ask_user, set_ask_user_callback
    from rune.utils.fast_serde import json_encode

    frames, answers = [], []
    broadcast = SseClientManager.broadcast

    def capture(manager, event, data):
        json_encode(data)
        frames.append((event, data))
        return broadcast(manager, event, data)

    class QuestionLoop(FakeLoop):
        def set_ask_user_callback(self, cb):
            self.callback = cb

        async def run(self, goal, **kwargs):
            from rune.capabilities.ask_user import AskUserParams

            set_ask_user_callback(self.callback)
            result = await ask_user(AskUserParams(question=goal, reason="test",
                                   options=[{"label": "개념 설명"}, {"label": "시스템 확인"}]))
            answers.append((goal, result))
            self._last_answer_text = result.output or result.error
            return SimpleNamespace(reason="completed", final_step=1, evidence_gate=None)

    monkeypatch.setattr(loop_module, "NativeAgentLoop", QuestionLoop)
    monkeypatch.setattr(SseClientManager, "broadcast", capture)
    return client, frames, answers


@pytest.mark.parametrize("selected_key", ["selectedIndex", "selected_index"])
def test_web_question_selection_contract(question_flow, selected_key):
    client, frames, answers = question_flow
    client.post("/api/message", json={"text": "TCP 소켓 질문", "sessionId": "question-test"})
    assert _wait_for(lambda: any(event == "question" for event, _ in frames))
    payload = next(data for event, data in frames if event == "question")
    assert payload["question"] == "TCP 소켓 질문"
    assert payload["options"][1]["label"] == "시스템 확인"
    assert client.post("/api/question", json={"id": payload["id"], "answer": "wrong", selected_key: 9}).status_code == 422
    response = client.post("/api/question", json={"id": payload["id"], "answer": "시스템 확인", selected_key: 1})
    assert response.status_code == 200
    assert _wait_for(lambda: len(answers) == 1)
    result = answers[0][1]
    assert result.success and result.metadata["selectedIndex"] == 1
    assert result.metadata["freeText"] is False
    assert '시스템 확인' in result.output
    assert _wait_for(lambda: any(event == "question_closed" for event, _ in frames))
    assert client.post("/api/question", json={"id": payload["id"], "answer": "duplicate"}).status_code == 410


def test_web_questions_stay_separate_and_support_free_text(question_flow):
    client, frames, answers = question_flow
    for name in ["first", "second"]:
        client.post("/api/message", json={"text": name, "sessionId": name})
    assert _wait_for(lambda: sum(event == "question" for event, _ in frames) == 2)
    questions = {data["question"]: data for event, data in frames if event == "question"}
    for name in ["second", "first"]:
        assert client.post("/api/question", json={"id": questions[name]["id"], "answer": name + " 답변"}).status_code == 200
    assert _wait_for(lambda: len(answers) == 2)
    for name, result in answers:
        assert result.output == f'User responded: "{name} 답변"'
        assert result.metadata["freeText"] is True


def test_aborting_a_question_removes_its_pending_response(question_flow):
    client, frames, answers = question_flow
    client.post("/api/message", json={"text": "cancel question", "sessionId": "cancel-question"})
    assert _wait_for(lambda: any(event == "question" for event, _ in frames))
    question = next(data for event, data in frames if event == "question")
    assert client.post("/api/abort", json={"runId": question["runId"]}).status_code == 200
    assert _wait_for(lambda: any(event == "question_closed" for event, _ in frames))
    assert client.post("/api/question", json={"id": question["id"], "answer": "late"}).status_code == 410
    assert not answers


def test_ndjson_question_is_serializable_and_returns_a_typed_empty_response(question_flow):
    client, _, answers = question_flow
    response = client.post("/api/v1/agent/execute", json={"goal": "headless question", "stream": True})
    assert response.status_code == 200
    frames = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    payload = next(frame["data"] for frame in frames if frame["event"] == "question")
    assert payload["question"] == "headless question"
    assert payload["autonomous"] is True
    assert answers[0][1].success
    assert "skipped" in answers[0][1].output


def test_websocket_can_answer_a_web_question(question_flow):
    client, frames, answers = question_flow
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["event"] == "connected"
        client.post("/api/message", json={"text": "websocket question", "sessionId": "websocket-question"})
        assert _wait_for(lambda: any(event == "question" for event, _ in frames))
        payload = next(data for event, data in frames if event == "question")
        ws.send_json({"type": "question", "id": payload["id"], "answer": "개념 설명", "selectedIndex": 0})
        assert _wait_for(lambda: len(answers) == 1)
        assert answers[0][1].metadata["selectedIndex"] == 0
        assert answers[0][1].success


def test_expired_question_is_closed_without_inventing_an_answer(question_flow, monkeypatch):
    client, frames, answers = question_flow
    wait_for = asyncio.wait_for

    async def expire_question(awaitable, timeout):
        if timeout == 300.0:
            awaitable.cancel()
            raise TimeoutError
        return await wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", expire_question)
    client.post("/api/message", json={"text": "expired question", "sessionId": "expired-question"})
    assert _wait_for(lambda: len(answers) == 1)
    result = answers[0][1]
    assert not result.success
    assert "expired" in result.error
    payload = next(data for event, data in frames if event == "question")
    assert any(event == "question_closed" for event, _ in frames)
    assert client.post("/api/question", json={"id": payload["id"], "answer": "late"}).status_code == 410


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


@pytest.mark.parametrize("stream", [False, True])
def test_completed_web_turn_transports_separate_check_status(client, monkeypatch, stream):
    from rune.api.server import WsClientManager

    events = []

    async def capture(self, event, data):
        events.append({"event": event, "data": data})

    monkeypatch.setattr(WsClientManager, "broadcast", capture)
    if stream:
        response = client.post("/api/v1/agent/execute", json={"goal": "read a page", "stream": True})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    else:
        response = client.post("/api/message", json={"text": "read a page"})
        assert response.status_code == 200
        assert _wait_for(lambda: any(e["event"] == "agent_complete" for e in events))
    completion = next(e["data"] for e in events if e["event"] == "agent_complete")
    assert completion["success"] is True
    trust = completion["trust"]
    assert trust["completionStatus"] == "completed"
    assert trust["verificationStatus"] == "not_checked"
    assert trust["verified"] is False
    assert trust["canEscalate"] is False
    assert trust["honestNote"] == ""


def test_workspace_pin_flows_into_agent_run(client, tmp_path):
    """A pinned workspace is stored, listed in recents, and validated."""
    ws = tmp_path / "proj"
    ws.mkdir()
    r = client.post(
        "/api/v1/rpc",
        json={"method": "workspace.set",
              "params": {"sessionId": "web_ws1", "path": str(ws)}},
    )
    assert r.json()["success"] is True
    assert r.json()["data"]["path"] == str(ws.resolve())

    r = client.post(
        "/api/v1/rpc",
        json={"method": "workspace.get", "params": {"sessionId": "web_ws1"}},
    )
    assert r.json()["data"]["path"] == str(ws.resolve())

    r = client.post(
        "/api/v1/rpc",
        json={"method": "workspace.recents", "params": {}},
    )
    assert str(ws.resolve()) in r.json()["data"]["paths"]

    r = client.post(
        "/api/v1/rpc",
        json={"method": "workspace.set",
              "params": {"sessionId": "web_ws1", "path": "/nope/missing"}},
    )
    assert r.json()["success"] is False


def test_files_read_jailed_to_workspace(client, tmp_path):
    ws = tmp_path / "proj2"
    ws.mkdir()
    (ws / "hello.py").write_text("print('hi')\n")
    (tmp_path / "secret.txt").write_text("outside\n")
    client.post(
        "/api/v1/rpc",
        json={"method": "workspace.set",
              "params": {"sessionId": "web_ws2", "path": str(ws)}},
    )

    r = client.post(
        "/api/v1/rpc",
        json={"method": "files.read",
              "params": {"sessionId": "web_ws2", "path": "hello.py"}},
    )
    assert r.json()["success"] is True
    assert "print" in r.json()["data"]["content"]

    r = client.post(
        "/api/v1/rpc",
        json={"method": "files.read",
              "params": {"sessionId": "web_ws2", "path": "../secret.txt"}},
    )
    assert r.json()["success"] is False


def test_sessions_rpc_serves_canonical_store(client, tmp_path):
    """The sidebar RPC must share the id space of /load and live sessionIds."""
    r = client.post(
        "/api/message",
        json={"text": "sidebar test turn", "sessionId": "web_sb1"},
    )
    assert r.status_code == 200
    assert _wait_for(lambda: len(_turn_rows(tmp_path / "conversations.db")) >= 2)

    r = client.post("/api/v1/rpc", json={"method": "sessions.list", "params": {}})
    body = r.json()
    assert body["success"] is True
    ids = [s["id"] for s in body["data"]["sessions"]]
    assert "web_sb1" in ids
    entry = next(s for s in body["data"]["sessions"] if s["id"] == "web_sb1")
    assert entry["turnCount"] >= 2

    r = client.post(
        "/api/v1/rpc",
        json={"method": "sessions.turns", "params": {"sessionId": "web_sb1"}},
    )
    body = r.json()
    assert body["success"] is True
    turns = body["data"]["turns"]
    assert turns[0]["role"] == "user" and "sidebar test turn" in turns[0]["content"]

    r = client.post(
        "/api/v1/rpc",
        json={"method": "sessions.turns", "params": {"sessionId": "missing"}},
    )
    assert r.json()["success"] is False


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


async def test_tool_paths_anchor_to_workspace(tmp_path, monkeypatch):
    """Relative file params and bash cwd resolve inside workspace_root, not
    the daemon's process cwd."""
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set

    ws = tmp_path / "anchored"
    ws.mkdir()
    (ws / "inner.txt").write_text("anchored content\n")

    opts = ToolAdapterOptions(
        allowed_tools=["file_read", "bash_execute"],
        workspace_root=str(ws),
    )
    tools = build_tool_set(opts)

    out = await tools["file_read"].function(path="inner.txt")
    assert "anchored content" in str(out)

    out = await tools["bash_execute"].function(command="pwd")
    assert str(ws) in str(out)


def test_workspace_listdirs(client, tmp_path):
    (tmp_path / "proj_a").mkdir()
    (tmp_path / "proj_b").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x")
    r = client.post(
        "/api/v1/rpc",
        json={"method": "workspace.listdirs", "params": {"dir": str(tmp_path)}},
    )
    data = r.json()["data"]
    assert data["entries"] == ["proj_a", "proj_b"]  # dirs only, noise filtered
    assert data["dir"] == str(tmp_path.resolve())
    assert data["parent"] == str(tmp_path.resolve().parent)


def test_workspace_set_empty_session_errors_cleanly(client, tmp_path):
    ws = tmp_path / "p"
    ws.mkdir()
    r = client.post(
        "/api/v1/rpc",
        json={"method": "workspace.set", "params": {"sessionId": "", "path": str(ws)}},
    )
    body = r.json()
    assert body["success"] is False  # clean error, not a 500
    assert r.status_code == 200


def test_files_read_null_byte_clean_error(client, tmp_path):
    ws = tmp_path / "nb"
    ws.mkdir()
    client.post("/api/v1/rpc", json={"method": "workspace.set",
                "params": {"sessionId": "nb1", "path": str(ws)}})
    r = client.post("/api/v1/rpc", json={"method": "files.read",
                "params": {"sessionId": "nb1", "path": "a\x00b"}})
    body = r.json()
    assert body["success"] is False  # clean error, not a 500
    assert r.status_code == 200


def test_listdirs_bad_path_falls_back(client, tmp_path):
    r = client.post("/api/v1/rpc", json={"method": "workspace.listdirs",
                "params": {"dir": "/nonexistent/deeply/nested/xyz"}})
    body = r.json()
    # Non-existent path falls back to a listable dir (parent or home), no crash.
    assert body["success"] is True
    assert "entries" in body["data"]



def test_build_trust_payload_verified():
    from types import SimpleNamespace

    from rune.api.server import build_trust_payload
    trace = SimpleNamespace(reason="completed", evidence_gate={
        "has_check": True, "last_verdict": "pass",
        "verdict_counts": {"pass": 3}, "last_evidence": "12 passed",
    })
    p = build_trust_payload(trace)
    assert p["verified"] is True
    assert p["evidenceGate"]["hasCheck"] is True
    assert p["evidenceGate"]["verdictCounts"] == {"pass": 3}


def test_build_trust_payload_honest_failure():
    from types import SimpleNamespace

    from rune.api.server import build_trust_payload
    # max_gate_blocked = the verify-or-fail case: solution failed its tests, so
    # RUNE refuses to claim done. The honest note must say exactly that.
    trace = SimpleNamespace(reason="max_gate_blocked", evidence_gate=None)
    p = build_trust_payload(trace)
    assert p["verified"] is False
    assert p["reason"] == "max_gate_blocked"
    assert "won't claim a result I can't verify" in p["honestNote"]


def test_build_trust_payload_budget_exhausted():
    from types import SimpleNamespace

    from rune.api.server import build_trust_payload
    # A fast-lane run cut off at the tool-round cap completes "successfully"
    # but may omit work that never ran — the payload must carry the fact so
    # the UI can say "stopped at tool budget" instead of a clean Completed.
    trace = SimpleNamespace(
        reason="completed", evidence_gate=None, tool_budget_exhausted=True,
    )
    p = build_trust_payload(trace)
    assert p["verified"] is False  # Completion without check evidence is not verification.
    assert p["budgetExhausted"] is True

    trace2 = SimpleNamespace(reason="completed", evidence_gate=None)
    assert build_trust_payload(trace2)["budgetExhausted"] is False


def test_escalation_status_reports_config(client, monkeypatch):
    from rune.config import get_config
    cfg = get_config().llm

    monkeypatch.setattr(cfg, "escalation_provider", None)
    monkeypatch.setattr(cfg, "escalation_model", None)
    r = client.post("/api/v1/rpc", json={"method": "escalation.status", "params": {}})
    d0 = r.json()["data"]
    assert d0["enabled"] is False and d0["provider"] == "" and d0["isCloud"] is False

    monkeypatch.setattr(cfg, "escalation_provider", "anthropic")
    monkeypatch.setattr(cfg, "escalation_model", "claude-sonnet-4-5")
    r = client.post("/api/v1/rpc", json={"method": "escalation.status", "params": {}})
    d = r.json()["data"]
    assert d["enabled"] is True and d["isCloud"] is True and d["model"] == "claude-sonnet-4-5"

    monkeypatch.setattr(cfg, "escalation_provider", "ollama")
    r = client.post("/api/v1/rpc", json={"method": "escalation.status", "params": {}})
    assert r.json()["data"]["isCloud"] is False  # local model stays on-machine


def test_escalation_suggests_local_model_when_unconfigured(client, monkeypatch):
    import rune.agent.advisor.tiers as tiers
    from rune.config import get_config
    cfg = get_config().llm
    monkeypatch.setattr(cfg, "escalation_provider", None)
    monkeypatch.setattr(cfg, "escalation_model", None)
    monkeypatch.setattr(cfg, "active_provider", "ollama")
    monkeypatch.setattr(cfg, "active_model", "qwen2.5-coder:7b")

    # Stub the ollama lister so the test doesn't need a running daemon.
    async def _fake_suggest(prov, model, **kw):
        return "qwen2.5-coder:32b" if "7b" in model else None
    monkeypatch.setattr(tiers, "suggest_local_escalation", _fake_suggest)

    r = client.post("/api/v1/rpc", json={"method": "escalation.status", "params": {}})
    d = r.json()["data"]
    assert d["enabled"] is False
    assert d["suggestion"] == "qwen2.5-coder:32b"

    # Accepting it via escalation.set flips to enabled/local.
    r = client.post("/api/v1/rpc", json={"method": "escalation.set",
                    "params": {"provider": "ollama", "model": "qwen2.5-coder:32b"}})
    assert r.json()["data"]["model"] == "qwen2.5-coder:32b"
    assert cfg.escalation_provider == "ollama"


def test_suggest_local_escalation_single_jump():
    """No multi-rung ladder and no cloud: pick the one strongest LOCAL model."""
    import asyncio

    import rune.agent.advisor.tiers as tiers

    async def _fake_tags(*a, **k):
        class R:
            status_code = 200
            def json(self):
                return {"models": [
                    {"name": "qwen2.5-coder:3b"},
                    {"name": "qwen2.5-coder:7b"},
                    {"name": "qwen2.5-coder:32b"},
                    {"name": "qwen3-coder:480b-cloud"},  # cloud → excluded
                ]}
        return R()

    import httpx
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return await _fake_tags()
    orig = httpx.AsyncClient
    httpx.AsyncClient = _Client
    try:
        got = asyncio.run(tiers.suggest_local_escalation("ollama", "qwen2.5-coder:7b"))
    finally:
        httpx.AsyncClient = orig
    assert got == "qwen2.5-coder:32b"  # single largest local, cloud excluded


def test_artifact_download_uses_conversation_workspace(client, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    workspace = tmp_path / "selected"
    workspace.mkdir()
    payload = b"%PDF-1.7\nfixture\n%%EOF"
    (workspace / "검토 보고서.pdf").write_bytes(payload)
    lookup = AsyncMock(return_value=str(workspace))
    monkeypatch.setattr("rune.api.conversation_wiring.get_workspace", lookup)
    response = client.get("/api/v1/files/download", params={"sessionId": "saved-office", "path": "검토 보고서.pdf"})
    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("attachment;")
    lookup.assert_awaited_once_with("saved-office")


def test_artifact_download_rejects_escape_and_unknown_session(client, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    workspace = tmp_path / "selected"
    workspace.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"private")
    (workspace / "escape.pdf").symlink_to(outside)
    monkeypatch.setattr("rune.api.conversation_wiring.get_workspace", AsyncMock(return_value=str(workspace)))
    for path in ("../outside.pdf", str(outside), "escape.pdf"):
        response = client.get("/api/v1/files/download", params={"sessionId": "office", "path": path})
        assert response.status_code == 403
        assert b"private" not in response.content
    monkeypatch.setattr("rune.api.conversation_wiring.get_workspace", AsyncMock(return_value=None))
    response = client.get("/api/v1/files/download", params={"sessionId": "missing", "path": "outside.pdf"})
    assert response.status_code == 404
