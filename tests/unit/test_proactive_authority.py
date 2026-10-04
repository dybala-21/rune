"""Execution evidence cannot manufacture user consent."""

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from rune.agent.autonomous import AutonomousExecution, AutonomousExecutor
from rune.memory.store import MemoryStore
from rune.proactive.bridge import BridgeConfig, ProactiveAgentBridge
from rune.proactive.engine import ProactiveEngine
from rune.proactive.types import Suggestion


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_execution_does_not_learn_user_feedback(success, monkeypatch):
    engine = ProactiveEngine()
    suggestion = Suggestion(title="Verify report")
    engine.add_suggestion(suggestion)
    autonomy, feedback, reflexion = AutonomousExecutor(), Mock(), Mock()
    monkeypatch.setattr("rune.proactive.reflexion.get_reflexion_learner", lambda: reflexion)
    bridge = ProactiveAgentBridge(
        engine, AsyncMock(return_value={"success": success, "verified": success}),
        BridgeConfig(auto_execute=True), feedback_learner=feedback, autonomous_executor=autonomy,
    )
    await bridge.execute_suggestion(suggestion)
    assert engine.get_stats()["interaction_count"] == 0
    assert engine.get_first_pending() is None
    assert autonomy.pattern_stats == {}
    assert len(autonomy.execution_history) == 1
    feedback.record_feedback.assert_not_called()
    reflexion.record_rejection.assert_not_called()
    reflexion.record_task_outcome.assert_called_once()


def test_only_explicit_execution_feedback_updates_autonomy():
    executor = AutonomousExecutor()
    execution = AutonomousExecution(id="run", domain="file", action="read", success=True)
    for _ in range(10):
        executor.record_execution(execution)
    assert executor.pattern_stats == {}
    executor.record_execution(replace(execution, user_feedback="approved"))
    assert executor.pattern_stats["file:read"].approved == 1


@pytest.mark.asyncio
async def test_cross_process_accept_preserves_contract_and_executes_once(tmp_path, monkeypatch):
    monkeypatch.setattr("rune.proactive.reflexion.get_reflexion_learner", lambda: Mock())
    store = MemoryStore(tmp_path / "memory.db")
    daemon, api = ProactiveEngine(), ProactiveEngine()
    daemon.load_persisted_suggestions(store)
    suggestion = Suggestion(title="Prepare report", verification=["pytest tests/test_report.py"])
    daemon.add_suggestion(suggestion)
    api.load_persisted_suggestions(store)
    assert api.handle_response(suggestion.id, True)
    assert api.handle_response(suggestion.id, True)
    daemon.save_suggestions(store)
    assert len(store.get_suggestion_state()) == 1
    daemon.load_persisted_suggestions(store)
    restored = daemon.get_suggestion(suggestion.id)
    assert restored.status == "accepted"
    assert restored.verification == suggestion.verification
    with pytest.raises(ValueError, match="different response"):
        api.handle_response(suggestion.id, False)
    assert not api.handle_response("unknown", True)
    with pytest.raises(ValueError, match="new ID"):
        daemon.add_suggestion(replace(suggestion, title="Different action"))

    monkeypatch.setattr(ProactiveEngine, "evaluate", AsyncMock(return_value=[]))
    factory = AsyncMock(return_value={"success": True, "verified": True})
    bridge = ProactiveAgentBridge(daemon, factory)
    await bridge._poll_once()
    await bridge._poll_once()
    factory.assert_awaited_once()
    assert factory.call_args.kwargs["verification"] == suggestion.verification
    restarted = ProactiveEngine()
    restarted.load_persisted_suggestions(store)
    assert restarted.get_suggestion(suggestion.id).execution_status == "success"
    assert restarted.get_stats()["interaction_count"] == 1
    store.close()


@pytest.mark.asyncio
async def test_legacy_acceptance_neither_runs_nor_trains_feedback(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.save_suggestion_state("insight", "accepted", {"suggestion_id": "old", "title": "Legacy"})
    engine = ProactiveEngine()
    engine.load_persisted_suggestions(store)
    assert engine.get_stats()["interaction_count"] == 0
    factory = AsyncMock()
    bridge = ProactiveAgentBridge(engine, factory)
    await bridge._dispatch_accepted()
    factory.assert_not_called()
    assert engine.handle_response("old", False)
    engine.load_persisted_suggestions(store)
    assert engine.get_suggestion("old").status == "dismissed"
    engine.delete_suggestion("old")
    engine.load_persisted_suggestions(store)
    assert engine.get_suggestion("old") is None
    store.close()
