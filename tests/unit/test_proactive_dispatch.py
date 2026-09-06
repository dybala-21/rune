"""Identity, crash boundaries and concurrent delivery through the real bridge."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from rune.proactive.bridge import BridgeConfig, ExecutionStatus, ProactiveAgentBridge
from rune.proactive.engine import ProactiveEngine
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.types import Suggestion


@pytest.fixture
def engine(monkeypatch):
    e = ProactiveEngine()
    monkeypatch.setattr(ProactiveEngine, "_gather_context", AsyncMock(return_value={}))
    monkeypatch.setattr(ProactiveEngine, "_filter_candidates", lambda self, candidates: candidates)
    return e


@pytest.mark.asyncio
async def test_real_engine_events_and_poll_start_once(engine, monkeypatch):
    suggestion = Suggestion(title="Prepare report", confidence=.9)
    monkeypatch.setattr(ProactiveEngine, "_generate_candidates", AsyncMock(return_value=[suggestion]))
    factory = AsyncMock(return_value={"success": True})
    bridge = ProactiveAgentBridge(engine, factory, BridgeConfig(auto_execute=True))
    bridge._running = True
    await bridge._poll_once()
    await asyncio.sleep(0)
    assert factory.await_count == 1
    assert len(bridge.history) == 1
    bridge.stop()


@pytest.mark.asyncio
async def test_concurrent_accepts_share_result_and_reject_changed_payload(engine):
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def factory(goal, **kwargs):
        calls.append(goal)
        entered.set()
        await finish.wait()
        return {"success": True}

    bridge = ProactiveAgentBridge(engine, factory)
    s = Suggestion(title="Prepare report")
    assert (await bridge.execute_suggestion(s)).status == ExecutionStatus.DELIVERED
    first = asyncio.create_task(bridge.execute_suggestion(s, force=True))
    await entered.wait()
    second = asyncio.create_task(bridge.execute_suggestion(s, force=True))
    conflict = await bridge.execute_suggestion(replace(s, title="Delete report"), force=True)
    assert conflict.status == ExecutionStatus.SKIPPED
    finish.set()
    a, b = await asyncio.gather(first, second)
    assert a is b
    assert len(calls) == 1
    await bridge.execute_suggestion(s, force=True)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_persisted_result_survives_restart(engine, tmp_path):
    path = tmp_path / "operations.db"
    factory = AsyncMock(return_value={"success": True})
    s = Suggestion(title="Prepare report")
    store = ExecutionStore(path)
    first = ProactiveAgentBridge(engine, factory, execution_store=store)
    result = await first.execute_suggestion(s, force=True)
    first.stop()
    store.close()
    reopened = ExecutionStore(path)
    second = ProactiveAgentBridge(engine, factory, execution_store=reopened)
    assert await second.execute_suggestion(s, force=True) == result
    assert factory.await_count == 1
    second.stop()
    reopened.close()


@pytest.mark.asyncio
async def test_interrupted_claim_is_not_replayed(engine, tmp_path):
    started = asyncio.Event()

    async def factory(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    path = tmp_path / "operations.db"
    store = ExecutionStore(path)
    bridge = ProactiveAgentBridge(engine, factory, execution_store=store)
    s = Suggestion(title="External update")
    waiter = asyncio.create_task(bridge.execute_suggestion(s, force=True))
    await started.wait()
    bridge.stop()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    store.close()
    reopened = ExecutionStore(path)
    retry = AsyncMock(return_value={"success": True})
    replacement = ProactiveAgentBridge(engine, retry, execution_store=reopened)
    result = await replacement.execute_suggestion(s, force=True)
    assert result.status == ExecutionStatus.UNVERIFIED
    retry.assert_not_called()
    reopened.close()


@pytest.mark.asyncio
async def test_failures_consume_budget_and_do_not_retry_by_default(engine):
    factory = AsyncMock(return_value={"success": False, "error": "outcome unknown"})
    bridge = ProactiveAgentBridge(
        engine, factory, BridgeConfig(auto_execute=True, max_executions_per_hour=2),
    )
    results = await asyncio.gather(*(bridge.execute_suggestion(
        Suggestion(title=f"Task {i}")
    ) for i in range(6)))
    assert factory.await_count == 2
    assert sum(r.status == ExecutionStatus.SKIPPED for r in results) == 4
    bridge.clear_history()
    assert bridge._is_rate_limited()


@pytest.mark.asyncio
async def test_start_stop_restores_one_listener_and_cancels_poll(engine, monkeypatch):
    monkeypatch.setattr(ProactiveEngine, "_generate_candidates", AsyncMock(return_value=[]))
    bridge = ProactiveAgentBridge(engine, AsyncMock())
    bridge.start()
    bridge.stop()
    bridge.start()
    assert len(engine._listeners["suggestion"]) == 1
    bridge.stop()
    await asyncio.sleep(0)
    assert engine._listeners["suggestion"] == []


def test_atomic_claim_and_budget_across_connections(tmp_path):
    path = tmp_path / "operations.db"
    a, b = ExecutionStore(path), ExecutionStore(path)
    assert a.claim("op", "payload", 1) == "claimed"
    assert b.claim("op", "payload", 1) == "exists"
    assert b.claim("another", "payload", 1) == "rate_limited"
    a.close()
    b.close()
