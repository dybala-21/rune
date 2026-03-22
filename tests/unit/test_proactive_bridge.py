"""Tests for proactive agent bridge."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from rune.proactive.bridge import (
    BridgeConfig,
    ExecutionRecord,
    ExecutionStatus,
    ProactiveAgentBridge,
    initialize_proactive_bridge,
)
from rune.proactive.engine import ProactiveEngine
from rune.proactive.types import Suggestion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(hints: list[dict] | None = None) -> ProactiveEngine:
    return ProactiveEngine({"min_confidence": 0.0})


def _make_suggestion(**overrides) -> Suggestion:
    defaults = {
        "title": "Test suggestion",
        "description": "Do something useful",
        "confidence": 0.8,
        "source": "test",
    }
    defaults.update(overrides)
    return Suggestion(**defaults)


async def _success_factory(goal: str) -> dict:
    return {"success": True}


async def _failure_factory(goal: str) -> dict:
    return {"success": False, "error": "Agent failed"}


async def _exception_factory(goal: str) -> dict:
    raise RuntimeError("Boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBridgeConfig:
    def test_defaults(self):
        cfg = BridgeConfig()
        assert cfg.poll_interval_seconds == 60.0
        assert cfg.max_retries == 2
        assert cfg.max_executions_per_hour == 5
        assert cfg.min_confidence == 0.5
        assert cfg.backoff_base_seconds == 2.0


class TestExecutionRecord:
    def test_record_fields(self):
        record = ExecutionRecord(
            suggestion_id="abc",
            suggestion_title="Test",
            status=ExecutionStatus.SUCCESS,
        )
        assert record.suggestion_id == "abc"
        assert record.status == ExecutionStatus.SUCCESS
        assert isinstance(record.timestamp, datetime)


class TestProactiveAgentBridge:
    def test_init(self):
        engine = _make_engine()
        bridge = ProactiveAgentBridge(engine, _success_factory)
        assert not bridge.is_running
        assert bridge.history == []

    @pytest.mark.asyncio
    async def test_execute_suggestion_success(self):
        engine = _make_engine()
        bridge = ProactiveAgentBridge(engine, _success_factory)

        suggestion = _make_suggestion()
        record = await bridge.execute_suggestion(suggestion)

        assert record.status == ExecutionStatus.SUCCESS
        assert record.attempt == 1
        assert record.error is None
        assert len(bridge.history) == 1

    @pytest.mark.asyncio
    async def test_execute_suggestion_failure_retries(self):
        engine = _make_engine()
        config = BridgeConfig(max_retries=2, backoff_base_seconds=0.01)
        bridge = ProactiveAgentBridge(engine, _failure_factory, config)

        suggestion = _make_suggestion()
        record = await bridge.execute_suggestion(suggestion)

        assert record.status == ExecutionStatus.FAILURE
        # Should have 3 records: 1 initial + 2 retries
        assert len(bridge.history) == 3
        assert bridge.history[-1].attempt == 3

    @pytest.mark.asyncio
    async def test_execute_suggestion_exception_retries(self):
        engine = _make_engine()
        config = BridgeConfig(max_retries=1, backoff_base_seconds=0.01)
        bridge = ProactiveAgentBridge(engine, _exception_factory, config)

        suggestion = _make_suggestion()
        record = await bridge.execute_suggestion(suggestion)

        assert record.status == ExecutionStatus.FAILURE
        assert "Boom" in record.error
        # 1 initial + 1 retry = 2 records
        assert len(bridge.history) == 2

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        engine = _make_engine()
        config = BridgeConfig(max_executions_per_hour=2)
        bridge = ProactiveAgentBridge(engine, _success_factory, config)

        # Execute 2 suggestions successfully
        for _ in range(2):
            await bridge.execute_suggestion(_make_suggestion())

        # Now we should be rate limited
        assert bridge._is_rate_limited() is True

    @pytest.mark.asyncio
    async def test_rate_limiting_not_triggered_below_limit(self):
        engine = _make_engine()
        config = BridgeConfig(max_executions_per_hour=5)
        bridge = ProactiveAgentBridge(engine, _success_factory, config)

        await bridge.execute_suggestion(_make_suggestion())
        assert bridge._is_rate_limited() is False

    @pytest.mark.asyncio
    async def test_start_stop(self):
        engine = _make_engine()
        config = BridgeConfig(poll_interval_seconds=100.0)
        bridge = ProactiveAgentBridge(engine, _success_factory, config)

        bridge.start()
        assert bridge.is_running is True

        bridge.stop()
        # Allow cancelled task to finalize
        await asyncio.sleep(0)
        assert bridge.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        engine = _make_engine()
        config = BridgeConfig(poll_interval_seconds=100.0)
        bridge = ProactiveAgentBridge(engine, _success_factory, config)

        bridge.start()
        bridge.start()  # should not raise
        assert bridge.is_running is True
        bridge.stop()
        await asyncio.sleep(0)

    def test_stop_when_not_running(self):
        engine = _make_engine()
        bridge = ProactiveAgentBridge(engine, _success_factory)
        bridge.stop()  # should not raise

    def test_get_history_filtered(self):
        engine = _make_engine()
        bridge = ProactiveAgentBridge(engine, _success_factory)

        # Manually add records
        bridge._history.append(ExecutionRecord(
            suggestion_id="a",
            suggestion_title="A",
            status=ExecutionStatus.SUCCESS,
        ))
        bridge._history.append(ExecutionRecord(
            suggestion_id="b",
            suggestion_title="B",
            status=ExecutionStatus.FAILURE,
            error="Err",
        ))
        bridge._history.append(ExecutionRecord(
            suggestion_id="c",
            suggestion_title="C",
            status=ExecutionStatus.SKIPPED,
        ))

        successes = bridge.get_history(status=ExecutionStatus.SUCCESS)
        assert len(successes) >= 1
        assert successes[0].suggestion_id == "a"

        failures = bridge.get_history(status=ExecutionStatus.FAILURE)
        assert len(failures) == 1

    def test_get_history_limit(self):
        engine = _make_engine()
        bridge = ProactiveAgentBridge(engine, _success_factory)

        for i in range(10):
            bridge._history.append(ExecutionRecord(
                suggestion_id=str(i),
                suggestion_title=f"S{i}",
                status=ExecutionStatus.SUCCESS,
            ))

        limited = bridge.get_history(limit=3)
        assert len(limited) == 3

    def test_clear_history(self):
        engine = _make_engine()
        bridge = ProactiveAgentBridge(engine, _success_factory)
        bridge._history.append(ExecutionRecord(
            suggestion_id="x",
            suggestion_title="X",
            status=ExecutionStatus.SUCCESS,
        ))
        bridge.clear_history()
        assert bridge.history == []


class TestInitializeProactiveBridge:
    def test_creates_bridge(self):
        engine = _make_engine()
        bridge = initialize_proactive_bridge(engine, _success_factory)
        assert isinstance(bridge, ProactiveAgentBridge)
        assert not bridge.is_running

    def test_replaces_existing_bridge(self):
        engine = _make_engine()
        bridge1 = initialize_proactive_bridge(engine, _success_factory)
        bridge2 = initialize_proactive_bridge(engine, _failure_factory)
        assert bridge1 is not bridge2

    @pytest.mark.asyncio
    async def test_stops_running_bridge_on_replace(self):
        engine = _make_engine()
        config = BridgeConfig(poll_interval_seconds=100.0)
        bridge1 = initialize_proactive_bridge(engine, _success_factory, config)
        bridge1.start()
        assert bridge1.is_running

        bridge2 = initialize_proactive_bridge(engine, _success_factory, config)
        await asyncio.sleep(0)  # let cancelled task finalize
        # Old bridge should have been stopped
        assert not bridge1.is_running
        assert bridge1 is not bridge2


class TestPollOnce:
    @pytest.mark.asyncio
    async def test_poll_executes_high_confidence_suggestions(self):
        engine = _make_engine()
        config = BridgeConfig(min_confidence=0.3)
        bridge = ProactiveAgentBridge(
            engine,
            _success_factory,
            config,
            context={
                "hints": [
                    {"title": "Do X", "description": "Desc X", "confidence": 0.9},
                ],
            },
        )
        await bridge._poll_once()
        successes = bridge.get_history(status=ExecutionStatus.SUCCESS)
        assert len(successes) >= 1

    @pytest.mark.asyncio
    async def test_poll_skips_low_confidence(self):
        engine = _make_engine()
        config = BridgeConfig(min_confidence=0.95)
        bridge = ProactiveAgentBridge(
            engine,
            _success_factory,
            config,
            context={
                "hints": [
                    {"title": "Low conf", "description": "Desc", "confidence": 0.5},
                ],
            },
        )
        await bridge._poll_once()
        skipped = bridge.get_history(status=ExecutionStatus.SKIPPED)
        assert len(skipped) >= 1
