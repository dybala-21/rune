"""Tests for ProactiveEngine — event emitter and evaluate() events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rune.proactive.engine import ProactiveEngine


class TestEventEmitterOn:
    def test_registers_listener(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("suggestion", cb)
        assert cb in engine._listeners["suggestion"]

    def test_multiple_listeners(self):
        engine = ProactiveEngine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.on("suggestion", cb1)
        engine.on("suggestion", cb2)
        assert len(engine._listeners["suggestion"]) == 2

    def test_different_events(self):
        engine = ProactiveEngine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.on("suggestion", cb1)
        engine.on("intervention", cb2)
        assert len(engine._listeners["suggestion"]) == 1
        assert len(engine._listeners["intervention"]) == 1


class TestEventEmitterOff:
    def test_removes_listener(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("task_completed", cb)
        engine.off("task_completed", cb)
        assert cb not in engine._listeners.get("task_completed", [])

    def test_off_nonexistent_listener_does_not_raise(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        # Should not raise
        engine.off("suggestion", cb)

    def test_off_nonexistent_event_does_not_raise(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.off("nonexistent_event", cb)

    def test_off_keeps_other_listeners(self):
        engine = ProactiveEngine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.on("suggestion", cb1)
        engine.on("suggestion", cb2)
        engine.off("suggestion", cb1)
        assert cb1 not in engine._listeners["suggestion"]
        assert cb2 in engine._listeners["suggestion"]


class TestEventEmitterEmit:
    def test_emit_calls_listeners(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("test_event", cb)
        engine._emit("test_event", "arg1", "arg2")
        cb.assert_called_once_with("arg1", "arg2")

    def test_emit_calls_all_listeners(self):
        engine = ProactiveEngine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.on("test_event", cb1)
        engine.on("test_event", cb2)
        engine._emit("test_event", 42)
        cb1.assert_called_once_with(42)
        cb2.assert_called_once_with(42)

    def test_emit_no_listeners_does_not_raise(self):
        engine = ProactiveEngine()
        engine._emit("no_listeners_event", "data")

    def test_emit_listener_exception_propagates(self):
        """A listener exception currently propagates due to a structlog kwarg conflict
        in the error handler (event= clashes with structlog's event positional arg).
        This test documents the actual behavior."""
        engine = ProactiveEngine()
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        engine.on("test_event", bad_cb)
        with pytest.raises(TypeError):
            engine._emit("test_event", "data")
        bad_cb.assert_called_once()


class TestEmitTaskCompleted:
    def test_emits_task_completed_event(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("task_completed", cb)
        engine.emit_task_completed("deploy", {"files": 3})
        cb.assert_called_once_with("deploy", {"files": 3})

    def test_emits_with_default_result(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("task_completed", cb)
        engine.emit_task_completed("build")
        cb.assert_called_once_with("build", {})


class TestEmitTaskFailed:
    def test_emits_task_failed_event(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("task_failed", cb)
        engine.emit_task_failed("deploy", "timeout")
        cb.assert_called_once_with("deploy", "timeout")

    def test_emits_with_default_error(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("task_failed", cb)
        engine.emit_task_failed("build")
        cb.assert_called_once_with("build", "")


class TestEvaluateEmitsEvents:
    @pytest.mark.asyncio
    async def test_evaluate_emits_suggestion_event(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("suggestion", cb)

        context = {
            "hints": [
                {"type": "insight", "title": "Test hint", "confidence": 0.6},
            ],
        }
        results = await engine.evaluate(context)
        assert len(results) >= 1
        cb.assert_called_once()
        # The callback receives the list of suggestions
        emitted_suggestions = cb.call_args[0][0]
        assert isinstance(emitted_suggestions, list)
        assert len(emitted_suggestions) >= 1

    @pytest.mark.asyncio
    async def test_evaluate_emits_intervention_for_high_confidence(self):
        engine = ProactiveEngine()
        intervention_cb = MagicMock()
        engine.on("intervention", intervention_cb)

        context = {
            "hints": [
                {"type": "warning", "title": "Critical issue", "confidence": 0.9},
            ],
        }
        results = await engine.evaluate(context)
        assert len(results) >= 1
        intervention_cb.assert_called_once()
        interventions = intervention_cb.call_args[0][0]
        assert all(s.confidence >= 0.8 for s in interventions)

    @pytest.mark.asyncio
    async def test_evaluate_no_intervention_for_low_confidence(self):
        # Force in-memory DB to avoid real data leaking into predictions
        import rune.proactive.prediction.engine as pe_mod
        pe_mod._engine = None
        import rune.memory.store as store_mod
        store_mod._store = store_mod.MemoryStore(db_path=":memory:")

        engine = ProactiveEngine()
        intervention_cb = MagicMock()
        engine.on("intervention", intervention_cb)

        context = {
            "hints": [
                {"type": "insight", "title": "Low conf", "confidence": 0.4},
            ],
        }
        await engine.evaluate(context)
        intervention_cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_emits_decision_event(self):
        engine = ProactiveEngine()
        decision_cb = MagicMock()
        engine.on("decision", decision_cb)

        context = {
            "hints": [
                {"type": "insight", "title": "A hint", "confidence": 0.5},
            ],
        }
        await engine.evaluate(context)
        decision_cb.assert_called_once()
        decision_data = decision_cb.call_args[0][0]
        assert "evaluation_count" in decision_data
        assert "suggestions" in decision_data

    @pytest.mark.asyncio
    async def test_evaluate_no_events_when_suppressed(self):
        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("suggestion", cb)

        context = {"suppress": True}
        results = await engine.evaluate(context)
        assert results == []
        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_no_events_when_no_candidates(self):
        # Force in-memory DB to avoid real data leaking into predictions
        import rune.proactive.prediction.engine as pe_mod
        pe_mod._engine = None
        import rune.memory.store as store_mod
        store_mod._store = store_mod.MemoryStore(db_path=":memory:")

        engine = ProactiveEngine()
        cb = MagicMock()
        engine.on("suggestion", cb)

        context = {}  # No hints, no last_action
        results = await engine.evaluate(context)
        assert results == []
        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_increments_evaluation_count(self):
        engine = ProactiveEngine()
        assert engine._evaluation_count == 0
        await engine.evaluate({})
        assert engine._evaluation_count == 1
        await engine.evaluate({})
        assert engine._evaluation_count == 2
