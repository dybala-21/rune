"""Tests for rune.utils.non_reentrant_loop — ported from non-reentrant-loop.test.ts."""

import asyncio

import pytest

from rune.utils.non_reentrant_loop import NonReentrantLoop


class TestNonReentrantLoop:
    """Tests for the NonReentrantLoop class."""

    @pytest.mark.asyncio
    async def test_runs_ticks_serially_without_overlap(self):
        calls = 0
        in_flight = 0
        max_in_flight = 0

        async def tick():
            nonlocal calls, in_flight, max_in_flight
            calls += 1
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1

        loop = NonReentrantLoop(interval_sec=0.02, tick=tick)
        loop.start()
        await asyncio.sleep(0.4)
        loop.stop()

        assert calls >= 3
        assert max_in_flight == 1

    @pytest.mark.asyncio
    async def test_stops_scheduling_after_stop(self):
        calls = 0

        def tick():
            nonlocal calls
            calls += 1

        loop = NonReentrantLoop(interval_sec=0.02, tick=tick)
        loop.start()
        await asyncio.sleep(0.1)
        loop.stop()
        calls_at_stop = calls
        await asyncio.sleep(0.1)

        assert calls == calls_at_stop

    @pytest.mark.asyncio
    async def test_forwards_tick_errors_and_continues(self):
        errors: list[BaseException] = []
        calls = 0

        async def tick():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("tick-failure")

        def on_error(exc: BaseException):
            errors.append(exc)

        loop = NonReentrantLoop(interval_sec=0.02, tick=tick, on_error=on_error)
        loop.start()
        await asyncio.sleep(0.15)
        loop.stop()

        assert len(errors) == 1
        assert calls >= 2

    @pytest.mark.asyncio
    async def test_trigger_now_runs_immediately(self):
        calls = 0
        in_flight = 0
        max_in_flight = 0

        async def tick():
            nonlocal calls, in_flight, max_in_flight
            calls += 1
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1

        loop = NonReentrantLoop(interval_sec=0.2, tick=tick)
        loop.start()
        loop.trigger_now()
        await asyncio.sleep(0.5)
        loop.stop()

        assert calls >= 2
        assert max_in_flight == 1

    @pytest.mark.asyncio
    async def test_is_active_property(self):
        loop = NonReentrantLoop(interval_sec=1.0, tick=lambda: None)
        assert loop.is_active is False
        loop.start()
        assert loop.is_active is True
        loop.stop()
        assert loop.is_active is False

    @pytest.mark.asyncio
    async def test_start_is_noop_when_already_running(self):
        tick_count = 0

        def tick():
            nonlocal tick_count
            tick_count += 1

        loop = NonReentrantLoop(interval_sec=0.02, tick=tick)
        loop.start()
        loop.start()  # should be no-op
        await asyncio.sleep(0.08)
        loop.stop()
        # Just verify it ran and didn't crash
        assert tick_count >= 1
