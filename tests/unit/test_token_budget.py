"""Tests for TokenBudget and StallState from the agent loop module."""

from __future__ import annotations

import time

from rune.agent.loop import NativeAgentLoop, StallState, TokenBudget


class TestTokenBudget:
    def test_default_values(self):
        budget = TokenBudget()
        assert budget.total == 500_000
        assert budget.used == 0
        assert budget.fraction == 0.0

    def test_fraction_computation(self):
        budget = TokenBudget(total=1000, used=250)
        assert budget.fraction == 0.25

    def test_fraction_zero_total(self):
        budget = TokenBudget(total=0, used=0)
        assert budget.fraction == 0.0

    def test_phase_1(self):
        budget = TokenBudget(total=1000, used=100)
        assert budget.phase == 1

    def test_phase_2(self):
        budget = TokenBudget(total=1000, used=650)
        # 0.65 >= phase_2 threshold (0.60)
        assert budget.phase == 2

    def test_phase_3(self):
        budget = TokenBudget(total=1000, used=800)
        # 0.80 >= phase_3 threshold (0.75)
        assert budget.phase == 3

    def test_phase_4(self):
        budget = TokenBudget(total=1000, used=900)
        # 0.90 >= phase_4 threshold (0.85)
        assert budget.phase == 4

    def test_needs_rollover_false(self):
        budget = TokenBudget(total=1000, used=500)
        # 0.50 < rollover phase_1 threshold (0.70)
        assert budget.needs_rollover is False

    def test_needs_rollover_true(self):
        budget = TokenBudget(total=1000, used=750)
        # 0.75 >= rollover phase_1 threshold (0.70)
        assert budget.needs_rollover is True

    def test_rollover_phase_0(self):
        budget = TokenBudget(total=1000, used=100)
        assert budget.rollover_phase == 0

    def test_rollover_phase_1(self):
        budget = TokenBudget(total=1000, used=720)
        assert budget.rollover_phase == 1

    def test_rollover_phase_2(self):
        budget = TokenBudget(total=1000, used=850)
        assert budget.rollover_phase == 2

    def test_rollover_phase_3(self):
        budget = TokenBudget(total=1000, used=920)
        assert budget.rollover_phase == 3

    def test_rollover_phase_4(self):
        budget = TokenBudget(total=1000, used=980)
        assert budget.rollover_phase == 4

    def test_wind_down_phase_resets_when_budget_drops(self):
        loop = NativeAgentLoop()
        loop._token_budget = TokenBudget(total=1000, used=980)
        loop._update_wind_down_phase()
        assert loop._wind_down_phase == "hard_stop"

        loop._token_budget = TokenBudget(total=1000, used=0)
        loop._update_wind_down_phase()
        assert loop._wind_down_phase == "none"


class TestStallState:
    def test_default_state(self):
        state = StallState()
        assert state.consecutive_no_progress == 0
        assert state.cumulative_no_progress == 0
        assert state.is_stalled is False

    def test_mark_no_progress(self):
        state = StallState()
        state.mark_no_progress()
        assert state.consecutive_no_progress == 1
        assert state.cumulative_no_progress == 1
        assert state.is_stalled is False

    def test_stall_after_consecutive(self):
        state = StallState()
        for _ in range(3):
            state.mark_no_progress()
        assert state.is_stalled is True

    def test_stall_after_cumulative(self):
        state = StallState()
        for _ in range(8):
            state.mark_no_progress()
            if state.consecutive_no_progress >= 2:
                state.mark_activity("tool")  # Reset consecutive but keep cumulative
        # After 8 cumulative no-progress events, should be stalled
        # We need to track the actual cumulative count
        state2 = StallState()
        for i in range(10):
            state2.mark_no_progress()
            if i < 7:
                state2.mark_activity()
        # cumulative is 10 >= 8, should be stalled
        assert state2.is_stalled is True

    def test_mark_activity_resets_consecutive(self):
        state = StallState()
        state.mark_no_progress()
        state.mark_no_progress()
        assert state.consecutive_no_progress == 2
        state.mark_activity("file_read")
        assert state.consecutive_no_progress == 0
        assert state.last_tool_call == "file_read"

    def test_time_since_activity(self):
        state = StallState()
        # Just created, time_since_activity should be very small
        assert state.time_since_activity < 1.0
        # After a brief wait
        time.sleep(0.05)
        assert state.time_since_activity >= 0.04

    def test_mark_activity_updates_time(self):
        state = StallState()
        time.sleep(0.05)
        elapsed_before = state.time_since_activity
        state.mark_activity("bash")
        elapsed_after = state.time_since_activity
        assert elapsed_after < elapsed_before
