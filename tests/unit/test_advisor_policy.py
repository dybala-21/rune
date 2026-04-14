"""Unit tests for the escalation policy state machine."""

from __future__ import annotations

from rune.agent.advisor.policy import (
    PolicyInput,
    PolicyState,
    mark_called,
    should_call,
)


def _inp(**overrides) -> PolicyInput:
    base = dict(
        is_complex_coding=True,
        goal_type="code_modify",
        activity_phase="exploration",
        reads=0,
        writes=0,
        web_fetches=0,
        files_written=0,
        gate_blocked_count=0,
        stall_consecutive=0,
        no_progress_steps=0,
        wind_down_phase="normal",
        hard_failures=0,
    )
    base.update(overrides)
    return PolicyInput(**base)


class TestTrivialGate:
    def test_chat_goals_disable_advisor(self):
        state = PolicyState()
        inp = _inp(goal_type="chat", is_complex_coding=False, reads=5)
        call, trigger = should_call(state, inp)
        assert call is False
        assert trigger is None

    def test_complex_coding_overrides_trivial(self):
        state = PolicyState()
        inp = _inp(goal_type="chat", is_complex_coding=True, reads=3)
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "early"


class TestEarlyTrigger:
    def test_fires_after_two_reads(self):
        state = PolicyState()
        inp = _inp(reads=2)
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "early"

    def test_fires_on_web_fetch(self):
        state = PolicyState()
        inp = _inp(web_fetches=1)
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "early"

    def test_does_not_fire_before_orientation(self):
        state = PolicyState()
        inp = _inp(reads=1, web_fetches=0)
        call, trigger = should_call(state, inp)
        assert call is False

    def test_does_not_fire_after_writes(self):
        state = PolicyState()
        inp = _inp(reads=5, writes=1)
        call, trigger = should_call(state, inp)
        # writes→0 gate blocks EARLY; other triggers inactive
        assert trigger != "early"

    def test_idempotent_after_marking(self):
        state = PolicyState()
        mark_called(state, "early", hard_failures=0)
        inp = _inp(reads=3)
        call, trigger = should_call(state, inp)
        assert trigger != "early"


class TestPreDoneTrigger:
    def test_fires_when_deliverable_durable(self):
        state = PolicyState(early_called=True)
        inp = _inp(
            writes=1,
            files_written=1,
            activity_phase="verification",
            reads=5,
        )
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "pre_done"

    def test_does_not_fire_without_durable_file(self):
        state = PolicyState(early_called=True)
        inp = _inp(writes=1, files_written=0, activity_phase="verification")
        call, trigger = should_call(state, inp)
        assert trigger != "pre_done"

    def test_idempotent(self):
        state = PolicyState(early_called=True)
        mark_called(state, "pre_done", hard_failures=0)
        inp = _inp(
            writes=2,
            files_written=2,
            activity_phase="verification",
        )
        call, trigger = should_call(state, inp)
        assert trigger != "pre_done"


class TestStuckTrigger:
    def test_gate_blocked_3_fires(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(gate_blocked_count=3)
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "stuck"

    def test_gate_blocked_2_does_not_fire(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(gate_blocked_count=2)
        call, trigger = should_call(state, inp)
        assert trigger != "stuck"

    def test_stall_consecutive_5_fires(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(stall_consecutive=5)
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"

    def test_no_progress_3_fires(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(no_progress_steps=3)
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"

    def test_wind_down_final_fires(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(wind_down_phase="final")
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"

    def test_stuck_bounded_by_max_calls(self):
        state = PolicyState(
            early_called=True, pre_done_called=True,
            stuck_calls=2, max_stuck_calls=2,
        )
        inp = _inp(gate_blocked_count=3)
        call, trigger = should_call(state, inp)
        assert call is False


class TestReconcileTrigger:
    def test_fires_when_advice_followed_and_regressed(self):
        state = PolicyState(
            early_called=True,
            followed_last_advice=True,
            hard_failures_at_last_advice=1,
        )
        inp = _inp(hard_failures=2)
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "reconcile"

    def test_does_not_fire_without_regression(self):
        state = PolicyState(
            followed_last_advice=True,
            hard_failures_at_last_advice=1,
        )
        inp = _inp(hard_failures=1)
        call, trigger = should_call(state, inp)
        assert trigger != "reconcile"

    def test_only_once_per_episode(self):
        state = PolicyState(
            followed_last_advice=True,
            hard_failures_at_last_advice=1,
        )
        mark_called(state, "reconcile", hard_failures=2)
        inp = _inp(hard_failures=3)
        call, trigger = should_call(state, inp)
        assert trigger != "reconcile"


class TestDisabled:
    def test_disabled_state_blocks_all(self):
        state = PolicyState(advisor_disabled=True)
        inp = _inp(reads=5, gate_blocked_count=3)
        call, trigger = should_call(state, inp)
        assert call is False


class TestWindDownRisingEdge:
    """P2: wind_down_phase='final' is monotonic (once final, stays final
    until hard_stop). Rising-edge detection prevents the stuck trigger
    from burning repeated calls on the same persistent signal."""

    def test_fresh_episode_fires_on_first_entry_to_final(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(wind_down_phase="final")
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "stuck"

    def test_does_not_refire_when_phase_persists(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(wind_down_phase="final")

        # First evaluation: phase transitions normal→final, should fire
        call1, trigger1 = should_call(state, inp)
        assert call1 is True
        assert trigger1 == "stuck"
        mark_called(
            state, "stuck", hard_failures=0, wind_down_phase="final",
        )

        # Second evaluation: still in final → no refire on same signal
        call2, trigger2 = should_call(state, inp)
        assert call2 is False
        assert trigger2 is None

    def test_mark_called_stores_wind_down_phase(self):
        state = PolicyState()
        mark_called(
            state, "stuck", hard_failures=0, wind_down_phase="final",
        )
        assert state.last_wind_down_phase == "final"

    def test_mark_called_backward_compat_default(self):
        """Legacy callers that don't pass wind_down_phase still work."""
        state = PolicyState()
        mark_called(state, "stuck", hard_failures=0)
        assert state.last_wind_down_phase == "normal"

    def test_other_stuck_signals_still_fire_when_wind_down_blocked(self):
        """After wind_down stuck fires, gate_blocked_count==3 on a later
        iteration must still be able to fire a second stuck call."""
        state = PolicyState(early_called=True, pre_done_called=True)

        # First stuck via wind_down
        inp1 = _inp(wind_down_phase="final")
        call1, _ = should_call(state, inp1)
        assert call1 is True
        mark_called(
            state, "stuck", hard_failures=0, wind_down_phase="final",
        )

        # Second stuck via gate_blocked on the SAME persistent wind_down.
        # max_stuck_calls=2, so this is still within budget.
        inp2 = _inp(wind_down_phase="final", gate_blocked_count=3)
        call2, trigger2 = should_call(state, inp2)
        assert call2 is True
        assert trigger2 == "stuck"

    def test_non_final_phases_never_fire_stuck(self):
        """Only 'final' should trigger the stuck wind_down branch."""
        state = PolicyState(early_called=True, pre_done_called=True)
        for phase in ("normal", "wrapping", "stopping"):
            inp = _inp(wind_down_phase=phase)
            call, trigger = should_call(state, inp)
            assert trigger != "stuck" or call is False


class TestStuckReason:
    """2b-3: policy records which stuck sub-condition fired as a side
    effect on state.last_stuck_reason so persistence can attribute
    each stuck call to a specific signal."""

    def test_gate_blocked_reason(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(gate_blocked_count=3)
        call, trigger = should_call(state, inp)
        assert call is True
        assert trigger == "stuck"
        assert state.last_stuck_reason == "gate_blocked"

    def test_stall_reason(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(stall_consecutive=5)
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"
        assert state.last_stuck_reason == "stall"

    def test_no_progress_reason(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(no_progress_steps=3)
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"
        assert state.last_stuck_reason == "no_progress"

    def test_wind_down_reason(self):
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(wind_down_phase="final")
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"
        assert state.last_stuck_reason == "wind_down"

    def test_non_stuck_triggers_do_not_set_reason(self):
        """EARLY/PRE_DONE/RECONCILE should NOT touch last_stuck_reason."""
        state = PolicyState()
        inp = _inp(reads=2)  # EARLY conditions
        call, trigger = should_call(state, inp)
        assert trigger == "early"
        assert state.last_stuck_reason == ""  # untouched

    def test_reason_priority_matches_policy_order(self):
        """When multiple stuck conditions are true at once, the first
        branch in should_call wins (gate_blocked > stall > no_progress
        > wind_down). This documents the current deterministic order."""
        state = PolicyState(early_called=True, pre_done_called=True)
        inp = _inp(
            gate_blocked_count=3,
            stall_consecutive=10,
            no_progress_steps=5,
            wind_down_phase="final",
        )
        call, trigger = should_call(state, inp)
        assert trigger == "stuck"
        assert state.last_stuck_reason == "gate_blocked"
