"""Tests for rune.ui.pending_focus — pending focus state derivation."""

from __future__ import annotations

from rune.ui.pending_focus import (
    PendingFocusInput,
    PendingFocusMode,
    get_pending_focus_state,
)


class TestGetPendingFocusState:
    def test_prioritizes_approval_over_others(self):
        inp = PendingFocusInput(
            has_approval=True,
            has_question=True,
            has_credential=True,
            approval_command="rm -rf /tmp/test",
            question="pick next step",
            question_option_count=3,
        )
        state = get_pending_focus_state(inp)
        assert state.active is True
        assert state.mode == PendingFocusMode.APPROVAL
        assert "input" in state.title.lower() or "waiting" in state.title.lower()
        assert "Command:" in state.prompt

    def test_question_mode_with_options(self):
        inp = PendingFocusInput(
            has_question=True,
            question="Continue the websocket gateway task?",
            question_option_count=2,
        )
        state = get_pending_focus_state(inp)
        assert state.mode == PendingFocusMode.QUESTION
        assert "answer" in state.headline.lower()
        assert "websocket" in state.prompt.lower() or "Continue" in state.prompt

    def test_idle_when_nothing_pending(self):
        inp = PendingFocusInput()
        state = get_pending_focus_state(inp)
        assert state.active is False
        assert state.mode == PendingFocusMode.IDLE
        assert state.action_hint == ""

    def test_setup_provider_selection(self):
        inp = PendingFocusInput(setup_phase="select-provider")
        state = get_pending_focus_state(inp)
        assert state.active is True
        assert state.mode == PendingFocusMode.SETUP
        assert "provider" in state.prompt.lower() or "provider" in state.detail.lower()

    def test_setup_enter_key(self):
        inp = PendingFocusInput(setup_phase="enter-key")
        state = get_pending_focus_state(inp)
        assert state.active is True
        assert state.mode == PendingFocusMode.SETUP
        assert "api key" in state.prompt.lower() or "API key" in state.prompt

    def test_credential_mode(self):
        inp = PendingFocusInput(has_credential=True)
        state = get_pending_focus_state(inp)
        assert state.active is True
        assert state.mode == PendingFocusMode.CREDENTIAL

    def test_question_with_custom_selected(self):
        inp = PendingFocusInput(
            has_question=True,
            question="Pick an option",
            question_option_count=3,
            question_custom_selected=True,
        )
        state = get_pending_focus_state(inp)
        assert state.mode == PendingFocusMode.QUESTION
        assert "custom" in state.action_hint.lower() or "type" in state.action_hint.lower()
