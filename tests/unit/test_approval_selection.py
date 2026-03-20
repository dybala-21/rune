"""Tests for rune.ui.approval_selection — approval UI cycling and input resolution."""

from __future__ import annotations

from rune.ui.approval_selection import (
    APPROVAL_OPTIONS,
    APPROVAL_PROMPT_SELECTION_INDEX,
    DEFAULT_APPROVAL_SELECTION_INDEX,
    ApprovalDecision,
    ApprovalInputDecision,
    ApprovalInputNeedsPrompt,
    is_approval_prompt_selection,
    move_approval_selection,
    resolve_approval_from_input,
    selected_approval_decision,
)


class TestDefaults:
    def test_default_selection_is_deny(self):
        assert DEFAULT_APPROVAL_SELECTION_INDEX == 2
        assert selected_approval_decision(DEFAULT_APPROVAL_SELECTION_INDEX) == "deny"


class TestMoveApprovalSelection:
    def test_wraps_left_from_zero(self):
        assert move_approval_selection(0, "left") == 3

    def test_wraps_right_from_max(self):
        assert move_approval_selection(3, "right") == 0

    def test_wraps_up_from_zero(self):
        assert move_approval_selection(0, "up") == 3

    def test_wraps_down_from_max(self):
        assert move_approval_selection(3, "down") == 0


class TestSelectedApprovalDecision:
    def test_negative_index_maps_to_approve_once(self):
        assert selected_approval_decision(-10) == "approve_once"

    def test_index_zero(self):
        assert selected_approval_decision(0) == "approve_once"

    def test_index_one(self):
        assert selected_approval_decision(1) == "approve_always"

    def test_index_two(self):
        assert selected_approval_decision(2) == "deny"

    def test_large_index_maps_to_deny(self):
        assert selected_approval_decision(99) == "deny"


class TestResolveApprovalFromInput:
    def test_empty_input_uses_current_selection(self):
        result = resolve_approval_from_input("", 0)
        assert result == ApprovalInputDecision(decision=ApprovalDecision.APPROVE_ONCE)

    def test_whitespace_input_uses_current_selection(self):
        result = resolve_approval_from_input("   ", 1)
        assert result == ApprovalInputDecision(decision=ApprovalDecision.APPROVE_ALWAYS)

    def test_empty_input_on_prompt_slot_returns_needs_prompt(self):
        result = resolve_approval_from_input("", APPROVAL_PROMPT_SELECTION_INDEX)
        assert result == ApprovalInputNeedsPrompt()

    def test_number_shortcuts_1_to_3(self):
        assert resolve_approval_from_input("1", 2) == ApprovalInputDecision(decision=APPROVAL_OPTIONS[0])
        assert resolve_approval_from_input("2", 2) == ApprovalInputDecision(decision=APPROVAL_OPTIONS[1])
        assert resolve_approval_from_input("3", 0) == ApprovalInputDecision(decision=APPROVAL_OPTIONS[2])

    def test_number_4_returns_needs_prompt(self):
        result = resolve_approval_from_input("4", 0)
        assert result == ApprovalInputNeedsPrompt()

    def test_y_shortcut(self):
        result = resolve_approval_from_input("y", 2)
        assert result == ApprovalInputDecision(decision=ApprovalDecision.APPROVE_ONCE)

    def test_a_shortcut_uppercase(self):
        result = resolve_approval_from_input("A", 2)
        assert result == ApprovalInputDecision(decision=ApprovalDecision.APPROVE_ALWAYS)

    def test_n_shortcut_uppercase(self):
        result = resolve_approval_from_input("N", 0)
        assert result == ApprovalInputDecision(decision=ApprovalDecision.DENY)


class TestIsApprovalPromptSelection:
    def test_prompt_slot(self):
        assert is_approval_prompt_selection(APPROVAL_PROMPT_SELECTION_INDEX) is True

    def test_non_prompt_slot(self):
        assert is_approval_prompt_selection(2) is False
