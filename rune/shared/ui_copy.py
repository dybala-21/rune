"""Shared UI copy strings and templates.

Ported from src/shared/ui-copy.ts - centralised text constants used
across CLI, TUI, and API surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

# Approval dialog copy

@dataclass(frozen=True, slots=True)
class _ApprovalCopy:
    title: str = "Approval Required"
    command_label: str = "Command"
    reason_label: str = "Reason"
    suggestions_label: str = "Suggestions"
    actions_label: str = "Select an action"
    allow_once_label: str = "Allow Once"
    always_allow_label: str = "Always Allow"
    deny_label: str = "Deny"
    deny_with_instructions_label: str = "Deny with Instructions"
    instructions_label: str = "Instructions"
    instructions_empty_label: str = "(type here)"
    denial_placeholder: str = "Add denial instructions..."
    denial_hint_empty: str = "Please type denial instructions first."
    denial_hint_prompt: str = "Type denial instructions and press Enter."


APPROVAL_COPY = _ApprovalCopy()


# Question dialog copy

@dataclass(frozen=True, slots=True)
class _QuestionCopy:
    title: str = "Question"
    answered_label: str = "Answered"
    custom_option_label: str = "Custom input (guide the flow)"
    custom_option_description: str = "Type your own instruction to steer the next step."
    answer_needed_headline: str = "Your answer is needed to proceed."
    select_or_type_hint: str = "Use Up/Down + Enter to select, or type a custom answer."
    custom_input_hint: str = "Type your instructions and press Enter."
    free_text_hint: str = "Type your answer and press Enter."
    select_or_type_placeholder: str = "Use Up/Down + Enter to select, or type your own input"
    custom_input_placeholder: str = "Enter your own instruction to steer the flow..."
    free_text_placeholder: str = "Type your answer and press Enter..."
    free_text_field_placeholder: str = "Type your answer..."
    secret_field_placeholder: str = "Enter secret value..."
    submit_label: str = "Submit"


QUESTION_COPY = _QuestionCopy()


# Helper functions (mirrors TS getQuestion* functions)

def get_question_entry_placeholder(
    has_options: bool,
    is_custom_selected: bool,
) -> str:
    """Return the appropriate placeholder text for the question input."""
    if has_options:
        if is_custom_selected:
            return QUESTION_COPY.custom_input_placeholder
        return QUESTION_COPY.select_or_type_placeholder
    return QUESTION_COPY.free_text_placeholder


def get_question_guidance_text(
    has_options: bool,
    is_custom_selected: bool,
) -> str:
    """Return the appropriate guidance hint for the question input."""
    if has_options:
        if is_custom_selected:
            return QUESTION_COPY.custom_input_hint
        return QUESTION_COPY.select_or_type_hint
    return QUESTION_COPY.free_text_hint


def get_question_field_placeholder(
    input_mode: str = "text",
) -> str:
    """Return the field placeholder based on input mode."""
    if input_mode == "secret":
        return QUESTION_COPY.secret_field_placeholder
    return QUESTION_COPY.free_text_field_placeholder
