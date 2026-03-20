"""Question prompt selection and rendering for RUNE TUI.

Ported from src/ui/question-selection.ts - option counting,
custom-selection detection, and cyclic navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CUSTOM_QUESTION_OPTION_LABEL = "Custom response"
CUSTOM_QUESTION_OPTION_DESCRIPTION = "Type your own answer below"


@dataclass(slots=True, frozen=True)
class QuestionOption:
    label: str
    description: str = ""


def get_question_menu_count(options: list[QuestionOption] | None = None) -> int:
    """Return total menu item count (options + 1 for custom entry).

    Returns 0 when there are no predefined options.
    """
    if not options:
        return 0
    return len(options) + 1


def is_custom_question_selection(
    selected_index: int,
    options: list[QuestionOption] | None = None,
) -> bool:
    """Return *True* when *selected_index* points to the custom entry."""
    if not options:
        return False
    return selected_index == len(options)


def move_question_selection(
    current_index: int,
    direction: Literal["up", "down"],
    options: list[QuestionOption] | None = None,
) -> int:
    """Move the selection cursor, wrapping around at edges."""
    count = get_question_menu_count(options)
    if count <= 0:
        return 0

    max_index = count - 1
    normalized = max(0, min(current_index, max_index))

    if direction == "up":
        return max_index if normalized <= 0 else normalized - 1
    return 0 if normalized >= max_index else normalized + 1


def render_question_options(
    options: list[QuestionOption] | None,
    selected_index: int = 0,
) -> list[str]:
    """Return a list of formatted option strings for display.

    The currently selected option is prefixed with ``> ``.
    """
    if not options:
        return []

    lines: list[str] = []
    all_options = list(options) + [
        QuestionOption(
            label=CUSTOM_QUESTION_OPTION_LABEL,
            description=CUSTOM_QUESTION_OPTION_DESCRIPTION,
        )
    ]
    for i, opt in enumerate(all_options):
        prefix = "> " if i == selected_index else "  "
        desc = f"  ({opt.description})" if opt.description else ""
        lines.append(f"{prefix}{i + 1}. {opt.label}{desc}")
    return lines
