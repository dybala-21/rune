"""Implicit need inference for RUNE.

Analyses recent actions and context to infer needs the user has
not explicitly stated (documentation, testing, refactoring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rune.proactive.context import AwarenessContext
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Tool patterns that hint at specific needs
# Use actual capability names (underscore-separated, not dot-separated)
_DOC_TOOLS = {"file_read", "web_search", "web_fetch"}
_TEST_TOOLS = {"bash_execute", "file_edit", "file_write"}
_REFACTOR_TOOLS = {"file_edit", "file_write", "file_search"}

# Thresholds
_DOC_READ_THRESHOLD = 5
_TEST_EDIT_THRESHOLD = 4
_REFACTOR_EDIT_THRESHOLD = 6


@dataclass(slots=True)
class InferredNeed:
    """An implicit user need detected from behaviour patterns."""

    need_type: str
    confidence: float
    context: dict[str, Any] = field(default_factory=dict)


class NeedInferer:
    """Infers implicit user needs from recent actions and context.

    Checks for documentation, testing, and refactoring needs based
    on tool-usage patterns and workspace state.
    """

    __slots__ = ()

    def infer(
        self,
        context: AwarenessContext,
        recent_actions: list[dict[str, Any]],
    ) -> list[InferredNeed]:
        """Infer needs from context and recent actions.

        Parameters:
            context: Current awareness context.
            recent_actions: List of recent action dicts (each should have
                at minimum a "tool" key).

        Returns:
            List of inferred needs, sorted by confidence descending.
        """
        needs: list[InferredNeed] = []

        doc_need = self._check_documentation_need(recent_actions)
        if doc_need is not None:
            needs.append(doc_need)

        test_need = self._check_testing_need(recent_actions)
        if test_need is not None:
            needs.append(test_need)

        refactor_need = self._check_refactor_need(recent_actions)
        if refactor_need is not None:
            needs.append(refactor_need)

        needs.sort(key=lambda n: n.confidence, reverse=True)

        if needs:
            log.debug(
                "needs_inferred",
                count=len(needs),
                types=[n.need_type for n in needs],
            )

        return needs

    def _check_documentation_need(
        self,
        actions: list[dict[str, Any]],
    ) -> InferredNeed | None:
        """Detect if the user likely needs documentation help.

        Heuristic: heavy file reading and web searching without writing
        suggests the user is looking for information.
        """
        if len(actions) < _DOC_READ_THRESHOLD:
            return None

        recent = actions[-10:]
        read_count = sum(1 for a in recent if a.get("tool") in _DOC_TOOLS)
        write_count = sum(1 for a in recent if a.get("tool") in {"file_write", "file_edit"})

        if read_count >= _DOC_READ_THRESHOLD and write_count <= 1:
            confidence = min(0.9, 0.4 + (read_count - _DOC_READ_THRESHOLD) * 0.1)
            return InferredNeed(
                need_type="documentation",
                confidence=round(confidence, 3),
                context={"read_count": read_count, "write_count": write_count},
            )

        return None

    def _check_testing_need(
        self,
        actions: list[dict[str, Any]],
    ) -> InferredNeed | None:
        """Detect if the user likely needs to write or run tests.

        Heuristic: many edits followed by executions without test-related
        keywords suggest the code is untested.
        """
        if len(actions) < _TEST_EDIT_THRESHOLD:
            return None

        recent = actions[-12:]
        edit_count = sum(1 for a in recent if a.get("tool") in _TEST_TOOLS)
        has_test_mention = any(
            "test" in str(a.get("args", "")).lower() for a in recent
        )

        if edit_count >= _TEST_EDIT_THRESHOLD and not has_test_mention:
            confidence = min(0.8, 0.3 + (edit_count - _TEST_EDIT_THRESHOLD) * 0.1)
            return InferredNeed(
                need_type="testing",
                confidence=round(confidence, 3),
                context={"edit_count": edit_count},
            )

        return None

    def _check_refactor_need(
        self,
        actions: list[dict[str, Any]],
    ) -> InferredNeed | None:
        """Detect if the user would benefit from refactoring.

        Heuristic: many edits across multiple files touching similar patterns
        suggests duplication or structural issues.
        """
        if len(actions) < _REFACTOR_EDIT_THRESHOLD:
            return None

        recent = actions[-15:]
        edit_actions = [a for a in recent if a.get("tool") in _REFACTOR_TOOLS]

        if len(edit_actions) < _REFACTOR_EDIT_THRESHOLD:
            return None

        # Check if edits span multiple distinct files
        files_edited = {a.get("file", "") for a in edit_actions if a.get("file")}
        if len(files_edited) >= 3:
            confidence = min(0.7, 0.3 + len(files_edited) * 0.05)
            return InferredNeed(
                need_type="refactoring",
                confidence=round(confidence, 3),
                context={"files_edited": len(files_edited), "edit_count": len(edit_actions)},
            )

        return None
