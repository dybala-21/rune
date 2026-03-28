"""Tool calling policy — adjusts LLM parameters based on runtime behavior.

Handles two failure modes common in weaker models:
1. Model generates text but never calls tools → retry with tool_choice="required"
2. Model calls the same tool in a loop → nudge, then block

Configurable via config.yaml under `tool_policy:`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Read-only tools are exempt from hard blocking because sequential
# reads of different files are a legitimate pattern.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "file_read", "file_list", "file_search",
    "code_analyze", "code_find_def", "code_find_refs",
    "project_map", "think",
})

_BLOCK_GRACE = 2  # extra calls allowed after nudge before hard block


@dataclass
class ToolCallPolicy:
    """Adjustable policy for tool calling behavior.

    Attributes:
        force_tool_on_empty: Retry with tool_choice="required" when model
            produces text without calling any tool.
        max_force_retries: Maximum number of forced retries per round (default 1).
        max_consecutive_same_tool: Trigger a nudge after this many consecutive
            calls to the same tool.
        disable_parallel: Set parallel_tool_calls=false (OpenAI-compatible only).
    """

    force_tool_on_empty: bool = True
    max_force_retries: int = 1
    max_consecutive_same_tool: int = 5
    disable_parallel: bool = False

    # Internal state — reset per round via reset()
    _force_count: int = field(default=0, repr=False)
    _tool_history: list[str] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        """Reset per-round state. Call at the start of each stream_text() round."""
        self._force_count = 0
        self._tool_history.clear()

    def should_force_tool(self, *, has_tool_calls: bool, has_text: bool) -> bool:
        """Should we retry with tool_choice='required'?

        Returns True when the model produced text but no tool calls,
        and we haven't exhausted retries.
        """
        if not self.force_tool_on_empty:
            return False
        if self._force_count >= self.max_force_retries:
            return False
        if has_tool_calls or not has_text:
            return False
        self._force_count += 1
        return True

    def record_tool_call(self, name: str) -> str | None:
        """Record a tool call and return a nudge message if looping.

        Returns a user-facing nudge string if the same tool was called
        N times consecutively, or None if no intervention needed.
        Does NOT remove tools — only nudges.
        """
        self._tool_history.append(name)
        n = self.max_consecutive_same_tool
        recent = self._tool_history[-n:]
        if len(recent) == n and len(set(recent)) == 1:
            return (
                f"You called {name} {n} times in a row. "
                f"Consider progressing to the next step."
            )
        return None

    def should_block_tool(self, name: str) -> bool:
        """Return True if *name* should be hard-blocked.

        After the nudge threshold (``max_consecutive_same_tool``) plus a
        grace window of 2 additional calls, the tool is blocked to prevent
        infinite loops.  Read-only tools are exempt because sequential
        reads of different files are a legitimate pattern.
        """
        if name in _READ_ONLY_TOOLS:
            return False
        block_at = self.max_consecutive_same_tool + _BLOCK_GRACE
        recent = self._tool_history[-block_at:]
        return len(recent) == block_at and len(set(recent)) == 1

    def get_extra_params(self) -> dict[str, Any]:
        """Extra parameters to pass to litellm.acompletion()."""
        params: dict[str, Any] = {}
        if self.disable_parallel:
            params["parallel_tool_calls"] = False
        return params
