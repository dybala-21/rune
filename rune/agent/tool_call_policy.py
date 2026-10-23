"""Nudge empty responses and stop repeated calls with identical arguments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Repeated reads may target different files, so tool names alone cannot block them.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "file_read", "file_list", "file_search", "document_read",
    "code_analyze", "code_find_def", "code_find_refs",
    "project_map", "think",
})

_BLOCK_GRACE = 2  # Calls allowed after the nudge.


@dataclass
class ToolCallPolicy:
    """Adjustable policy for tool calling behavior.

    Attributes:
        force_tool_on_empty: Retry with tool_choice="required" when model
            produces text without calling any tool.
        max_force_retries: Maximum number of forced retries per round (default 1).
        max_consecutive_same_tool: Nudge after this many identical calls.
        disable_parallel: Set parallel_tool_calls=false (OpenAI-compatible only).
    """

    force_tool_on_empty: bool = True
    max_force_retries: int = 1
    max_consecutive_same_tool: int = 5
    disable_parallel: bool = False

    # Reset at the start of each stream.
    _force_count: int = field(default=0, repr=False)
    _tool_history: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _rejected_rounds: int = field(default=0, repr=False)

    def reset(self) -> None:
        """Reset per-round state. Call at the start of each stream_text() round."""
        self._force_count = 0
        self._tool_history.clear()
        self._rejected_rounds = 0

    def record_result(self, name: str, result: Any) -> str | None:
        """Bound rejected rounds without stopping executed checks that failed."""
        from rune.agent.tool_output import RejectedToolResult

        if not isinstance(result, RejectedToolResult):
            self._rejected_rounds = 0
            return None
        self._rejected_rounds += 1
        from rune.utils.logger import get_logger

        get_logger(__name__).info("tool_call_rejected", tool=name, code=result.code, consecutive=self._rejected_rounds)
        if self._rejected_rounds >= 3:
            return (f"Stopped after {self._rejected_rounds} consecutive rounds of rejected tool calls. "
                    f"Last rejection: {name} ({result.code}). These calls made no changes; the task remains unfinished.")
        return None

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

    @staticmethod
    def _call_key(name: str, args: dict[str, Any] | None) -> tuple[str, str]:
        params = {key: value for key, value in args.items() if key != "think"} if isinstance(args, dict) else args
        encoded = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return name, hashlib.sha256(encoded.encode()).hexdigest()

    def record_tool_call(self, name: str, args: dict[str, Any] | None = None) -> str | None:
        """Record the call and nudge when identical arguments keep repeating."""
        self._tool_history.append(self._call_key(name, args))
        self._tool_history = self._tool_history[-(self.max_consecutive_same_tool + _BLOCK_GRACE):]
        n = self.max_consecutive_same_tool
        recent = self._tool_history[-n:]
        if len(recent) == n and len(set(recent)) == 1:
            return (
                f"You called {name} with the same arguments {n} times in a row. "
                "Inspect the result before repeating this action."
            )
        return None

    def should_block_tool(self, name: str, args: dict[str, Any] | None = None) -> bool:
        """Block identical calls after the nudge and grace window."""
        if name in _READ_ONLY_TOOLS:
            return False
        block_at = self.max_consecutive_same_tool + _BLOCK_GRACE
        recent = self._tool_history[-block_at:]
        key = self._call_key(name, args)
        return len(recent) == block_at and all(previous == key for previous in recent)

    def get_extra_params(self) -> dict[str, Any]:
        """Extra parameters to pass to litellm.acompletion()."""
        params: dict[str, Any] = {}
        if self.disable_parallel:
            params["parallel_tool_calls"] = False
        return params
