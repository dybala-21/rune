"""DelayedCommitController for RUNE TUI.

Buffers streaming text deltas and flushes them after a configurable
delay (default 500ms). Tool calls and step boundaries interrupt or
clear the buffer as appropriate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from rune.utils.logger import get_logger

log = get_logger(__name__)


class DelayedCommitController:
    """Buffers text deltas and flushes to a callback after a delay.

    This prevents rapid small updates from thrashing the UI; instead,
    text accumulates and is committed in batches.
    """

    def __init__(
        self,
        on_flush: Callable[[str], None],
        delay_ms: int = 500,
    ) -> None:
        self._on_flush = on_flush
        self._delay_s = delay_ms / 1000.0
        self._buffer: str = ""
        self._flush_handle: asyncio.TimerHandle | None = None

    # Public API ------------------------------------------------------------

    def push_delta(self, text: str) -> None:
        """Append *text* to the buffer and schedule a flush."""
        self._buffer += text
        self._schedule_flush()

    def handle_tool_call(self) -> None:
        """A tool call interrupts text streaming - flush immediately then clear."""
        self._cancel_scheduled()
        if self._buffer.strip():
            self._do_flush()
        self._buffer = ""

    def handle_step_finish(self, is_final: bool) -> None:
        """A step finished. Flush if final step, otherwise clear."""
        self._cancel_scheduled()
        if is_final and self._buffer.strip():
            self._do_flush()
        self._buffer = ""

    def handle_complete(self) -> None:
        """Agent loop complete - flush any remaining text then clear."""
        self._cancel_scheduled()
        if self._buffer.strip():
            self._do_flush()
        self._buffer = ""

    def flush(self) -> None:
        """Manually flush the buffer."""
        self._cancel_scheduled()
        if self._buffer.strip():
            self._do_flush()
        self._buffer = ""

    # Internal --------------------------------------------------------------

    def _schedule_flush(self) -> None:
        self._cancel_scheduled()
        try:
            loop = asyncio.get_running_loop()
            self._flush_handle = loop.call_later(self._delay_s, self._timer_fired)
        except RuntimeError:
            # No running loop - flush immediately
            self._do_flush()

    def _cancel_scheduled(self) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None

    def _timer_fired(self) -> None:
        """Called when the delay timer fires."""
        self._flush_handle = None
        if self._buffer.strip():
            self._do_flush()
            self._buffer = ""

    def _do_flush(self) -> None:
        """Invoke the flush callback with the buffered text."""
        text = self._buffer
        try:
            self._on_flush(text)
        except Exception:
            log.exception("delayed_commit_flush_error")
