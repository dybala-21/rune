"""Drain stdout and stderr while retaining the start and end of the output."""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
from collections import deque

from rune.utils.logger import get_logger

log = get_logger(__name__)


class OutputBuffer:
    def __init__(self, limit: int) -> None:
        self.limit = max(2, limit)
        self._head: list[str] = []
        self._tail: deque[str] = deque()
        self._head_size = 0
        self._tail_size = 0
        self.size = 0

    def append(self, text: str) -> None:
        self.size += len(text)
        keep = min(len(text), self.limit // 2 - self._head_size)
        if keep:
            self._append_chunk(self._head, text[:keep])
            self._head_size += keep
        rest = text[keep:]
        tail_limit = self.limit - self.limit // 2
        if len(rest) >= tail_limit:
            self._tail.clear()
            self._tail_size = 0
            rest = rest[-tail_limit:]
        if rest:
            self._append_chunk(self._tail, rest)
            self._tail_size += len(rest)
        excess = self._tail_size - tail_limit
        while excess > 0:
            first = self._tail.popleft()
            removed = min(excess, len(first))
            self._tail_size -= removed
            excess -= removed
            if removed < len(first):
                self._tail.appendleft(first[removed:])

    @staticmethod
    def _append_chunk(chunks: list[str] | deque[str], text: str) -> None:
        # Coalesce small reads to bound the number of retained chunks.
        if chunks and len(chunks[-1]) + len(text) <= 4096:
            chunks[-1] += text
        else:
            chunks.append(text)

    def render(self) -> str:
        omitted = self.size - self._head_size - self._tail_size
        marker = [f"\n... ({omitted} characters omitted) ...\n"] if omitted > 0 else []
        return "".join([*self._head, *marker, *self._tail])


async def capture_output(process: asyncio.subprocess.Process, limit: int) -> tuple[str, str]:
    async def read(stream: asyncio.StreamReader) -> str:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = OutputBuffer(limit)
        while chunk := await stream.read(16 * 1024):
            buffer.append(decoder.decode(chunk))
        buffer.append(decoder.decode(b"", final=True))
        return buffer.render()

    assert process.stdout is not None and process.stderr is not None
    stdout, stderr, _ = await asyncio.gather(read(process.stdout), read(process.stderr), process.wait())
    return stdout, stderr


async def stop_capture(process: asyncio.subprocess.Process, capture: asyncio.Task) -> None:
    try:
        if hasattr(os, "killpg"):
            # The session leader may have exited while a child still holds its pipes.
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        log.debug("command_already_exited", pid=process.pid)
    except PermissionError as exc:
        log.warning("command_group_termination_failed", pid=process.pid, error=str(exc))
        if process.returncode is None:
            process.kill()
    try:
        await asyncio.wait_for(capture, timeout=3)
    except TimeoutError:
        log.warning("command_output_drain_timeout", pid=process.pid)
    except Exception as exc:
        log.debug("command_output_drain_failed", pid=process.pid, error=str(exc))
