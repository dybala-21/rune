"""Stream an opt-in POSIX PTY to the embedded terminal.

The handshake requires loopback, a valid Origin and a single-use token. This blocks cross-site
access, not code running in the renderer's own origin: renderer compromise grants a local shell
when the terminal is enabled. Guardian covers agent commands, not this interactive shell.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import signal
import struct
import termios
import threading
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# token -> {"workspace": str, "used": bool}
_tokens: dict[str, dict[str, Any]] = {}
_MAX_TOKENS = 32


def is_enabled() -> bool:
    """Whether the embedded terminal capability is turned on (default off)."""
    from rune.cloud.boundary import hosted

    if hosted():
        return False
    if os.environ.get("RUNE_TERMINAL_ENABLED", "").strip() in ("1", "true", "yes"):
        return True
    try:
        from rune.config import get_config

        return bool(getattr(get_config(), "terminal_enabled", False))
    except Exception:
        return False


def mint_token(workspace: str) -> str:
    """Mint a workspace-bound, single-use token after the caller checks is_enabled()."""
    import secrets

    if len(_tokens) >= _MAX_TOKENS:
        # Evict spent tokens first, then the oldest, to bound unused token storage.
        spent = next((k for k, v in _tokens.items() if v.get("used")), None)
        _tokens.pop(spent if spent is not None else next(iter(_tokens)), None)
    token = secrets.token_urlsafe(24)
    _tokens[token] = {"workspace": workspace, "used": False}
    return token


def redeem_token(token: str) -> str | None:
    """Consume *token*, returning its workspace, or None if invalid/spent."""
    entry = _tokens.get(token)
    if entry is None or entry.get("used"):
        return None
    entry["used"] = True
    return entry.get("workspace") or ""


class TerminalSession:
    """Read a PTY on a thread and stream its output through an asyncio queue."""

    # Pause PTY reads when the queue fills so kernel backpressure bounds memory without data loss.
    _MAX_QUEUE = 256
    _RESUME_AT = 64

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace if Path(workspace).is_dir() else str(Path.home())
        self._pid: int = -1
        self._fd: int = -1
        self.out_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=self._MAX_QUEUE
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._reader_paused = False

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        shell = os.environ.get("SHELL", "/bin/bash")
        pid, fd = os.forkpty()
        if pid == 0:
            # Child: minimal, predictable environment in the workspace.
            try:
                os.chdir(self._workspace)
            except OSError:
                pass
            os.environ["TERM"] = "xterm-256color"
            os.execvp(shell, [shell])
            os._exit(1)  # unreachable on success
        self._pid = pid
        self._fd = fd
        self._loop.add_reader(fd, self._on_readable)

    def _on_readable(self) -> None:
        try:
            data = os.read(self._fd, 65536)
        except OSError:
            data = b""
        if not data:
            self.close()
            return
        self.out_queue.put_nowait(data)
        # Pause until notify_consumed re-arms the reader after the queue drains.
        if self.out_queue.qsize() >= self._MAX_QUEUE - 1 and not self._reader_paused:
            self._reader_paused = True
            if self._loop is not None and self._fd >= 0:
                with contextlib.suppress(Exception):
                    self._loop.remove_reader(self._fd)

    def notify_consumed(self) -> None:
        """Resume PTY reads after the WebSocket pump drains enough queued output."""
        if (
            self._reader_paused
            and not self._closed
            and self.out_queue.qsize() <= self._RESUME_AT
            and self._loop is not None
            and self._fd >= 0
        ):
            self._reader_paused = False
            with contextlib.suppress(Exception):
                self._loop.add_reader(self._fd, self._on_readable)

    def write(self, data: str) -> None:
        if self._fd >= 0 and not self._closed:
            with contextlib.suppress(OSError):
                os.write(self._fd, data.encode("utf-8", "replace"))

    def resize(self, rows: int, cols: int) -> None:
        if self._fd < 0 or self._closed:
            return
        with contextlib.suppress(OSError):
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, winsize)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._fd >= 0:
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self._fd)
        if self._pid > 0:
            pid = self._pid
            # Kill the shell, then reap off-thread to avoid zombies without blocking the event loop.
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(pid, signal.SIGKILL)

            def _reap(target: int) -> None:
                with contextlib.suppress(ChildProcessError, OSError):
                    os.waitpid(target, 0)

            threading.Thread(target=_reap, args=(pid,), daemon=True).start()
            self._pid = -1
        if self._fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1
        # Make room for the close sentinel without blocking on a full queue.
        try:
            self.out_queue.put_nowait(None)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self.out_queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self.out_queue.put_nowait(None)
