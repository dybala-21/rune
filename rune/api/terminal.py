"""Embedded terminal: a PTY hosted in the daemon, streamed to xterm.js.

Jupyter/terminado's architecture (server-side PTY over a socket to xterm.js),
reimplemented on the stdlib so the Electron shell stays "dumb" — no node-pty,
no native module, no new IPC surface in the renderer.

Security posture (deliberately conservative — an interactive terminal is
arbitrary code execution for whoever reaches the socket):

- **Off by default, and that is the real boundary.** Enabled only when
  ``RUNE_TERMINAL_ENABLED=1`` (or the config flag). Honest limitation: once
  enabled, a script running *inside the renderer's own origin* (e.g. a
  markdown-XSS in chat) shares the loopback origin that the auth guard trusts,
  so it CAN call ``terminal.token`` and open a shell — enabling the terminal
  makes a renderer compromise equivalent to a local shell. The per-open token
  only stops *cross-site* pages (which fail the CSRF/origin bypass and cannot
  mint). So: keep it off unless you need it; a truly un-forgeable gate would
  require a native (main-process) confirmation dialog, which is future work.
- Gating the *capability*, not keystrokes: per-command filtering of a live
  shell has no real boundary (the user can spawn bash/python). Guardian keeps
  gating what the *agent* runs; this is a separate, opt-in human capability.
- The WebSocket handshake checks default-off, loopback, the single-use token,
  and the request Origin (defense-in-depth against cross-site handshakes).

POSIX only (stdlib ``pty``/``os``). Windows needs pywinpty — out of scope here.
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
    if os.environ.get("RUNE_TERMINAL_ENABLED", "").strip() in ("1", "true", "yes"):
        return True
    try:
        from rune.config import get_config

        return bool(getattr(get_config(), "terminal_enabled", False))
    except Exception:
        return False


def mint_token(workspace: str) -> str:
    """Mint a one-shot terminal token bound to *workspace*. Caller must have
    already checked :func:`is_enabled`."""
    import secrets

    if len(_tokens) >= _MAX_TOKENS:
        # Prefer evicting a spent token; otherwise drop the oldest so repeated
        # minting without connecting can't grow the map without bound.
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
    """One PTY-backed shell. Reads run on a thread and land on an asyncio queue
    so the WebSocket handler can await them."""

    # Bound the output queue and apply real backpressure: when a flooding
    # shell (`yes`, `cat bigfile`) fills it, stop draining the PTY so the
    # kernel's PTY buffer fills and the shell's write() blocks — natural Unix
    # flow control, no data loss and no unbounded memory growth. Resume once
    # the consumer drains below the low-water mark.
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
        # If the consumer is behind, stop reading the PTY. The shell's next
        # write() then blocks on the full PTY buffer — backpressure, not memory
        # growth. notify_consumed() re-arms the reader once drained.
        if self.out_queue.qsize() >= self._MAX_QUEUE - 1 and not self._reader_paused:
            self._reader_paused = True
            if self._loop is not None and self._fd >= 0:
                with contextlib.suppress(Exception):
                    self._loop.remove_reader(self._fd)

    def notify_consumed(self) -> None:
        """Called by the WebSocket pump after draining a chunk; re-arms the PTY
        reader once the backlog is low enough."""
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
            # SIGKILL the shell (can't be trapped, unlike SIGHUP). Killing the
            # session-leader shell makes the kernel SIGHUP the foreground
            # process group, so a pipeline like `yes | head` dies too. Then
            # reap on a daemon thread: WNOHANG here would return before the
            # just-killed child is dead and leave a zombie/leak (measured);
            # a blocking waitpid off the event loop reaps without stalling it.
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
        # Sentinel so the pump ends. The queue is bounded, but close() must
        # never block or lose the sentinel — evict one chunk if full.
        try:
            self.out_queue.put_nowait(None)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self.out_queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self.out_queue.put_nowait(None)
