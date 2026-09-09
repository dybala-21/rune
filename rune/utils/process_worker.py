"""Bounded JSON requests to a child interpreter."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from rune.utils.logger import get_logger

log = get_logger(__name__)

MAX_MESSAGE = 16 * 1024 * 1024


class WorkerUnavailable(RuntimeError):
    pass


def _timeout(sock: socket.socket, deadline: float | None) -> None:
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("Worker deadline exceeded")
    sock.settimeout(remaining)


def send_message(sock: socket.socket, value: dict[str, Any], deadline: float | None) -> None:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError("Worker message exceeds 16 MiB")
    _timeout(sock, deadline)
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive_message(sock: socket.socket, deadline: float | None) -> dict[str, Any]:
    def read(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            _timeout(sock, deadline)
            part = sock.recv(min(size - len(chunks), 65536))
            if not part:
                raise EOFError("Worker connection closed")
            chunks.extend(part)
        return bytes(chunks)

    size = struct.unpack("!I", read(4))[0]
    if size > MAX_MESSAGE:
        raise ValueError("Worker response exceeds 16 MiB")
    value = json.loads(read(size))
    if not isinstance(value, dict):
        raise ValueError("Worker response must be an object")
    return value


def watch_parent() -> None:
    """Exit a busy child if its parent disappears."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch() -> None:
        while parent.is_alive():
            time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="rune-parent-watch").start()


class ProcessWorker:
    def __init__(
        self, target: Callable[..., None], *, args: tuple[Any, ...] = (),
        cooldown: float = 60.0,
    ) -> None:
        self._target = target
        self._args = args
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._process: Any = None
        self._socket: socket.socket | None = None
        self._retry_at = 0.0
        self.generation = 0
        self.last_error: str | None = None

    @property
    def status(self) -> dict[str, Any]:
        process = self._process
        try:
            running = process is not None and process.is_alive()
            pid = process.pid if process is not None else None
        except ValueError:
            log.debug("worker_status_after_close")
            running, pid = False, None
        return {
            "state": "unavailable" if self._closed.is_set() else (
                "degraded" if self.last_error else (
                    "ready" if running else "idle"
                )
            ),
            "generation": self.generation,
            "pid": pid,
            "last_error": self.last_error,
        }

    def _start(self) -> None:
        parent, child = socket.socketpair()
        process = mp.get_context("spawn").Process(
            target=self._target, args=(child, *self._args), daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self._socket = parent
        self._process = process
        self.generation += 1

    def _stop(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        process = self._process
        if process is not None:
            process.join(0.1)
            if process.is_alive():
                process.terminate()
                process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join(0.5)
            if process.is_alive():
                raise WorkerUnavailable("Worker did not exit after kill")
            self._process = None
            process.close()

    def request(self, payload: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        if not self._lock.acquire(timeout=max(0, timeout)):
            raise WorkerUnavailable("Worker queue deadline exceeded")
        try:
            if self._closed.is_set():
                raise WorkerUnavailable("Worker is closed")
            if time.monotonic() < self._retry_at:
                raise WorkerUnavailable(self.last_error or "Worker cooling down")
            try:
                if self._process is not None and not self._process.is_alive():
                    raise EOFError(f"Worker exited ({self._process.exitcode})")
                if self._process is None:
                    self._start()
                if self._closed.is_set():
                    raise WorkerUnavailable("Worker is closed")
                assert self._socket is not None
                request_id = uuid4().hex
                send_message(self._socket, {"id": request_id, "payload": payload}, deadline)
                response = receive_message(self._socket, deadline)
                if response.get("id") != request_id:
                    raise ValueError("Worker response identity mismatch")
                if response.get("error"):
                    raise WorkerUnavailable(str(response["error"]))
                result = response.get("result")
                if not isinstance(result, dict):
                    raise ValueError("Worker result must be an object")
                self.last_error = None
                return result
            except Exception as exc:
                # Keep input text out of supervisor errors and health responses.
                self.last_error = type(exc).__name__
                self._retry_at = time.monotonic() + self._cooldown
                self._stop()
                raise WorkerUnavailable(f"Worker unavailable: {self.last_error}") from exc
        finally:
            self._lock.release()

    def close(self) -> None:
        self._closed.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError as exc:
                log.debug("worker_socket_already_closed", error=type(exc).__name__)
        with self._lock:
            self._stop()

    def invalidate(self, reason: str) -> None:
        with self._lock:
            self.last_error = reason
            self._retry_at = time.monotonic() + self._cooldown
            self._stop()
