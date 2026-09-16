"""Launch the native app through macOS and keep its pipes private to this run."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from rune.computer.protocol import DesktopError
from rune.utils.logger import get_logger

log = get_logger(__name__)
MAX_REPLY = 32 * 1024 * 1024


class _InputPipe:
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.pending = bytearray()
        self.ready: asyncio.Future | None = None

    def write(self, data: bytes) -> None:
        self.pending.extend(data)

    async def drain(self) -> None:
        loop = asyncio.get_running_loop()
        while self.pending:
            try:
                count = os.write(self.fd, self.pending)
                del self.pending[:count]
            except BlockingIOError:
                ready = loop.create_future()
                self.ready = ready
                fd = self.fd
                loop.add_writer(fd, lambda ready=ready: None if ready.done() else ready.set_result(None))
                try:
                    await ready
                finally:
                    if self.fd == fd:
                        loop.remove_writer(fd)
                    self.ready = None

    def close(self) -> None:
        if self.fd >= 0:
            if self.ready:
                asyncio.get_running_loop().remove_writer(self.fd)
                if not self.ready.done():
                    self.ready.set_exception(BrokenPipeError("The native input pipe was closed"))
            os.close(self.fd)
            self.fd = -1
        self.pending.clear()


def host_path() -> Path:
    from rune.utils.paths import rune_home
    return rune_home() / "native" / "Rune Computer.app" / "Contents" / "MacOS" / "RuneComputer"


class MacHost:
    def __init__(self, executable: Path | None = None) -> None:
        self.executable = executable
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.control_fd: int | None = None
        self.epoch = 0
        self.input: _InputPipe | None = None
        self.output: asyncio.StreamReader | None = None
        self.read_transport: asyncio.ReadTransport | None = None
        self.directory: tempfile.TemporaryDirectory | None = None
        self.generation = 0

    @property
    def connected(self) -> bool:
        return bool(self.process and self.process.returncode is None and self.input and self.output
                    and self.control_fd is not None)

    async def _start(self) -> None:
        if sys.platform != "darwin":
            raise DesktopError("Native desktop access currently requires macOS 14 or later.")
        executable = self.executable or host_path()
        bundle = executable.parents[2]
        if not executable.is_file() or bundle.suffix != ".app":
            raise DesktopError("Install the native host with: python -m rune.computer.build")
        await self.close()
        self.directory = tempfile.TemporaryDirectory(prefix="rune-desktop-")
        directory = Path(self.directory.name)
        for name in ("input", "output", "control"):
            os.mkfifo(directory / name, 0o600)
        self.input = _InputPipe(os.open(directory / "input", os.O_RDWR | os.O_NONBLOCK))
        self.control_fd = os.open(directory / "control", os.O_RDWR | os.O_NONBLOCK)
        output = os.open(directory / "output", os.O_RDONLY | os.O_NONBLOCK)
        # Hold the reader open until Launch Services attaches the app's stdout.
        keeper = os.open(directory / "output", os.O_WRONLY | os.O_NONBLOCK)
        self.epoch = 0
        try:
            self.output = asyncio.StreamReader(limit=MAX_REPLY)
            output_file = os.fdopen(output, "rb", buffering=0)
            try:
                self.read_transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                    lambda: asyncio.StreamReaderProtocol(self.output), output_file)
            except BaseException:
                output_file.close()
                raise
            self.process = await asyncio.create_subprocess_exec(
                "/usr/bin/open", "-n", "-W", "-g", "--stdin", str(directory / "input"),
                "--stdout", str(directory / "output"), "--stderr", "/dev/null", str(bundle),
                "--args", str(directory / "control"),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env={key: os.environ[key] for key in ("HOME", "TMPDIR", "LANG") if key in os.environ},
            )
            async with asyncio.timeout(15):
                ready = asyncio.create_task(self.output.readline())
                ended = asyncio.create_task(self.process.wait())
                try:
                    completed, _ = await asyncio.wait((ready, ended), return_when=asyncio.FIRST_COMPLETED)
                    if ready not in completed or not ready.result():
                        raise DesktopError("Rune Computer could not start. Rebuild the native app and retry.")
                    hello = json.loads(ready.result())
                    if (hello.get("protocol") != 3 or hello.get("bundleId") != "dev.rune.computer"
                            or Path(hello.get("appPath", "")).resolve() != bundle.resolve()):
                        raise DesktopError("Rune Computer needs an update. Rebuild the native app and retry.")
                finally:
                    ready.cancel()
                    ended.cancel()
                    await asyncio.gather(ready, ended, return_exceptions=True)
        finally:
            os.close(keeper)

    def cancel_pending(self) -> None:
        if self.control_fd is not None:
            self.epoch += 1
            try:
                os.write(self.control_fd, b"\x01")
            except OSError:
                # EOF also revokes the native lease if the pipe fills or breaks.
                os.close(self.control_fd)
                self.control_fd = None

    async def request(self, method: str, params: dict | None = None, *, guard: Callable[[], None] | None = None) -> dict:
        async with self.lock:
            dispatched = False
            try:
                if guard is not None:
                    guard()
                if self.process is None or self.process.returncode is not None:
                    try:
                        await self._start()
                    except BaseException:
                        await self.close()
                        raise
                request_id = uuid4().hex
                payload = json.dumps({"id": request_id, "epoch": self.epoch, "method": method, "params": params or {}}, allow_nan=False).encode() + b"\n"
                if len(payload) > 32 * 1024:
                    raise DesktopError("Native request exceeds the size limit.")
                assert self.input and self.output
                if guard is not None:
                    guard()
                self.input.write(payload)
                dispatched = True
                async with asyncio.timeout(180 if method in {"grant", "permissions", "act"} else 15):
                    await self.input.drain()
                    raw = await self.output.readline()
                if not raw or len(raw) > MAX_REPLY:
                    raise ValueError("Missing or oversized native response")
                reply = json.loads(raw)
                if reply.get("id") != request_id:
                    raise ValueError("Native response identity mismatch")
                if reply.get("ok") is not True:
                    outcome = reply.get("outcome")
                    raise DesktopError(str(reply.get("error", "Native operation failed")),
                                       outcome=outcome if outcome in {"unknown", "not_executed"} else "unknown")
                if not isinstance(reply.get("data"), dict):
                    raise ValueError("Invalid native result")
                return reply["data"]
            except DesktopError:
                raise
            except BaseException as exc:
                await self.close()
                if not isinstance(exc, Exception):
                    raise
                log.warning("desktop_transport_failed", method=method, error=type(exc).__name__)
                raise DesktopError("The native host did not confirm the operation. Inspect the app before continuing.",
                                   outcome="unknown" if dispatched and method == "act" else "not_executed") from exc

    async def close(self) -> None:
        self.generation += 1
        process, self.process = self.process, None
        control_fd, self.control_fd = self.control_fd, None
        writer, self.input = self.input, None
        transport, self.read_transport = self.read_transport, None
        directory, self.directory = self.directory, None
        self.output = None
        if control_fd is not None:
            os.close(control_fd)
        if writer:
            writer.close()
        try:
            if process is not None and process.returncode is None:
                try:
                    async with asyncio.timeout(3):
                        await process.wait()
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        log.debug("desktop_host_already_exited")
                    await process.wait()
        finally:
            if transport:
                transport.close()
            if directory:
                directory.cleanup()
