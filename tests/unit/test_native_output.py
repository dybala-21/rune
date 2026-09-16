"""Send native responses through the same named FIFO and async reader as Rune."""

import asyncio
import os
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_output(tmp_path_factory):
    if sys.platform != "darwin" or not shutil.which("swiftc"):
        pytest.skip("Requires macOS and the Swift toolchain")
    root = tmp_path_factory.mktemp("native-output")
    source = Path(__file__).parents[2] / "rune/computer/native/CommandOutput.swift"
    harness = root / "Harness.swift"
    harness.write_text('''
import AppKit
import Foundation

@main struct Harness {
    @MainActor static func main() {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let mode = CommandLine.arguments[1]
        let size = Int(CommandLine.arguments[2])!
        let output = try! CommandOutput(.standardOutput)
        var revoked = false
        let task = Task { @MainActor in
            do {
                var data = Data(String(repeating: "한글→", count: size / 9 + 1).utf8).prefix(size)
                data.append(10)
                FileHandle.standardError.write(Data("started\\n".utf8))
                try await output.write(data) {
                    if revoked { throw CancellationError() }
                }
                try await output.write(Data("next response\\n".utf8))
                exit(0)
            } catch {
                do {
                    try await output.write(Data("must not follow a partial response\\n".utf8))
                    exit(3)
                } catch {
                    FileHandle.standardError.write(Data("stopped\\n".utf8))
                    exit(2)
                }
            }
        }
        Timer.scheduledTimer(withTimeInterval: 0.15, repeats: false) { _ in
            MainActor.assumeIsolated {
                FileHandle.standardError.write(Data("responsive\\n".utf8))
                if mode == "cancel" { task.cancel() }
                if mode == "revoke" { revoked = true }
            }
        }
        NSApplication.shared.run()
    }
}
''')
    executable = root / "harness"
    built = subprocess.run(["swiftc", "-parse-as-library", "-module-cache-path", str(root / "cache"),
        str(source), str(harness), "-o", str(executable)], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stderr
    return executable


@asynccontextmanager
async def response_pipe(executable, directory, size, mode="stream", paused=False):
    fifo = directory / "output"
    os.mkfifo(fifo, 0o600)
    fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    writer = os.open(fifo, os.O_WRONLY)
    reader = asyncio.StreamReader(limit=32 * 1024 * 1024)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(fd, "rb", buffering=0))
    if paused:
        transport.pause_reading()
    process = None
    try:
        process = await asyncio.create_subprocess_exec(str(executable), mode, str(size),
            stdin=asyncio.subprocess.DEVNULL, stdout=writer, stderr=asyncio.subprocess.PIPE)
        os.close(writer)
        writer = None
        await marker(process.stderr, b"started\n")
        yield process, reader, transport
    finally:
        if writer is not None:
            os.close(writer)
        if process is not None:
            if process.returncode is None:
                process.kill()
            await process.communicate()
        transport.close()


async def marker(stream, expected):
    async with asyncio.timeout(3):
        while line := await stream.readline():
            if line == expected:
                return
    pytest.fail(f"Native process exited before {expected!r}")


def run_with_loop(factory, coro):
    if factory == "uvloop":
        loop_factory = pytest.importorskip("uvloop").new_event_loop
    else:
        loop_factory = asyncio.new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(coro)


@pytest.mark.parametrize("factory", ["asyncio", "uvloop"])
@pytest.mark.parametrize("size", [8192, 1024 * 1024, 28 * 1024 * 1024])
def test_native_large_reply_preserves_bytes_and_next_response(native_output, tmp_path, factory, size):
    async def run():
        async with response_pipe(native_output, tmp_path, size) as (process, reader, _):
            unit = "한글→".encode()
            expected = (unit * (size // len(unit) + 1))[:size] + b"\n"
            assert await asyncio.wait_for(reader.readline(), 3) == expected
            assert await asyncio.wait_for(reader.readline(), 3) == b"next response\n"
            assert await asyncio.wait_for(process.wait(), 3) == 0
    run_with_loop(factory, run())


async def test_native_backpressure_keeps_appkit_responsive(native_output, tmp_path):
    async with response_pipe(native_output, tmp_path, 1024 * 1024, paused=True) as (process, reader, transport):
        await marker(process.stderr, b"responsive\n")
        assert process.returncode is None
        transport.resume_reading()
        assert len(await asyncio.wait_for(reader.readline(), 3)) == 1024 * 1024 + 1
        assert await asyncio.wait_for(reader.readline(), 3) == b"next response\n"
        assert await asyncio.wait_for(process.wait(), 3) == 0


@pytest.mark.parametrize("mode", ["cancel", "revoke", "disconnect"])
async def test_native_stops_partial_reply_on_cancel_or_disconnect(native_output, tmp_path, mode):
    async with response_pipe(native_output, tmp_path, 1024 * 1024, mode, paused=True) as (process, _, transport):
        await marker(process.stderr, b"responsive\n")
        if mode == "disconnect":
            transport.close()
        await marker(process.stderr, b"stopped\n")
        assert await asyncio.wait_for(process.wait(), 3) == 2
