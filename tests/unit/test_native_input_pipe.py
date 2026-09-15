import asyncio
import os

import pytest

from rune.computer.macos import _InputPipe


@pytest.mark.parametrize("close_while_full", [False, True])
async def test_native_pipe_backpressure_and_disconnect(close_while_full):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    os.set_blocking(write_fd, False)
    writer = _InputPipe(write_fd)
    data = bytes(range(256)) * 4096
    writer.write(data)
    task = asyncio.create_task(writer.drain())
    try:
        async with asyncio.timeout(3):
            while writer.ready is None:
                await asyncio.sleep(0)
            if close_while_full:
                writer.close()
                with pytest.raises(BrokenPipeError):
                    await task
            else:
                received = bytearray()
                while len(received) < len(data):
                    try:
                        received.extend(os.read(read_fd, 65536))
                    except BlockingIOError:
                        await asyncio.sleep(0)
                await task
                assert received == data
                assert not writer.pending
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        writer.close()
        os.close(read_fd)
