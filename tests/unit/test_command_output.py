"""Exercise real command output through the verification tracker."""

import asyncio
import os
import shlex
import sys
from contextlib import suppress

import pytest

from rune.agent.verification_state import VerificationState
from rune.capabilities.bash import BashParams, _execute_oneshot


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["pass", "fail", "empty"])
async def test_unittest_stderr_reaches_verification(tmp_path, outcome):
    if outcome != "empty":
        (tmp_path / "test_example.py").write_text(
            "import unittest\n"
            "class Example(unittest.TestCase):\n"
            f"    def test_value(self): self.assertEqual(1, {1 if outcome == 'pass' else 2})\n"
        )
    command = shlex.join([sys.executable, "-m", "unittest", "discover", "-v"])
    result = await _execute_oneshot(BashParams(command=command, cwd=str(tmp_path)))
    assert f"Ran {0 if outcome == 'empty' else 1} test" in result.output
    if outcome != "empty":
        assert result.success is (outcome == "pass")
    assert bool(result.error) is not result.success
    state = VerificationState()
    state.changed()
    state.observe_command(command, result.success, result.output, str(tmp_path))
    assert state.tests_passed_after_edit is (outcome == "pass")


@pytest.mark.asyncio
async def test_successful_command_preserves_both_streams(tmp_path):
    command = shlex.join([sys.executable, "-c", "import sys; print('result'); print('warning', file=sys.stderr)"])
    result = await _execute_oneshot(BashParams(command=command, cwd=str(tmp_path)))
    assert result.success and result.error is None
    assert "result" in result.output and "warning" in result.output
    assert result.metadata["exit_code"] == 0
    assert result.metadata["action_status"] == "completed"


@pytest.mark.asyncio
async def test_large_test_output_keeps_last_summary_without_losing_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr("rune.capabilities.bash.DEFAULT_OUTPUT_BUFFER_LIMIT", 1024)
    (tmp_path / "test_verbose.py").write_text(
        "import sys, unittest\n"
        "class Verbose(unittest.TestCase):\n"
        "    def test_output(self):\n"
        "        print('start')\n"
        "        print('x' * 200000)\n"
        "        print('한' * 100000, file=sys.stderr)\n"
    )
    command = shlex.join([sys.executable, "-m", "unittest", "discover", "-v"])
    result = await _execute_oneshot(BashParams(command=command, cwd=str(tmp_path)))
    assert result.success and "start" in result.output
    assert "Ran 1 test" in result.output and result.output.rstrip().endswith("OK")
    assert len(result.output) < 2400
    state = VerificationState()
    state.changed()
    state.observe_command(command, result.success, result.output, str(tmp_path))
    assert state.tests_passed_after_edit


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_noisy_command_is_reaped_on_timeout_or_cancellation(tmp_path, cancel):
    pid_file = tmp_path / "pid"
    code = f"import os; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid()))\nwhile True: os.write(1, b'x' * 8192); os.write(2, b'y' * 8192)"
    command = "exec " + shlex.join([sys.executable, "-c", code])
    task = asyncio.create_task(_execute_oneshot(BashParams(command=command, timeout=500 if not cancel else 10000)))
    async with asyncio.timeout(3):
        while not pid_file.exists():
            await asyncio.sleep(.005)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert not result.success and result.metadata["timeout"]
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires process groups")
async def test_timeout_stops_children_even_after_the_shell_exits(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = f"import os,time; from pathlib import Path\npid=os.fork()\nif pid: os._exit(0)\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\nwhile True: os.write(1,b'x'*8192);time.sleep(.01)"
    command = "exec " + shlex.join([sys.executable, "-c", code])
    try:
        result = await asyncio.wait_for(_execute_oneshot(BashParams(command=command, timeout=300)), 4)
        assert result.metadata.get("timeout") is True
    finally:
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 9)


@pytest.mark.parametrize("limit", [2, 3, 127, 8192])
def test_chunk_buffer_keeps_exact_head_tail_across_unicode_and_tiny_reads(limit):
    from rune.capabilities.command_output import OutputBuffer

    buffer = OutputBuffer(limit)
    original = ""
    for part in ["", "한", "🚀", "x" * 20000, *list("small reads" * 600), "끝"]:
        original += part
        buffer.append(part)
        if len(original) <= limit:
            expected = original
        else:
            expected = (original[:limit // 2] + f"\n... ({len(original) - limit} characters omitted) ...\n"
                        + original[-(limit - limit // 2):])
        assert buffer.render() == expected
    assert len(buffer._head) + len(buffer._tail) <= limit // 2048 + 4
