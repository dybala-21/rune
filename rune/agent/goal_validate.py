"""Deterministic SPEC validation runner for the ``/goal`` loop.

The outer loop runs the SPEC's validation commands itself and checks exit
codes, rather than trusting the inner completion gate or the agent's
self-report. Commands run sequentially and non-interactively with a
per-command timeout; the first failure short-circuits. The shell exec is
injected so this can be unit-tested without spawning processes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from rune.utils.logger import get_logger

log = get_logger(__name__)

# (command, cwd, timeout_s) -> (exit_code, combined_output)
ExecFn = Callable[[str, str, float], Awaitable[tuple[int, str]]]


async def _default_exec(command: str, cwd: str, timeout_s: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd or None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,  # non-interactive
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"timeout after {timeout_s:.0f}s"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


def make_validate_fn(
    *,
    cwd: str = "",
    timeout_s: float = 600.0,
    exec_fn: ExecFn | None = None,
) -> Callable[[list[str]], Awaitable[tuple[bool, str]]]:
    """Build a ``GoalLoop`` ``validate_fn``. Empty command list => pass."""
    run = exec_fn or _default_exec

    async def _validate(commands: list[str]) -> tuple[bool, str]:
        if not commands:
            return True, "no validation commands"
        transcript: list[str] = []
        for cmd in commands:
            try:
                code, output = await run(cmd, cwd, timeout_s)
            except Exception as exc:  # treat exec failure as validation failure
                log.debug("goal_validate_exec_error", cmd=cmd[:120], error=str(exc)[:200])
                return False, f"`{cmd}` could not run: {exc}"[:300]
            tail = "\n".join(output.strip().splitlines()[-8:])[:600]
            transcript.append(f"$ {cmd}\n[exit {code}]\n{tail}".rstrip())
            if code != 0:
                # Deterministic evidence the reviewer can trust (not a claim).
                return False, "DETERMINISTIC VALIDATION OUTPUT:\n" + "\n\n".join(transcript)
        return True, "DETERMINISTIC VALIDATION OUTPUT (all commands exited 0):\n" + "\n\n".join(
            transcript
        )

    return _validate
