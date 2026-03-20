"""Bugfix Probe for RUNE evaluation system.

Tests the agent's ability to identify and fix a bug (off-by-one error)
in a Python script, then verifies the fix by executing the script.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult

_BUGGY_CODE = """\
# sum_up_to(n) should return the sum of 1 to n inclusive
def sum_up_to(n):
    total = 0
    for i in range(1, n):  # BUG: should be range(1, n + 1)
        total += i
    return total

print(sum_up_to(5))
"""


class BugfixProbe(Probe):
    """Probe that presents a buggy Python script and asks the agent to fix it."""

    @property
    def name(self) -> str:
        return "bugfix"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        agent = context.get("agent")
        if agent is None:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output="No agent provided in context",
                expected="15",
                score=0.0,
            )

        timestamp = int(time.time() * 1000)
        test_file = os.path.join(
            tempfile.gettempdir(), f"rune-bugfix-probe-{timestamp}.py",
        )

        try:
            Path(test_file).write_text(_BUGGY_CODE, encoding="utf-8")

            prompt = (
                f'The file "{test_file}" has a bug. Running it prints 10, '
                "but it should print 15 (sum of 1+2+3+4+5). Fix the bug in "
                f'the file, then run it with "python {test_file}" to verify '
                "the output is 15."
            )

            await agent.run(prompt)

            # Host-side verification
            output = ""
            try:
                cp = subprocess.run(
                    ["python", test_file],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                output = cp.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                output = "EXEC_FAILED"

            passed = "15" in output

            return ProbeResult(
                probe_name=self.name,
                success=passed,
                output=output[:200],
                expected="15",
                score=1.0 if passed else 0.0,
            )

        except Exception as exc:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Exception: {exc}",
                expected="15",
                score=0.0,
            )

        finally:
            with contextlib.suppress(OSError):
                os.unlink(test_file)


BUGFIX_PROBE_CONFIG: dict[str, Any] = {
    "name": "bugfix",
    "description": "Tests agent ability to find and fix a bug in Python code",
    "timeout": 120_000,
    "retries": 1,
    "enabled": True,
}
