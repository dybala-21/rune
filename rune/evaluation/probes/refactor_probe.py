"""Refactor Probe for RUNE evaluation system.

Tests the agent's ability to refactor messy-but-working code into clean,
readable code while preserving its exact behaviour.
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

_EXPECTED_OUTPUT = "[1, 2, 3, 4, 5, 6, 9]"

_MESSY_CODE = """\
# Deduplicate and sort a list of numbers
def d(a):
    if len(a) == 0:
        return []
    r = []
    for i in range(len(a)):
        f = True
        for j in range(len(r)):
            if r[j] == a[i]:
                f = False
                break
        if f:
            r.append(a[i])
    r.sort()
    return r

print(d([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]))
"""


class RefactorProbe(Probe):
    """Probe that asks the agent to refactor messy code while preserving output."""

    @property
    def name(self) -> str:
        return "refactor"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        agent = context.get("agent")
        if agent is None:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output="No agent provided in context",
                expected=_EXPECTED_OUTPUT,
                score=0.0,
            )

        timestamp = int(time.time() * 1000)
        test_file = os.path.join(
            tempfile.gettempdir(), f"rune-refactor-probe-{timestamp}.py",
        )

        try:
            Path(test_file).write_text(_MESSY_CODE, encoding="utf-8")

            prompt = (
                f'Refactor the code in "{test_file}" to be clean and readable: '
                "use meaningful variable/function names, modern Python idioms "
                "(set, sorted, list comprehensions if appropriate). Do NOT change "
                f"the behaviour -- the output must remain exactly '{_EXPECTED_OUTPUT}'. "
                f'After refactoring, run it with "python {test_file}" to verify '
                "the output is unchanged."
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

            passed = output == _EXPECTED_OUTPUT

            return ProbeResult(
                probe_name=self.name,
                success=passed,
                output=output[:200],
                expected=_EXPECTED_OUTPUT,
                score=1.0 if passed else 0.0,
            )

        except Exception as exc:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Exception: {exc}",
                expected=_EXPECTED_OUTPUT,
                score=0.0,
            )

        finally:
            with contextlib.suppress(OSError):
                os.unlink(test_file)


REFACTOR_PROBE_CONFIG: dict[str, Any] = {
    "name": "refactor",
    "description": "Tests agent ability to refactor messy code while preserving behavior",
    "timeout": 120_000,
    "retries": 1,
    "enabled": True,
}
