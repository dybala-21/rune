"""Function Generation Probe for RUNE evaluation system.

Tests the agent's ability to generate a correct function implementation
with tests.  The agent is asked to create a Fibonacci function and a
self-test script, then the script is executed to verify correctness.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import time
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult


class FunctionGenProbe(Probe):
    """Probe that asks the agent to generate a Fibonacci function with tests."""

    @property
    def name(self) -> str:
        return "function-gen"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        agent = context.get("agent")
        if agent is None:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output="No agent provided in context",
                expected="ALL_PASS",
                score=0.0,
            )

        timestamp = int(time.time() * 1000)
        test_file = os.path.join(
            tempfile.gettempdir(), f"rune-fgen-{timestamp}.py",
        )

        try:
            prompt = (
                f'Create a single file at "{test_file}" that:\n'
                "1. Defines a function fibonacci(n) that returns the nth "
                "Fibonacci number (fibonacci(0)=0, fibonacci(1)=1)\n"
                "2. Tests it: fibonacci(0)==0, fibonacci(1)==1, "
                "fibonacci(6)==8, fibonacci(10)==55\n"
                '3. Prints "ALL_PASS" if all tests pass, "FAIL" otherwise\n\n'
                f'Then run it with "python {test_file}".'
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

            passed = "ALL_PASS" in output

            return ProbeResult(
                probe_name=self.name,
                success=passed,
                output=output[:200],
                expected="ALL_PASS",
                score=1.0 if passed else 0.0,
            )

        except Exception as exc:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Exception: {exc}",
                expected="ALL_PASS",
                score=0.0,
            )

        finally:
            with contextlib.suppress(OSError):
                os.unlink(test_file)


FUNCTION_GEN_PROBE_CONFIG: dict[str, Any] = {
    "name": "function-gen",
    "description": "Tests agent ability to generate a function and write tests",
    "timeout": 180_000,
    "retries": 1,
    "enabled": True,
}
