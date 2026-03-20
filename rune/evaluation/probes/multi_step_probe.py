"""Multi-Step Probe for RUNE evaluation system.

Tests an agent's ability to perform a compound task:
1. Read numbers from an input file.
2. Calculate their sum.
3. Write the result to an output file.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult


class MultiStepProbe(Probe):
    """Probe that verifies multi-step file I/O task completion."""

    @property
    def name(self) -> str:
        return "multi-step-probe"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        """Run the multi-step probe.

        Parameters:
            context: Must contain ``"agent"`` with an async ``run(prompt)``
                method.

        Returns:
            ProbeResult indicating whether the output file contains the
            correct sum.
        """
        agent = context.get("agent")
        if agent is None:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output="No agent provided in context",
                expected="",
                score=0.0,
            )

        numbers = [10, 20, 30, 40, 50]
        expected_sum = sum(numbers)  # 150
        timestamp = int(time.time() * 1000)
        tmp = tempfile.gettempdir()
        input_file = os.path.join(tmp, f"rune-input-{timestamp}.txt")
        output_file = os.path.join(tmp, f"rune-output-{timestamp}.txt")

        try:
            # Write input file
            Path(input_file).write_text(
                "\n".join(str(n) for n in numbers), encoding="utf-8",
            )

            prompt = (
                "Do the following steps:\n"
                f'1. Read the numbers from "{input_file}" (one number per line)\n'
                "2. Calculate their sum\n"
                f'3. Write ONLY the sum (just the number, nothing else) to "{output_file}"\n\n'
                "After completing, confirm the task is done."
            )

            await agent.run(prompt)

            # Verify output
            actual_output = ""
            passed = False
            try:
                actual_output = Path(output_file).read_text(encoding="utf-8").strip()
                actual_sum = int(actual_output)
                passed = actual_sum == expected_sum
            except (FileNotFoundError, ValueError):
                actual_output = actual_output or "FILE_NOT_FOUND"

            return ProbeResult(
                probe_name=self.name,
                success=passed,
                output=actual_output[:100],
                expected=str(expected_sum),
                score=1.0 if passed else 0.0,
            )

        except Exception as exc:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Exception: {exc}",
                expected=str(expected_sum),
                score=0.0,
            )

        finally:
            for f in (input_file, output_file):
                with contextlib.suppress(OSError):
                    os.unlink(f)


MULTI_STEP_PROBE_CONFIG: dict[str, Any] = {
    "name": "multi-step-probe",
    "description": "Tests agent ability to perform multi-step tasks",
    "timeout": 120_000,
    "retries": 1,
    "enabled": True,
}
