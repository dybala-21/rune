"""Command execution probe for RUNE evaluation system.

Tests the bash.execute capability by running a simple echo command
and verifying the output matches the expected string.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

_TEST_TOKEN = "RUNE_EXEC_PROBE_OK_7734"


class ExecProbe(Probe):
    """Tests the bash.execute capability.

    Runs a simple ``echo`` command and verifies the output.
    """

    @property
    def name(self) -> str:
        return "bash.execute"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        """Execute the bash probe.

        Parameters:
            context: May contain an "exec_fn" async callable for custom
                command execution. Falls back to asyncio subprocess.

        Returns:
            ProbeResult indicating whether command execution works correctly.
        """
        command = f"echo {_TEST_TOKEN}"

        try:
            exec_fn = context.get("exec_fn")
            if exec_fn is not None:
                output = await exec_fn(command)
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                output = stdout.decode("utf-8", errors="replace").strip()

            success = output.strip() == _TEST_TOKEN
            score = 1.0 if success else 0.0

            return ProbeResult(
                probe_name=self.name,
                success=success,
                output=output.strip(),
                expected=_TEST_TOKEN,
                score=score,
            )

        except Exception as exc:
            log.error("exec_probe_error", error=str(exc))
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Error: {exc}",
                expected=_TEST_TOKEN,
                score=0.0,
            )
