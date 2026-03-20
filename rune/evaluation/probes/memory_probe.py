"""Memory Probe for RUNE evaluation system.

Tests an agent's ability to track information within a long context.
A secret code is presented, followed by a distraction task, then the
agent is asked to recall the secret code.
"""

from __future__ import annotations

import secrets
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult


class MemoryProbe(Probe):
    """Probe that tests single-conversation information tracking.

    Steps executed inside a single prompt:
    1. Present a random secret code.
    2. Request a distraction calculation.
    3. Ask the agent to recall the secret code.
    """

    @property
    def name(self) -> str:
        return "memory-probe"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        """Run the memory recall probe.

        Parameters:
            context: Must contain ``"agent"`` - an object with an async
                ``run(prompt: str)`` method returning a result with an
                ``answer`` attribute.

        Returns:
            ProbeResult indicating whether the secret was recalled.
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

        secret_code = f"SECRET_{secrets.token_hex(4).upper()}"

        prompt = (
            "This is a memory test with multiple parts.\n\n"
            f"PART 1: I'm giving you a secret code to remember: {secret_code}\n\n"
            "PART 2: Now, as a distraction, please calculate 847 + 293.\n\n"
            "PART 3: After the calculation, tell me the secret code from PART 1.\n\n"
            "Your final answer MUST end with the exact line:\n"
            "SECRET_CODE: [the code from PART 1]"
        )

        try:
            result = await agent.run(prompt)
            answer: str = getattr(result, "answer", "") or ""

            passed = secret_code in answer
            return ProbeResult(
                probe_name=self.name,
                success=passed,
                output=answer[:300],
                expected=secret_code,
                score=1.0 if passed else 0.0,
            )

        except Exception as exc:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Exception: {exc}",
                expected=secret_code,
                score=0.0,
            )


MEMORY_PROBE_CONFIG: dict[str, Any] = {
    "name": "memory-probe",
    "description": "Tests agent ability to remember context across turns",
    "timeout": 120_000,
    "retries": 1,
    "enabled": True,
}
