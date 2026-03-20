"""Probe runner for RUNE evaluation system.

Manages probe registration and execution, producing EvaluationRun results.
"""

from __future__ import annotations

import time
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import EvaluationRun, ProbeResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


class ProbeRunner:
    """Manages and executes evaluation probes.

    Probes are registered by name and can be run individually or
    collectively to produce an EvaluationRun.
    """

    __slots__ = ("_probes",)

    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}

    def register(self, probe: Probe) -> None:
        """Register a probe for execution.

        Parameters:
            probe: The probe instance to register.
        """
        self._probes[probe.name] = probe
        log.debug("probe_registered", name=probe.name)

    async def run_all(self, context: dict[str, Any]) -> EvaluationRun:
        """Run all registered probes and return an EvaluationRun.

        Parameters:
            context: Execution context passed to each probe.

        Returns:
            An EvaluationRun with all probe results and total score.
        """
        results: list[ProbeResult] = []

        for name in sorted(self._probes):
            result = await self.run_probe(name, context)
            results.append(result)

        total_score = 0.0
        if results:
            total_score = sum(r.score for r in results) / len(results)

        run = EvaluationRun(
            probe_results=results,
            total_score=round(total_score, 4),
        )

        log.info(
            "evaluation_run_complete",
            run_id=run.id,
            probes=len(results),
            total_score=run.total_score,
        )
        return run

    async def run_probe(self, name: str, context: dict[str, Any]) -> ProbeResult:
        """Run a single probe by name.

        Parameters:
            name: The name of the probe to run.
            context: Execution context passed to the probe.

        Returns:
            The ProbeResult from the probe execution.

        Raises:
            KeyError: If no probe with the given name is registered.
        """
        probe = self._probes.get(name)
        if probe is None:
            raise KeyError(f"No probe registered with name: {name}")

        start = time.monotonic()
        try:
            result = await probe.run(context)
            elapsed_ms = (time.monotonic() - start) * 1000
            result.duration_ms = round(elapsed_ms, 2)

            log.debug(
                "probe_executed",
                name=name,
                success=result.success,
                score=result.score,
                duration_ms=result.duration_ms,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            log.error("probe_failed", name=name, error=str(exc))
            return ProbeResult(
                probe_name=name,
                success=False,
                output=f"Exception: {exc}",
                expected="",
                duration_ms=round(elapsed_ms, 2),
                score=0.0,
            )
