"""Self-Validation Probe for RUNE evaluation system.

Tests the agent's core capabilities (file create, read, exec, delete) via
an end-to-end self-validation task and verifies by inspecting the agent's
action history and the actual filesystem state.
"""

from __future__ import annotations

import os
import time
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult


class SelfValidationProbe(Probe):
    """Probe that asks the agent to perform file/exec operations and
    verifies completion through history inspection and filesystem checks.
    """

    @property
    def name(self) -> str:
        return "self-validation-probe"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        agent = context.get("agent")
        if agent is None:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output="No agent provided in context",
                expected="3+/4 checks passed",
                score=0.0,
            )

        timestamp = int(time.time() * 1000)
        test_file = f"/tmp/rune-self-test-{timestamp}.txt"
        test_content = f"VALIDATION_OK_{timestamp}"

        # Clean up before test
        try:
            if os.path.exists(test_file):
                os.unlink(test_file)
        except OSError:
            pass

        try:
            prompt = (
                "Perform these file operations in order:\n"
                f"1. Create file {test_file} with content: {test_content}\n"
                "2. Read the file to verify the content\n"
                '3. Run: echo "EXEC_CHECK_OK"\n'
                f"4. Delete the file {test_file}\n\n"
                "Complete all 4 steps."
            )

            result = await agent.run(prompt)

            # Inspect history if available
            history: list[Any] = getattr(result, "history", []) or []
            history_str = " ".join(
                getattr(h, "content", str(h)) for h in history
            )

            checks = {
                "file_create": (
                    "file" in history_str.lower()
                    and (
                        "write" in history_str.lower()
                        or "create" in history_str.lower()
                        or test_file in history_str
                    )
                ),
                "file_read": (
                    "file" in history_str.lower() and "read" in history_str.lower()
                ),
                "exec_test": "EXEC_CHECK_OK" in history_str,
                "file_delete": (
                    "rm " in history_str
                    or "unlink" in history_str.lower()
                    or "delete" in history_str.lower()
                ),
            }

            # Physical file check - should have been deleted
            file_still_exists = os.path.exists(test_file)
            if not file_still_exists:
                checks["file_delete"] = True

            passed_count = sum(checks.values())
            all_passed = passed_count >= 3

            report_lines = [
                f"File Create: {'PASS' if checks['file_create'] else 'FAIL'}",
                f"File Read:   {'PASS' if checks['file_read'] else 'FAIL'}",
                f"Exec Test:   {'PASS' if checks['exec_test'] else 'FAIL'}",
                f"File Delete: {'PASS' if checks['file_delete'] else 'FAIL'}",
            ]

            return ProbeResult(
                probe_name=self.name,
                success=all_passed,
                output=f"{passed_count}/4 checks passed\n" + "\n".join(report_lines),
                expected="3+/4 checks passed",
                score=round(passed_count / 4, 2),
            )

        except Exception as exc:
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Exception: {exc}",
                expected="3+/4 checks passed",
                score=0.0,
            )

        finally:
            try:
                if os.path.exists(test_file):
                    os.unlink(test_file)
            except OSError:
                pass


SELF_VALIDATION_PROBE_CONFIG: dict[str, Any] = {
    "name": "self-validation-probe",
    "description": "Agent performs end-to-end self-validation",
    "timeout": 180_000,
    "retries": 1,
    "enabled": True,
}
