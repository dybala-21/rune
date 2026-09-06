"""Track checks after the latest code change within a run.

Sequence numbers count tool events, including multiple events in one LLM turn.
They establish check freshness, not test coverage or program correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.agent.auto_verify import assertions_ran
from rune.agent.bash_parsing import is_verification_command


@dataclass(slots=True)
class VerificationState:
    sequence: int = 0
    last_write: int = 0
    last_pass: int = 0
    last_test_pass: int = 0
    last_check_failed: bool = False
    command: str = ""

    def changed(self) -> None:
        self.sequence += 1
        self.last_write = self.sequence

    def observe_command(self, command: str, success: bool, output: str) -> bool:
        """Record a recognized check and return whether it passed.

        Quiet linters and builds use the exit code but do not count as tests.
        Failed checks and empty test suites invalidate earlier passes.
        """
        self.sequence += 1
        if not is_verification_command(command):
            return False
        self.command = command
        asserted = assertions_ran(output)
        self.last_check_failed = not success or asserted is False
        if self.last_check_failed:
            return False
        self.last_pass = self.sequence
        if asserted is True:
            self.last_test_pass = self.sequence
        return True

    @property
    def passed(self) -> bool:
        return self.last_pass > self.last_write and not self.last_check_failed

    @property
    def pending(self) -> bool:
        return self.last_check_failed or (self.last_write > 0 and not self.passed)

    @property
    def tests_passed_after_edit(self) -> bool | None:
        if not self.last_write:
            return None
        return self.last_test_pass > self.last_write and not self.last_check_failed

    def snapshot(self) -> dict[str, Any]:
        return {
            "required": self.last_write > 0,
            "status": "pass" if self.passed else (
                "fail" if self.last_check_failed else "unverified"
            ),
            "last_write": self.last_write,
            "last_pass": self.last_pass,
            "command": self.command,
            "tests_passed_after_edit": self.tests_passed_after_edit,
        }


def verified_outcome(result: Any) -> bool | None:
    """Return the verification outcome, or None when evidence is absent.

    Explicit failures take precedence. Code changes require a fresh check;
    other traces can use the evidence gate or mechanical check result.
    """
    def get(name: str, default: Any = None) -> Any:
        return result.get(name, default) if isinstance(result, dict) else getattr(
            result, name, default
        )

    reason = get("reason", "")
    if reason and reason not in ("completed", "verified"):
        return False
    if get("success") is False:
        return False
    verification = get("verification") or {}
    gate = get("evidence_gate") or {}
    verdict = gate.get("last_verdict")
    mech = get("mech_check", "")
    if verdict == "fail" or mech == "fail":
        return False
    if verification.get("required") or verification.get("status") == "fail":
        return verification.get("status") == "pass"
    if get("tests_passed_after_edit") is False:
        return False
    if verdict == "pass" or mech == "pass" or verification.get("status") == "pass":
        return True
    return None
