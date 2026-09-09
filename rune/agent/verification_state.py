"""Track check results and their order relative to code changes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rune.agent.auto_verify import assertions_ran, tests_failed
from rune.agent.check_commands import check_commands


@dataclass(slots=True)
class CheckResult:
    command: str
    cwd: str
    kind: str
    sequence: int
    write_sequence: int
    status: str
    asserted: bool | None


@dataclass(slots=True)
class VerificationState:
    sequence: int = 0
    last_write: int = 0
    last_pass: int = 0
    last_test_pass: int = 0
    command: str = ""
    checks: dict[tuple[str, str], CheckResult] = field(default_factory=dict)

    def changed(self) -> None:
        self.sequence += 1
        self.last_write = self.sequence

    def observe_command(self, command: str, success: bool, output: str, cwd: str = "") -> bool:
        """Record recognized checks and return True if any passed.

        Results replace only the same command in the same directory.
        A passing subset cannot clear a failed full suite.
        """
        self.sequence += 1
        invocations = check_commands(command, cwd)
        if not invocations:
            return False
        self.command = command
        asserted = assertions_ran(output)
        failed = not success or tests_failed(output) or asserted is False
        passed = False
        for invocation in invocations:
            status = "fail" if failed else (
                "pass" if invocation.reliable_exit else "inconclusive"
            )
            self.checks[invocation.key] = CheckResult(
                invocation.command, invocation.cwd, invocation.kind, self.sequence,
                self.last_write, status, asserted if invocation.kind == "test" else None,
            )
            if status == "pass":
                self.last_pass = self.sequence
                passed = True
                if invocation.kind == "test" and asserted is True:
                    self.last_test_pass = self.sequence
        return passed

    @property
    def last_check_failed(self) -> bool:
        return any(check.status == "fail" for check in self.checks.values())

    @property
    def unresolved(self) -> list[CheckResult]:
        return [check for check in self.checks.values()
                if check.status != "pass" or check.sequence <= self.last_write]

    def guidance(self) -> str:
        checks = "; ".join(f"{check.command} (cwd: {check.cwd})" for check in self.unresolved)
        return ("Re-run these checks directly after the latest change: " + checks
                if checks else "Run a verification after the latest code change.")

    @property
    def passed(self) -> bool:
        return self.last_pass > self.last_write and not self.unresolved

    @property
    def pending(self) -> bool:
        return bool(self.unresolved) or (self.last_write > 0 and not self.passed)

    @property
    def tests_passed_after_edit(self) -> bool | None:
        if not self.last_write:
            return None
        tests = [check for check in self.checks.values() if check.kind == "test"]
        return bool(tests) and all(
            check.status == "pass" and check.sequence > self.last_write and check.asserted is True
            for check in tests
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "required": self.last_write > 0,
            "status": "pass" if self.passed else (
                "fail" if self.last_check_failed else (
                    "inconclusive" if self.unresolved else "unverified"
                )
            ),
            "last_write": self.last_write,
            "last_pass": self.last_pass,
            "command": self.command,
            "tests_passed_after_edit": self.tests_passed_after_edit,
            "checks": [asdict(check) for check in self.checks.values()],
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
    table = get("table_acceptance") or {}
    gate = get("evidence_gate") or {}
    verdict = gate.get("last_verdict")
    mech = get("mech_check", "")
    if verdict == "fail" or mech == "fail" or verification.get("status") == "fail":
        return False
    if table.get("required") and table.get("status") != "pass":
        return False if table.get("status") == "fail" else None
    if verification.get("status") == "inconclusive":
        return False if verification.get("required") else None
    if verification.get("required") or verification.get("status") == "fail":
        return verification.get("status") == "pass"
    if get("tests_passed_after_edit") is False:
        return False
    if verdict == "pass" or mech == "pass" or verification.get("status") == "pass" or table.get("status") == "pass":
        return True
    return None
