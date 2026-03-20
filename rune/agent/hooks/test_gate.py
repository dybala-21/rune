"""Test gate - test execution gate for task completion.

Ported from src/agent/hooks/test-gate.ts (129 lines) - task_completed
hook that runs configured test/typecheck commands and blocks or warns
based on results.

Intent-aware: skips gate when code verification is not required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from rune.agent.hooks.runner import HookHandler, HookResult, TaskCompletedContext
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

TestGateMode = str  # "off" | "advisory" | "required"


@dataclass(slots=True)
class TestGateConfig:
    """Configuration for the test gate hook."""

    mode: TestGateMode = "advisory"
    commands: list[str] = field(
        default_factory=lambda: ["npm run -s typecheck", "npm run -s test"]
    )
    timeout_ms: int = 10 * 60 * 1000  # 10 minutes
    only_when_files_change: bool = True
    include_extensions: list[str] = field(
        default_factory=lambda: [".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml"]
    )
    exclude_path_prefixes: list[str] = field(
        default_factory=lambda: ["node_modules/", "dist/", ".git/", "coverage/", ".rune/"]
    )


DEFAULT_TEST_GATE_CONFIG = TestGateConfig()


# Helpers

def _should_gate_by_files(files: list[str], config: TestGateConfig) -> bool:
    """Determine if the test gate should run based on changed files."""
    if not config.only_when_files_change:
        return True
    if not files:
        return False

    for file in files:
        normalized = file.replace("\\", "/")

        # Skip excluded paths
        if any(
            normalized.startswith(prefix) or f"/{prefix}" in normalized
            for prefix in config.exclude_path_prefixes
        ):
            continue

        # Check included extensions
        if any(normalized.endswith(ext) for ext in config.include_extensions):
            return True

    return False


async def _run_command(
    command: str,
    timeout_ms: int,
) -> tuple[bool, str]:
    """Run a shell command asynchronously. Returns (success, output)."""
    timeout_s = timeout_ms / 1000.0
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        output = "\n".join(filter(None, [stdout, stderr])).strip()
        success = proc.returncode == 0
        return success, output
    except TimeoutError:
        return False, f"Command timed out after {timeout_s:.0f}s"
    except Exception as exc:
        return False, str(exc)


# Hook factory

def create_test_gate_hook(
    config: TestGateConfig | None = None,
) -> HookHandler:
    """Create a task_completed hook for test gating.

    Returns a handler that runs configured test commands after
    task completion and reports results.
    """
    cfg = config or DEFAULT_TEST_GATE_CONFIG

    async def handler(context: TaskCompletedContext) -> HookResult:
        if cfg.mode == "off":
            return HookResult(decision="pass")

        # Intent-aware: skip if code verification not required
        if context.requires_code_verification is False:
            return HookResult(
                decision="pass",
                metadata={"skipped": True, "reason": "code verification not required by intent"},
            )

        should_run = _should_gate_by_files(context.changed_files, cfg)
        if not should_run:
            return HookResult(
                decision="pass",
                metadata={"skipped": True, "reason": "no relevant file changes"},
            )

        failures: list[dict[str, str]] = []
        for command in cfg.commands:
            success, output = await _run_command(command, cfg.timeout_ms)
            if not success:
                failures.append({"command": command, "output": output})

        if not failures:
            return HookResult(
                decision="pass",
                metadata={"ran": len(cfg.commands), "failures": 0},
            )

        summary = "\n\n".join(
            f"({i + 1}) {f['command']}\n{f['output'][:400]}"
            for i, f in enumerate(failures)
        )

        log.warning(
            "test_gate_failed",
            commands=[f["command"] for f in failures],
            mode=cfg.mode,
        )

        if cfg.mode == "required":
            return HookResult(
                decision="block",
                message=f"Test gate failed:\n{summary}",
                metadata={"failures": len(failures)},
            )

        return HookResult(
            decision="warn",
            message=f"Test gate advisory failure:\n{summary}",
            metadata={"failures": len(failures)},
        )

    return handler
