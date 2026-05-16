"""Tests for rune.agent.goal_validate — deterministic SPEC validation runner.

The shell exec is injected; no processes are spawned.
"""

from __future__ import annotations

from rune.agent.goal_validate import make_validate_fn


def fake_exec(script: dict[str, tuple[int, str]]):
    seen: list[str] = []

    async def _exec(cmd: str, cwd: str, timeout_s: float) -> tuple[int, str]:
        seen.append(cmd)
        if cmd not in script:
            raise RuntimeError(f"unscripted: {cmd}")
        return script[cmd]

    return _exec, seen


async def test_no_commands_passes() -> None:
    v = make_validate_fn(exec_fn=fake_exec({})[0])
    assert await v([]) == (True, "no validation commands")


async def test_all_pass() -> None:
    ex, seen = fake_exec({"pytest": (0, "ok"), "ruff": (0, "clean")})
    v = make_validate_fn(exec_fn=ex)

    passed, detail = await v(["pytest", "ruff"])

    assert passed is True
    assert "all commands exited 0" in detail
    assert "$ pytest" in detail and "$ ruff" in detail  # transcript = evidence
    assert seen == ["pytest", "ruff"]


async def test_first_failure_short_circuits() -> None:
    ex, seen = fake_exec({"pytest": (1, "E   assert False\n2 failed")})
    v = make_validate_fn(exec_fn=ex)

    passed, detail = await v(["pytest", "ruff"])

    assert passed is False
    assert "$ pytest" in detail and "[exit 1]" in detail
    assert seen == ["pytest"]  # ruff never ran


async def test_exec_exception_is_failure() -> None:
    async def boom(cmd: str, cwd: str, timeout_s: float) -> tuple[int, str]:
        raise OSError("no shell")

    v = make_validate_fn(exec_fn=boom)
    passed, detail = await v(["pytest"])

    assert passed is False
    assert "could not run" in detail


async def test_timeout_surfaces_as_failure() -> None:
    ex, _ = fake_exec({"slow": (124, "timeout after 600s")})
    v = make_validate_fn(exec_fn=ex)

    passed, detail = await v(["slow"])

    assert passed is False
    assert "[exit 124]" in detail
