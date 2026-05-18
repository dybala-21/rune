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


# auto_root: deterministically locate the project root when the agent put it
# in a subdir (crystallize<->validate cwd mismatch fix).


def fake_exec_cwd():
    seen: list[tuple[str, str]] = []

    async def _exec(cmd: str, cwd: str, timeout_s: float) -> tuple[int, str]:
        seen.append((cmd, cwd))
        return 0, "ok"

    return _exec, seen


async def test_root_manifest_no_redirect(tmp_path) -> None:
    # well-formed project (manifest at root) -> unchanged behavior, no header.
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(cwd=str(tmp_path), exec_fn=ex)
    passed, detail = await v(["cargo build"])
    assert passed is True
    assert seen == [("cargo build", str(tmp_path))]
    assert "# validation cwd:" not in detail


async def test_single_subdir_manifest_redirects(tmp_path) -> None:
    # the observed failure: agent built in rust_web/ ; validate must follow.
    (tmp_path / "rust_web").mkdir()
    (tmp_path / "rust_web" / "Cargo.toml").write_text("[package]\n")
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(cwd=str(tmp_path), exec_fn=ex)
    passed, detail = await v(["cargo build"])
    assert passed is True
    assert seen == [("cargo build", str(tmp_path / "rust_web"))]
    assert f"# validation cwd: {tmp_path / 'rust_web'}" in detail


async def test_ambiguous_multiple_manifests_stays_root(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "Cargo.toml").write_text("")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "go.mod").write_text("")
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(cwd=str(tmp_path), exec_fn=ex)
    await v(["build"])
    assert seen == [("build", str(tmp_path))]  # conservative: stay at root


async def test_no_manifest_anywhere_stays_root(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main(){}")
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(cwd=str(tmp_path), exec_fn=ex)
    await v(["build"])
    assert seen == [("build", str(tmp_path))]


async def test_manifest_in_build_output_is_ignored(tmp_path) -> None:
    # a Cargo.toml inside target/ must NOT count as the project root.
    (tmp_path / "target" / "x").mkdir(parents=True)
    (tmp_path / "target" / "x" / "Cargo.toml").write_text("")
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(cwd=str(tmp_path), exec_fn=ex)
    await v(["build"])
    assert seen == [("build", str(tmp_path))]  # excluded dir -> stay at root


async def test_auto_root_disabled_stays_at_cwd(tmp_path) -> None:
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "go.mod").write_text("module x\n")
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(cwd=str(tmp_path), exec_fn=ex, auto_root=False)
    await v(["go build"])
    assert seen == [("go build", str(tmp_path))]


async def test_empty_cwd_unchanged(tmp_path) -> None:
    # the existing call pattern (cwd="") must never scan or add a header.
    ex, seen = fake_exec_cwd()
    v = make_validate_fn(exec_fn=ex)
    passed, detail = await v(["pytest"])
    assert seen == [("pytest", "")]
    assert "# validation cwd:" not in detail
