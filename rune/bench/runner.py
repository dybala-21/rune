"""Minimal local benchmark runner for RUNE agent attempts."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SNAPSHOT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".rune",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}
_TEXT_DIFF_MAX_BYTES = 1_000_000
BENCHMARK_PROMPT_POLICY = "aa-coding-agent-v1"
DEFAULT_INSTALL_FINGERPRINT_PATH = Path("/logs/agent/rune_install_fingerprint.json")
BENCHMARK_INSTRUCTION_PREFIX = """You are running in an unattended coding-agent benchmark.

Benchmark rules and operating constraints:
- Complete the task end-to-end in this container. Do not ask follow-up questions.
- Do not finish with a partial solution, a blocker report, or a request for the user to continue.
- If a required compiler, runtime, package manager, or utility is missing, install it when possible.
- Use only task files and public resources. Do not inspect hidden evaluator paths such as /tests or /oracle.
- Do not inspect VCS history with commands such as git log, git show, git blame, or reflog
  unless the task explicitly asks you to use repository history.
- Run available in-workspace checks, examples, or self-written sanity tests before finalizing.
- Before finalizing, run at least one clean-process smoke test through the public entrypoint
  using the required command or function name and likely evaluator call style.
- Match the requested file paths, function signatures, command names, output formats, and argument order exactly.
- When a public function or CLI is named without a full signature, support natural positional usage
  and common named aliases in addition to your preferred named-argument style.
- If your current approach fails a check, keep debugging within the benchmark budget instead of stopping.
"""


@dataclass(slots=True)
class FileSnapshot:
    digest: str
    size: int
    text: str | None = None


@dataclass(slots=True)
class BenchRunOptions:
    benchmark: str
    task_id: str
    instruction: str
    output_dir: Path
    rune_home: Path
    cwd: Path
    attempt_index: int = 1
    model: str | None = None
    provider: str | None = None
    memory_mode: str = "default"
    dry_run: bool = False
    agent_variant_id: str | None = None


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)[:180]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _run_short_command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _selected_env(keys: tuple[str, ...]) -> dict[str, str]:
    return {key: value for key in keys if (value := os.environ.get(key))}


def _install_fingerprint_path() -> Path:
    raw_path = os.environ.get("RUNE_BENCH_INSTALL_FINGERPRINT")
    return Path(raw_path) if raw_path else DEFAULT_INSTALL_FINGERPRINT_PATH


def collect_runtime_fingerprint(options: BenchRunOptions) -> dict[str, Any]:
    """Collect enough provenance to prove which RUNE executable ran this attempt."""
    rune_file: str | None
    rune_version: str | None
    rune_import_error: str | None
    try:
        import rune
    except Exception as exc:  # pragma: no cover - impossible in normal runner execution.
        rune_file = None
        rune_version = None
        rune_import_error = str(exc)
    else:
        rune_file = getattr(rune, "__file__", None)
        rune_version = getattr(rune, "__version__", None)
        rune_import_error = None

    try:
        package_version = importlib.metadata.version("rune-ai")
    except importlib.metadata.PackageNotFoundError:
        package_version = None

    install_fingerprint_path = _install_fingerprint_path()
    install_fingerprint = _read_json(install_fingerprint_path)

    return {
        "agent": "rune",
        "agent_variant_id": options.agent_variant_id
        or os.environ.get("RUNE_BENCH_AGENT_VARIANT_ID"),
        "benchmark": options.benchmark,
        "task_id": options.task_id,
        "attempt_index": options.attempt_index,
        "benchmark_prompt_policy": BENCHMARK_PROMPT_POLICY,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "rune": {
            "module_file": rune_file,
            "module_version": rune_version,
            "package_version": package_version,
            "import_error": rune_import_error,
            "cli_version": _run_short_command(["rune", "--version"]),
        },
        "install_fingerprint_path": str(install_fingerprint_path),
        "install_fingerprint": install_fingerprint,
        "env": _selected_env(
            (
                "RUNE_BENCH_AGENT_VARIANT_ID",
                "RUNE_BENCH_EXPECT_AGENT_VARIANT_ID",
                "RUNE_BENCH_EXPECT_INSTALL_MODE",
                "RUNE_BENCH_EXPECT_PROMPT_POLICY",
                "RUNE_BENCH_EXPECT_SOURCE_GIT_SHA",
                "RUNE_BENCH_EXPECT_WHEELHOUSE_SHA256",
                "RUNE_BENCH_BLOCK_VCS_HISTORY",
                "RUNE_BENCH_ALLOW_VCS_HISTORY",
                "RUNE_BENCH_REQUIRE_FINGERPRINT",
                "RUNE_BENCH_TASK_ID",
                "RUNE_MODEL",
                "RUNE_PROVIDER",
            )
        ),
    }


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def evaluate_fingerprint_gate(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Evaluate optional expected provenance values for official benchmark runs."""
    errors: list[str] = []
    warnings: list[str] = []

    if fingerprint.get("benchmark_prompt_policy") != BENCHMARK_PROMPT_POLICY:
        errors.append("benchmark prompt policy mismatch")

    if not fingerprint.get("rune", {}).get("module_file"):
        errors.append("missing runtime rune module file")

    expected_policy = os.environ.get("RUNE_BENCH_EXPECT_PROMPT_POLICY")
    if expected_policy and expected_policy != BENCHMARK_PROMPT_POLICY:
        errors.append(
            f"expected prompt policy {expected_policy!r}, got {BENCHMARK_PROMPT_POLICY!r}"
        )

    expected_variant = os.environ.get("RUNE_BENCH_EXPECT_AGENT_VARIANT_ID")
    actual_variant = fingerprint.get("agent_variant_id")
    if expected_variant and actual_variant != expected_variant:
        errors.append(f"expected agent variant {expected_variant!r}, got {actual_variant!r}")

    install = fingerprint.get("install_fingerprint")
    install = install if isinstance(install, dict) else {}

    expected_install_mode = os.environ.get("RUNE_BENCH_EXPECT_INSTALL_MODE")
    actual_install_mode = install.get("install_mode")
    if expected_install_mode and actual_install_mode != expected_install_mode:
        errors.append(
            f"expected install mode {expected_install_mode!r}, got {actual_install_mode!r}"
        )

    expected_source_git_sha = os.environ.get("RUNE_BENCH_EXPECT_SOURCE_GIT_SHA")
    actual_source_git_sha = install.get("source_git_sha")
    if expected_source_git_sha and actual_source_git_sha != expected_source_git_sha:
        errors.append(
            f"expected source git sha {expected_source_git_sha!r}, got {actual_source_git_sha!r}"
        )

    expected_wheelhouse_sha = os.environ.get("RUNE_BENCH_EXPECT_WHEELHOUSE_SHA256")
    actual_wheelhouse_sha = install.get("wheelhouse_sha256")
    if expected_wheelhouse_sha and actual_wheelhouse_sha != expected_wheelhouse_sha:
        errors.append(
            f"expected wheelhouse sha256 {expected_wheelhouse_sha!r}, "
            f"got {actual_wheelhouse_sha!r}"
        )

    if _env_flag("RUNE_BENCH_REQUIRE_FINGERPRINT"):
        if not actual_variant:
            errors.append("RUNE_BENCH_REQUIRE_FINGERPRINT requires agent_variant_id")
        if not actual_install_mode:
            errors.append("RUNE_BENCH_REQUIRE_FINGERPRINT requires install_mode")
        if actual_install_mode == "wheelhouse" and not actual_wheelhouse_sha:
            errors.append("wheelhouse install requires wheelhouse_sha256")
        if not install:
            errors.append("RUNE_BENCH_REQUIRE_FINGERPRINT requires install_fingerprint")
    elif not install:
        warnings.append("install_fingerprint not found; run is acceptable only for local smoke tests")

    return {
        "valid": not errors,
        "required": _env_flag("RUNE_BENCH_REQUIRE_FINGERPRINT"),
        "errors": errors,
        "warnings": warnings,
    }


def build_agent_instruction(options: BenchRunOptions) -> str:
    """Return the benchmark policy prompt plus the task instruction."""
    return (
        f"{BENCHMARK_INSTRUCTION_PREFIX}\n"
        f"Benchmark: {options.benchmark}\n"
        f"Task ID: {options.task_id}\n\n"
        "Task instruction:\n"
        f"{options.instruction}"
    )


def _git_diff(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--binary", "--", "."],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode not in (0, 1):
        return ""

    diff = result.stdout
    for rel_path in _git_untracked_files(cwd):
        try:
            untracked = subprocess.run(
                ["git", "diff", "--binary", "--no-index", "--", "/dev/null", rel_path],
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception:
            continue
        if untracked.stdout:
            diff += "\n" + untracked.stdout
    return diff


def _git_untracked_files(cwd: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _git_status(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout if result.returncode == 0 else ""


def _is_git_repo(cwd: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _iter_snapshot_paths(cwd: Path) -> list[Path]:
    paths: list[Path] = []
    for path in cwd.rglob("*"):
        rel_parts = path.relative_to(cwd).parts
        if any(part in _SNAPSHOT_EXCLUDE_DIRS for part in rel_parts):
            continue
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def _file_snapshot(path: Path) -> FileSnapshot | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    text = None
    if len(data) <= _TEXT_DIFF_MAX_BYTES:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
    return FileSnapshot(
        digest=hashlib.sha256(data).hexdigest(),
        size=len(data),
        text=text,
    )


def _snapshot_files(cwd: Path) -> dict[str, FileSnapshot]:
    snapshot: dict[str, FileSnapshot] = {}
    for path in _iter_snapshot_paths(cwd):
        file_snapshot = _file_snapshot(path)
        if file_snapshot is not None:
            snapshot[str(path.relative_to(cwd))] = file_snapshot
    return snapshot


def _filesystem_status(cwd: Path, before: dict[str, FileSnapshot]) -> str:
    after = _snapshot_files(cwd)
    rows: list[str] = []
    for rel_path in sorted(set(before) | set(after)):
        if rel_path not in before:
            rows.append(f"?? {rel_path}")
        elif rel_path not in after:
            rows.append(f" D {rel_path}")
        elif before[rel_path].digest != after[rel_path].digest:
            rows.append(f" M {rel_path}")
    return "\n".join(rows) + ("\n" if rows else "")


def _filesystem_diff(cwd: Path, before: dict[str, FileSnapshot]) -> str:
    after = _snapshot_files(cwd)
    chunks: list[str] = []
    for rel_path in sorted(set(before) | set(after)):
        old = before.get(rel_path)
        new = after.get(rel_path)
        if old and new and old.digest == new.digest:
            continue

        if old is None:
            if new and new.text is not None:
                chunks.extend(
                    difflib.unified_diff(
                        [],
                        new.text.splitlines(keepends=True),
                        fromfile="/dev/null",
                        tofile=f"b/{rel_path}",
                    )
                )
            else:
                chunks.append(f"Binary file b/{rel_path} created ({new.size if new else 0} bytes)\n")
        elif new is None:
            if old.text is not None:
                chunks.extend(
                    difflib.unified_diff(
                        old.text.splitlines(keepends=True),
                        [],
                        fromfile=f"a/{rel_path}",
                        tofile="/dev/null",
                    )
                )
            else:
                chunks.append(f"Binary file a/{rel_path} deleted ({old.size} bytes)\n")
        elif old.text is not None and new.text is not None:
            chunks.extend(
                difflib.unified_diff(
                    old.text.splitlines(keepends=True),
                    new.text.splitlines(keepends=True),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                )
            )
        else:
            chunks.append(f"Binary file {rel_path} changed ({old.size} -> {new.size} bytes)\n")
    return "".join(chunks)


def create_attempt_dir(options: BenchRunOptions) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_dir = (
        options.output_dir
        / _safe_path_part(options.benchmark)
        / _safe_path_part(options.task_id)
        / f"attempt-{options.attempt_index:03d}-{timestamp}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir


def write_attempt_inputs(attempt_dir: Path, options: BenchRunOptions) -> None:
    _write_json(
        attempt_dir / "task.json",
        {
            "benchmark": options.benchmark,
            "task_id": options.task_id,
            "attempt_index": options.attempt_index,
            "instruction": options.instruction,
        },
    )
    _write_json(
        attempt_dir / "agent_config.json",
        {
            "agent": "rune",
            "model": options.model,
            "provider": options.provider,
            "memory_mode": options.memory_mode,
            "benchmark_prompt_policy": BENCHMARK_PROMPT_POLICY,
            "agent_variant_id": options.agent_variant_id
            or os.environ.get("RUNE_BENCH_AGENT_VARIANT_ID"),
            "attempt_index": options.attempt_index,
            "rune_home": str(options.rune_home),
            "cwd": str(options.cwd),
            "dry_run": options.dry_run,
        },
    )
    (attempt_dir / "effective_instruction.txt").write_text(
        build_agent_instruction(options),
        encoding="utf-8",
    )


def run_bench_attempt(options: BenchRunOptions) -> Path:
    """Run one benchmark attempt and return its artifact directory."""
    attempt_dir = create_attempt_dir(options)
    write_attempt_inputs(attempt_dir, options)
    fingerprint = collect_runtime_fingerprint(options)
    fingerprint_gate = evaluate_fingerprint_gate(fingerprint)
    _write_json(attempt_dir / "fingerprint.json", fingerprint)
    _write_json(attempt_dir / "fingerprint_gate.json", fingerprint_gate)
    if fingerprint_gate["required"] and not fingerprint_gate["valid"]:
        _write_json(
            attempt_dir / "completion_trace.json",
            {
                "reason": "fingerprint_gate_failed",
                "errors": fingerprint_gate["errors"],
            },
        )
        _write_json(attempt_dir / "timing.json", {"agent_wall_time_ms": 0})
        _write_json(attempt_dir / "audit.json", {"source_diff_present": False})
        _write_jsonl(attempt_dir / "events.jsonl", [])
        _write_jsonl(attempt_dir / "tool_calls.jsonl", [])
        _write_jsonl(attempt_dir / "model_usage.jsonl", [])
        (attempt_dir / "final_answer.txt").write_text("", encoding="utf-8")
        (attempt_dir / "patch.diff").write_text("", encoding="utf-8")
        return attempt_dir
    if options.dry_run:
        _write_json(attempt_dir / "completion_trace.json", {"reason": "dry_run"})
        _write_json(attempt_dir / "timing.json", {"agent_wall_time_ms": 0})
        _write_json(attempt_dir / "audit.json", {"source_diff_present": False})
        _write_jsonl(attempt_dir / "events.jsonl", [])
        _write_jsonl(attempt_dir / "tool_calls.jsonl", [])
        _write_jsonl(attempt_dir / "model_usage.jsonl", [])
        (attempt_dir / "final_answer.txt").write_text("", encoding="utf-8")
        (attempt_dir / "patch.diff").write_text("", encoding="utf-8")
        return attempt_dir

    asyncio.run(_run_agent(options, attempt_dir))
    return attempt_dir


async def _run_agent(options: BenchRunOptions, attempt_dir: Path) -> None:
    old_rune_home = os.environ.get("RUNE_HOME")
    old_cwd = Path.cwd()
    start = time.monotonic()
    output_parts: list[str] = []
    events: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    model_usage: list[dict[str, Any]] = []

    os.environ["RUNE_HOME"] = str(options.rune_home)
    options.rune_home.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(options.cwd)

        from rune.agent.loop import NativeAgentLoop
        from rune.types import AgentConfig

        config = AgentConfig()
        if options.model:
            config.model = options.model
        if options.provider:
            config.provider = options.provider
        if options.model or options.provider:
            config._overridden = True

        loop = NativeAgentLoop(config=config)

        async def _approval(capability: str, reason: str) -> bool:
            events.append(
                {
                    "event": "approval",
                    "capability": capability,
                    "reason": reason,
                    "decision": "approve",
                }
            )
            return True

        async def _ask_user(params: Any) -> Any:
            events.append({"event": "ask_user", "params": _jsonable(params), "answer": ""})
            try:
                from rune.capabilities.ask_user import UserResponse

                return UserResponse(selected_index=-1, answer="", raw_input="", free_text=True)
            except Exception:
                return ""

        async def _on_step(step: int) -> None:
            events.append({"event": "step", "step": step})

        async def _on_text(delta: str) -> None:
            output_parts.append(delta)

        async def _on_tool_call(info: dict[str, Any]) -> None:
            row = {"event": "tool_call", **_jsonable(info)}
            events.append(row)
            tool_calls.append(row)

        async def _on_tool_result(info: dict[str, Any]) -> None:
            events.append({"event": "tool_result", **_jsonable(info)})

        async def _on_step_tokens(
            step: int,
            step_tokens: int,
            total_tokens: int,
            token_budget: int,
        ) -> None:
            model_usage.append(
                {
                    "event": "step_tokens",
                    "step": step,
                    "step_tokens": step_tokens,
                    "total_tokens": total_tokens,
                    "token_budget": token_budget,
                    "input_tokens": None,
                    "cached_input_tokens": None,
                    "cache_write_tokens": None,
                    "reasoning_tokens": None,
                    "output_tokens": None,
                }
            )

        loop.set_approval_callback(_approval)
        loop.set_ask_user_callback(_ask_user)
        loop.on("step", _on_step)
        loop.on("text_delta", _on_text)
        loop.on("tool_call", _on_tool_call)
        loop.on("tool_result", _on_tool_result)
        loop.on("step_tokens", _on_step_tokens)

        run_context: dict[str, Any] = {"workspace_root": str(options.cwd)}
        use_git_diff = _is_git_repo(options.cwd)
        pre_snapshot = {} if use_git_diff else _snapshot_files(options.cwd)
        if options.memory_mode != "off":
            try:
                from rune.memory.manager import get_memory_manager

                memory_context = await get_memory_manager().build_memory_context(options.instruction)
                if memory_context:
                    run_context["memory_context"] = memory_context
            except Exception as exc:
                events.append({"event": "memory_context_error", "error": str(exc)})

        agent_instruction = build_agent_instruction(options)
        trace = await loop.run(agent_instruction, context=run_context)
        agent_wall_time_ms = int((time.monotonic() - start) * 1000)
        final_answer = "".join(output_parts)
        if options.memory_mode != "off" and final_answer.strip():
            try:
                from rune.agent.agent_context import (
                    AgentContext,
                    PostProcessInput,
                    post_process_agent_result,
                )

                await post_process_agent_result(
                    PostProcessInput(
                        context=AgentContext(
                            goal=agent_instruction,
                            original_goal=options.instruction,
                            channel="bench",
                            workspace_root=str(options.cwd),
                            conversation_id=(
                                f"bench:{options.benchmark}:{options.task_id}:"
                                f"{options.attempt_index}"
                            ),
                        ),
                        success=trace.reason in {"completed", "verified"},
                        answer=final_answer,
                        duration_ms=agent_wall_time_ms,
                    )
                )
                events.append({"event": "memory_post_process", "success": True})
            except Exception as exc:
                events.append({"event": "memory_post_process", "success": False, "error": str(exc)})

        _write_json(attempt_dir / "completion_trace.json", asdict(trace))
        _write_json(attempt_dir / "timing.json", {"agent_wall_time_ms": agent_wall_time_ms})
        _write_jsonl(attempt_dir / "events.jsonl", events)
        _write_jsonl(attempt_dir / "tool_calls.jsonl", tool_calls)
        _write_jsonl(attempt_dir / "model_usage.jsonl", model_usage)
        (attempt_dir / "final_answer.txt").write_text(final_answer, encoding="utf-8")

        diff = _git_diff(options.cwd) if use_git_diff else _filesystem_diff(options.cwd, pre_snapshot)
        git_status = (
            _git_status(options.cwd)
            if use_git_diff
            else _filesystem_status(options.cwd, pre_snapshot)
        )
        (attempt_dir / "patch.diff").write_text(diff, encoding="utf-8")
        (attempt_dir / "git_status.txt").write_text(git_status, encoding="utf-8")
        _write_json(
            attempt_dir / "audit.json",
            {
                "source_diff_present": bool(diff.strip()),
                "git_status_present": bool(git_status.strip()),
            },
        )

    finally:
        os.chdir(old_cwd)
        if old_rune_home is None:
            os.environ.pop("RUNE_HOME", None)
        else:
            os.environ["RUNE_HOME"] = old_rune_home
