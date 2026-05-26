"""Harbor installed-agent adapter for RUNE.

This module is intentionally importable without Harbor installed so unit tests
can validate command construction in the normal RUNE dev environment. Harbor
will provide the real base classes at benchmark runtime.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by Harbor, not unit tests.
    from harbor.agents.installed.base import (  # type: ignore[import-not-found]
        BaseInstalledAgent,
        with_prompt_template,
    )
    from harbor.environments.base import BaseEnvironment  # type: ignore[import-not-found]
    from harbor.models.agent.context import AgentContext  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - lets local tests import this file.
    BaseEnvironment = Any
    AgentContext = Any

    class BaseInstalledAgent:  # type: ignore[no-redef]
        pass

    def with_prompt_template(fn: Any) -> Any:
        return fn


DEFAULT_INSTALL_COMMAND = (
    "if [ -x /rune-src/benchmarks/harbor/install_rune.sh ]; then "
    "/rune-src/benchmarks/harbor/install_rune.sh; "
    "else python3 -m pip install --user rune-ai; fi"
)
DEFAULT_RUN_ROOT = Path("/logs/agent/rune")
DEFAULT_RUNE_HOME = Path("/logs/agent/rune_home")
DEFAULT_RUNE_VENV = Path("/logs/agent/rune_venv")
DEFAULT_INSTALL_FINGERPRINT = Path("/logs/agent/rune_install_fingerprint.json")
PASSTHROUGH_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "UV_CACHE_DIR",
    "UV_LINK_MODE",
    "UV_PYTHON",
    "RUNE_MODEL",
    "RUNE_PROVIDER",
    "RUNE_LOG_LEVEL",
    "RUNE_HARBOR_INSTALL_MODE",
    "RUNE_HARBOR_WHEELHOUSE",
    "RUNE_HARBOR_AGENT_VARIANT_ID",
    "RUNE_BENCH_AGENT_VARIANT_ID",
    "RUNE_BENCH_EXPECT_AGENT_VARIANT_ID",
    "RUNE_BENCH_EXPECT_INSTALL_MODE",
    "RUNE_BENCH_EXPECT_PROMPT_POLICY",
    "RUNE_BENCH_EXPECT_SOURCE_GIT_SHA",
    "RUNE_BENCH_EXPECT_SOURCE_DIFF_SHA256",
    "RUNE_BENCH_EXPECT_WHEELHOUSE_SHA256",
    "RUNE_BENCH_BLOCK_VCS_HISTORY",
    "RUNE_BENCH_ALLOW_VCS_HISTORY",
    "RUNE_BENCH_REQUIRE_FINGERPRINT",
    "RUNE_BENCH_TASK_ID",
    "RUNE_BENCH_SOURCE_GIT_SHA",
    "RUNE_BENCH_SOURCE_DIFF_SHA256",
    "RUNE_BENCH_WHEELHOUSE_SHA256",
)


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _double_quoted_path_part(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


def _normalise_context_value(name: str, value: str) -> str:
    if name == "path":
        return Path(value).name
    return value


def _nested_context_value(value: Any) -> tuple[str, str] | None:
    for key in ("id", "name", "path"):
        if isinstance(value, dict):
            nested = value.get(key)
        else:
            nested = getattr(value, key, None)
        if isinstance(nested, str) and nested:
            return key, nested
    return None


def _context_value(context: Any, *names: str) -> str | None:
    for name in names:
        value = context.get(name) if isinstance(context, dict) else getattr(context, name, None)
        if isinstance(value, str) and value:
            return _normalise_context_value(name, value)
        if value is not None:
            nested_pair = _nested_context_value(value)
            if nested_pair is not None:
                nested_name, nested = nested_pair
                return _normalise_context_value(nested_name, nested)
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _container_env() -> dict[str, str]:
    return {key: value for key in PASSTHROUGH_ENV_KEYS if (value := os.environ.get(key))}


def harbor_task_id(context: Any) -> str:
    """Extract a stable task ID across Harbor context versions."""
    return (
        _context_value(context, "task_id", "task_name", "id", "name")
        or _context_value(getattr(context, "config", None), "task_id", "task_name", "task")
        or _context_value(getattr(context, "task", None), "id", "name")
        or "harbor-task"
    )


def build_rune_bench_command(
    *,
    instruction: str,
    benchmark: str,
    task_id: str,
    run_root: Path = DEFAULT_RUN_ROOT,
    rune_home: Path = DEFAULT_RUNE_HOME,
    rune_venv: Path = DEFAULT_RUNE_VENV,
    install_fingerprint: Path = DEFAULT_INSTALL_FINGERPRINT,
    cwd: Path = Path("/app"),
    model: str | None = None,
    provider: str | None = None,
    memory_mode: str = "default",
    agent_variant_id: str | None = None,
    attempt_index: int = 1,
) -> str:
    """Build the command Harbor should execute inside the task container."""
    command_prefix = [
        "rune",
        "bench",
        "run",
        "--benchmark",
        benchmark,
        "--task-id",
    ]
    command_suffix = [
        "--instruction",
        instruction,
        "--output-dir",
        str(run_root),
        "--attempt-index",
        str(attempt_index),
        "--rune-home",
        str(rune_home),
        "--cwd",
        str(cwd),
        "--memory-mode",
        memory_mode,
    ]
    if agent_variant_id:
        command_suffix.extend(["--agent-variant-id", agent_variant_id])
    if model:
        command_suffix.extend(["--model", model])
    if provider:
        command_suffix.extend(["--provider", provider])

    rune_venv_bin = _double_quoted_path_part(str(rune_venv / "bin"))
    return (
        f"mkdir -p {_shell_join([str(run_root), str(rune_home)])} && "
        f'export PATH="{rune_venv_bin}:/uv-cache/bin:'
        '$HOME/.local/bin:$HOME/.cargo/bin:$PATH" && '
        f"export RUNE_HOME={shlex.quote(str(rune_home))} && "
        f"export RUNE_BENCH_INSTALL_FINGERPRINT={shlex.quote(str(install_fingerprint))} && "
        f"export RUNE_BENCH_TASK_ID=${{RUNE_HARBOR_TASK_ID:-{shlex.quote(task_id)}}} && "
        f"cd {shlex.quote(str(cwd))} && "
        f'{_shell_join(command_prefix)} "$RUNE_BENCH_TASK_ID" {_shell_join(command_suffix)}'
    )


class RuneInstalledAgent(BaseInstalledAgent):  # type: ignore[misc]
    """Run RUNE as a Harbor installed agent."""

    @staticmethod
    def name() -> str:
        return "rune"

    def version(self) -> str | None:
        try:
            from rune import __version__

            return __version__
        except Exception:
            return None

    async def install(self, environment: BaseEnvironment) -> None:
        if _env_flag("RUNE_HARBOR_SKIP_INSTALL"):
            return
        install_command = os.environ.get("RUNE_HARBOR_INSTALL_CMD", DEFAULT_INSTALL_COMMAND)
        await self.exec_as_agent(environment, command=install_command, env=_container_env())

    @with_prompt_template  # type: ignore[untyped-decorator]
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        benchmark = os.environ.get("RUNE_HARBOR_BENCHMARK", "terminal-bench-v2")
        task_id = os.environ.get("RUNE_HARBOR_TASK_ID") or harbor_task_id(context)
        model = os.environ.get("RUNE_HARBOR_MODEL") or os.environ.get("RUNE_MODEL")
        provider = os.environ.get("RUNE_HARBOR_PROVIDER") or os.environ.get("RUNE_PROVIDER")
        memory_mode = os.environ.get("RUNE_HARBOR_MEMORY_MODE", "default")
        agent_variant_id = os.environ.get("RUNE_HARBOR_AGENT_VARIANT_ID") or os.environ.get(
            "RUNE_BENCH_AGENT_VARIANT_ID"
        )
        attempt_index = _env_int("RUNE_HARBOR_ATTEMPT_INDEX", 1)
        command = build_rune_bench_command(
            instruction=instruction,
            benchmark=benchmark,
            task_id=task_id,
            model=model,
            provider=provider,
            memory_mode=memory_mode,
            agent_variant_id=agent_variant_id,
            attempt_index=attempt_index,
        )
        await self.exec_as_agent(environment, command=command, env=_container_env())

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Expose the latest RUNE artifacts to Harbor context consumers when present."""
        artifact_root = DEFAULT_RUN_ROOT
        try:
            latest = max(artifact_root.glob("*/*/attempt-*"), key=lambda path: path.stat().st_mtime)
        except Exception:
            return

        final_answer = latest / "final_answer.txt"
        completion_trace = latest / "completion_trace.json"
        if final_answer.exists():
            with final_answer.open(encoding="utf-8") as f:
                context.output = f.read()
        if completion_trace.exists():
            with completion_trace.open(encoding="utf-8") as f:
                context.rune_completion_trace = json.load(f)
