"""Harbor installed-agent adapter for RUNE.

This module is intentionally importable without Harbor installed so unit tests
can validate command construction in the normal RUNE dev environment. Harbor
will provide the real base classes at benchmark runtime.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping
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
CONTROL_ENV_KEYS = (
    "UV_CACHE_DIR",
    "UV_LINK_MODE",
    "UV_PYTHON",
    "RUNE_MODEL",
    "RUNE_PROVIDER",
    "RUNE_LOG_LEVEL",
    "RUNE_HARBOR_INSTALL_MODE",
    "RUNE_HARBOR_WHEELHOUSE",
    "RUNE_HARBOR_BENCHMARK",
    "RUNE_HARBOR_TASK_ID",
    "RUNE_HARBOR_MODEL",
    "RUNE_HARBOR_PROVIDER",
    "RUNE_HARBOR_MEMORY_MODE",
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
    "RUNE_HARBOR_MAX_STEPS",
    "RUNE_HARBOR_TIMEOUT_SECONDS",
    # Optional per-turn write/execute cap (off by default in the runner). Must
    # be forwarded so an operator can enable it from `harbor run --agent-env`
    # for a canary without editing the runner defaults.
    "RUNE_BENCH_MAX_WRITE_EXEC_PER_TURN",
    # Optional Evidence Gate (output-correctness re-verification before finalize;
    # off by default). Forwarded so a canary can toggle it via --agent-env.
    "RUNE_BENCH_EVIDENCE_GATE",
)
PROVIDER_ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    "bedrock": (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
    ),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
    ),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "vertex": (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "GOOGLE_API_KEY",
    ),
}
_KNOWN_CREDENTIAL_ENV_KEYS = tuple(
    sorted({key for keys in PROVIDER_ENV_KEYS.values() for key in keys})
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _raw_context_value(context: Any, name: str) -> Any:
    if isinstance(context, dict):
        return context.get(name)
    return getattr(context, name, None)


def _env_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(key, str) and item is not None
    }


def _agent_env_from_context(context: Any) -> dict[str, str]:
    """Return Harbor agent env entries when available on the runtime context."""
    config = _raw_context_value(context, "config")
    agent = _raw_context_value(config, "agent")
    agent_env = _env_mapping(_raw_context_value(agent, "env"))
    if agent_env:
        return agent_env

    agents = _raw_context_value(config, "agents")
    if isinstance(agents, list) and agents:
        first_agent = agents[0]
        return _env_mapping(_raw_context_value(first_agent, "env"))
    return {}


def _env_value(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key) or os.environ.get(key)
    return value if value else None


def _env_int_value(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    raw = (env or {}).get(name) or os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _env_int_optional(name: str, env: Mapping[str, str] | None = None) -> int | None:
    raw = (env or {}).get(name) or os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_key(env: Mapping[str, str] | None = None) -> str | None:
    merged = env or {}
    raw = (
        merged.get("RUNE_HARBOR_PROVIDER")
        or os.environ.get("RUNE_HARBOR_PROVIDER")
        or merged.get("RUNE_PROVIDER")
        or os.environ.get("RUNE_PROVIDER")
    )
    if not raw:
        return None
    provider = raw.strip().lower().replace("_", "-")
    aliases = {
        "anthropic-bedrock": "bedrock",
        "aws-bedrock": "bedrock",
        "google-vertex": "vertex",
        "vertex-ai": "vertex",
    }
    return aliases.get(provider, provider)


def _credential_env_keys(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    provider = _provider_key(env)
    if provider in PROVIDER_ENV_KEYS:
        return PROVIDER_ENV_KEYS[provider]

    merged = env or {}
    present = [key for key in _KNOWN_CREDENTIAL_ENV_KEYS if merged.get(key) or os.environ.get(key)]
    if len(present) == 1:
        return (present[0],)
    return ()


def _explicit_env_keys(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw = (env or {}).get("RUNE_HARBOR_PASS_ENV") or os.environ.get("RUNE_HARBOR_PASS_ENV", "")
    keys = raw.replace(",", " ").split()
    return tuple(key for key in keys if _ENV_NAME_RE.fullmatch(key))


def _container_env(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = {**os.environ, **dict(extra_env or {})}
    keys = (*CONTROL_ENV_KEYS, *_credential_env_keys(merged), *_explicit_env_keys(merged))
    return {key: value for key in keys if (value := merged.get(key))}


def harbor_task_id(context: Any) -> str:
    """Extract a stable task ID across Harbor context versions."""
    return (
        _context_value(context, "task_id", "task_name", "id", "name")
        or _context_value(getattr(context, "config", None), "task_id", "task_name", "task")
        or _context_value(getattr(context, "task", None), "id", "name", "path")
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
    max_steps: int | None = None,
    timeout_seconds: int | None = None,
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
    if max_steps:
        command_suffix.extend(["--max-steps", str(max_steps)])
    if timeout_seconds:
        command_suffix.extend(["--timeout-seconds", str(timeout_seconds)])

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
        agent_env = _agent_env_from_context(context)
        benchmark = _env_value(agent_env, "RUNE_HARBOR_BENCHMARK") or "terminal-bench-v2"
        task_id = _env_value(agent_env, "RUNE_HARBOR_TASK_ID") or harbor_task_id(context)
        model = _env_value(agent_env, "RUNE_HARBOR_MODEL") or _env_value(agent_env, "RUNE_MODEL")
        provider = _env_value(agent_env, "RUNE_HARBOR_PROVIDER") or _env_value(
            agent_env,
            "RUNE_PROVIDER",
        )
        memory_mode = _env_value(agent_env, "RUNE_HARBOR_MEMORY_MODE") or "default"
        agent_variant_id = _env_value(
            agent_env,
            "RUNE_HARBOR_AGENT_VARIANT_ID",
        ) or _env_value(agent_env, "RUNE_BENCH_AGENT_VARIANT_ID")
        attempt_index = _env_int_value("RUNE_HARBOR_ATTEMPT_INDEX", 1, agent_env)
        max_steps = _env_int_optional("RUNE_HARBOR_MAX_STEPS", agent_env)
        timeout_seconds = _env_int_optional("RUNE_HARBOR_TIMEOUT_SECONDS", agent_env)
        command = build_rune_bench_command(
            instruction=instruction,
            benchmark=benchmark,
            task_id=task_id,
            model=model,
            provider=provider,
            memory_mode=memory_mode,
            agent_variant_id=agent_variant_id,
            attempt_index=attempt_index,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
        await self.exec_as_agent(environment, command=command, env=_container_env(agent_env))

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
