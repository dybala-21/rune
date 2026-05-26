from __future__ import annotations

from pathlib import Path

from benchmarks.harbor.rune_agent import (
    DEFAULT_INSTALL_COMMAND,
    _container_env,
    _env_flag,
    build_rune_bench_command,
    harbor_task_id,
)

_CREDENTIAL_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
)


def _clear_credential_env(monkeypatch):
    for key in _CREDENTIAL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class _Task:
    id = "task-from-nested"


class _Context:
    task = _Task()


class _TaskId:
    path = "/terminal-bench/tasks/task-from-path"


class _PathContext:
    task_id = _TaskId()


class _MixedTaskId:
    id = 123
    path = "/terminal-bench/tasks/task-from-fallback-path"


class _MixedPathContext:
    task_id = _MixedTaskId()


def test_harbor_task_id_reads_nested_task_id():
    assert harbor_task_id(_Context()) == "task-from-nested"


def test_harbor_task_id_reads_task_path():
    assert harbor_task_id(_PathContext()) == "task-from-path"


def test_harbor_task_id_skips_non_string_nested_fields():
    assert harbor_task_id(_MixedPathContext()) == "task-from-fallback-path"


def test_build_rune_bench_command_quotes_instruction():
    command = build_rune_bench_command(
        instruction="write 'hello world'",
        benchmark="terminal-bench-v2",
        task_id="adaptive-rejection-sampler",
        model="gpt-5.4",
        provider="openai",
    )

    assert "export RUNE_HOME=/logs/agent/rune_home" in command
    assert 'export PATH="/logs/agent/rune_venv/bin:/uv-cache/bin:' in command
    assert "RUNE_BENCH_INSTALL_FINGERPRINT=/logs/agent/rune_install_fingerprint.json" in command
    assert "--benchmark terminal-bench-v2" in command
    assert 'export RUNE_BENCH_TASK_ID=${RUNE_HARBOR_TASK_ID:-adaptive-rejection-sampler}' in command
    assert '--task-id "$RUNE_BENCH_TASK_ID"' in command
    assert "--attempt-index 1" in command
    assert "--model gpt-5.4" in command
    assert "--provider openai" in command
    assert "'write '\"'\"'hello world'\"'\"''" in command


def test_container_env_passes_llm_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("RUNE_HARBOR_PROVIDER", "openai")
    monkeypatch.setenv("UV_CACHE_DIR", "/uv-cache")
    monkeypatch.setenv("RUNE_BENCH_REQUIRE_FINGERPRINT", "1")
    monkeypatch.setenv("RUNE_BENCH_BLOCK_VCS_HISTORY", "1")
    monkeypatch.setenv("RUNE_BENCH_EXPECT_SOURCE_DIFF_SHA256", "abc123")
    monkeypatch.setenv("UNRELATED_KEY", "ignored")

    env = _container_env()

    assert env["OPENAI_API_KEY"] == "secret"
    assert env["UV_CACHE_DIR"] == "/uv-cache"
    assert env["RUNE_BENCH_REQUIRE_FINGERPRINT"] == "1"
    assert env["RUNE_BENCH_BLOCK_VCS_HISTORY"] == "1"
    assert env["RUNE_BENCH_EXPECT_SOURCE_DIFF_SHA256"] == "abc123"
    assert "UNRELATED_KEY" not in env


def test_container_env_does_not_pass_unselected_provider_credentials(monkeypatch):
    monkeypatch.setenv("RUNE_HARBOR_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    env = _container_env()

    assert env["OPENAI_API_KEY"] == "openai-secret"
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_container_env_allows_explicit_extra_env(monkeypatch):
    monkeypatch.setenv("RUNE_HARBOR_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("RUNE_HARBOR_PASS_ENV", "AWS_REGION,INVALID-NAME")

    env = _container_env()

    assert env["OPENAI_API_KEY"] == "openai-secret"
    assert env["AWS_REGION"] == "us-east-1"
    assert "INVALID-NAME" not in env


def test_container_env_passes_single_credential_when_provider_unset(monkeypatch):
    _clear_credential_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    env = _container_env()

    assert env["GEMINI_API_KEY"] == "gemini-secret"


def test_env_flag_accepts_common_truthy_values(monkeypatch):
    monkeypatch.setenv("RUNE_HARBOR_SKIP_INSTALL", "yes")

    assert _env_flag("RUNE_HARBOR_SKIP_INSTALL")


def test_default_install_command_prefers_mounted_repo():
    assert "/rune-src/benchmarks/harbor/install_rune.sh" in DEFAULT_INSTALL_COMMAND
    assert "python3 -m pip install --user rune-ai" in DEFAULT_INSTALL_COMMAND


def test_install_script_uses_attempt_local_venv_and_fingerprint():
    script = Path("benchmarks/harbor/install_rune.sh").read_text(encoding="utf-8")

    assert "RUNE_VENV" in script
    assert "RUNE_INSTALL_FINGERPRINT" in script
    assert "--no-index" in script
    assert "--find-links" in script
    assert "--refresh-package rune-ai" in script
    assert "wheelhouse_sha256" in script


def test_build_rune_bench_command_passes_agent_variant():
    command = build_rune_bench_command(
        instruction="write hello",
        benchmark="terminal-bench-v2",
        task_id="regex-log",
        agent_variant_id="rune-aa-terminal-v1",
    )

    assert "--agent-variant-id rune-aa-terminal-v1" in command
