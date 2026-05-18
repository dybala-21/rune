from __future__ import annotations

from benchmarks.harbor.rune_agent import (
    _container_env,
    _env_flag,
    build_rune_bench_command,
    harbor_task_id,
)


class _Task:
    id = "task-from-nested"


class _Context:
    task = _Task()


class _TaskId:
    path = "task-from-path"


class _PathContext:
    task_id = _TaskId()


def test_harbor_task_id_reads_nested_task_id():
    assert harbor_task_id(_Context()) == "task-from-nested"


def test_harbor_task_id_reads_task_path():
    assert harbor_task_id(_PathContext()) == "task-from-path"


def test_build_rune_bench_command_quotes_instruction():
    command = build_rune_bench_command(
        instruction="write 'hello world'",
        benchmark="terminal-bench-v2",
        task_id="adaptive-rejection-sampler",
        model="gpt-5.4",
        provider="openai",
    )

    assert "export RUNE_HOME=/logs/agent/rune_home" in command
    assert 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' in command
    assert "--benchmark terminal-bench-v2" in command
    assert "--task-id adaptive-rejection-sampler" in command
    assert "--attempt-index 1" in command
    assert "--model gpt-5.4" in command
    assert "--provider openai" in command
    assert "'write '\"'\"'hello world'\"'\"''" in command


def test_container_env_passes_llm_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("UV_CACHE_DIR", "/uv-cache")
    monkeypatch.setenv("UNRELATED_KEY", "ignored")

    env = _container_env()

    assert env["OPENAI_API_KEY"] == "secret"
    assert env["UV_CACHE_DIR"] == "/uv-cache"
    assert "UNRELATED_KEY" not in env


def test_env_flag_accepts_common_truthy_values(monkeypatch):
    monkeypatch.setenv("RUNE_HARBOR_SKIP_INSTALL", "yes")

    assert _env_flag("RUNE_HARBOR_SKIP_INSTALL")
