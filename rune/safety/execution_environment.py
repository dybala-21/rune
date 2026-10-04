"""Keep a run's commands and checks on the same configured backend."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from rune.config.schema import SandboxConfig

_environment: ContextVar[tuple[SandboxConfig, str] | None] = ContextVar("execution_environment", default=None)


def execution_config() -> SandboxConfig:
    from rune.cloud.boundary import hosted, sandbox
    from rune.config import get_config

    if hosted():
        return sandbox()
    current = _environment.get()
    return (current[0] if current else get_config().safety.sandbox).model_copy(deep=True)


def execution_workspace(default: str | None = None) -> str:
    current = _environment.get()
    return current[1] if current else default or os.getcwd()


@contextmanager
def environment_scope(workspace: str, config: SandboxConfig | None = None):
    from rune.cloud.boundary import check_path, hosted, sandbox

    if reason := check_path(workspace):
        raise ValueError(reason)
    if hosted():
        config = sandbox()
    token = _environment.set(((config or execution_config()).model_copy(deep=True), str(Path(workspace).resolve())))
    try:
        yield
    finally:
        _environment.reset(token)


def with_execution_environment(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        workspace = (kwargs.get("context") or {}).get("workspace_root") or os.getcwd()
        with environment_scope(workspace):
            return await function(*args, **kwargs)
    return wrapped
