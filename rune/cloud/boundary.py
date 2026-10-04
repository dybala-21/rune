"""Deployment constraints that user/project settings cannot relax."""

import os
from pathlib import Path

from rune.config.schema import SandboxConfig

_HOSTED = os.environ.get("RUNE_CLOUD_WORKER") == "1"
_WORKSPACE = os.environ.get("RUNE_ISOLATION_ROOT") if _HOSTED else None
_TOOLS = frozenset({
    "think", "ask_user", "task_blocked", "task_create", "task_update", "task_list",
    "file_read", "file_write", "file_edit", "file_delete", "file_list", "file_search",
    "bash_execute", "web_search", "web_fetch", "connector_list", "connector_request",
    "browser_navigate", "browser_open", "browser_observe", "browser_act", "browser_batch", "browser_extract",
    "browser_find", "browser_screenshot", "browser_discover_apis", "browser_workflow",
    "memory_search", "memory_save", "cron_create", "cron_list", "cron_update", "cron_delete",
})


def hosted() -> bool:
    return _HOSTED or os.environ.get("RUNE_CLOUD_WORKER") == "1"


def workspace_root() -> str | None:
    return _WORKSPACE if _HOSTED else os.environ.get("RUNE_ISOLATION_ROOT")


def sandbox() -> SandboxConfig:
    return SandboxConfig(backend="container", image="python:3.13-slim", enabled=True, allow_network=False)


def check_path(path: str) -> str | None:
    if not hosted():
        return None
    root = workspace_root()
    if not root or not Path(path).expanduser().resolve().is_relative_to(Path(root).resolve()):
        return "Hosted file access is restricted to the owner's workspace"
    return None


def check_tool(name: str, params: dict) -> str | None:
    if not hosted():
        return None
    if name not in _TOOLS:
        return "This tool is not enabled in the hosted execution boundary"
    fields = ("path", "file_path", "directory", "cwd") if name.startswith("file_") or name == "bash_execute" else ("path",) if name == "browser_screenshot" else ()
    for field in fields:
        if params.get(field) and (reason := check_path(str(params[field]))):
            return reason
    return None
