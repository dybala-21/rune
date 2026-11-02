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


def browser_proxy() -> dict[str, str] | None:
    """Playwright proxy settings for a hosted worker whose egress is forced through a proxy.

    Chromium does not read HTTP(S)_PROXY reliably, so a hosted browser would otherwise try
    direct connections that the host firewall drops. Local runs are left untouched.
    """
    if not hosted():
        return None
    from urllib.parse import unquote, urlsplit

    raw = next((os.environ[name] for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
                if os.environ.get(name, "").strip()), "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw if "://" in raw else f"http://{raw}")
    if parsed.scheme not in {"http", "https", "socks5"} or not parsed.hostname:
        raise ValueError("The hosted egress proxy must be an http, https or socks5 URL")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    proxy: dict[str, str] = {"server": f"{parsed.scheme}://{host}" + (f":{parsed.port}" if parsed.port else "")}
    bypass = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").strip()
    if bypass:
        proxy["bypass"] = bypass
    if parsed.username:
        if parsed.scheme == "socks5":
            # Chromium cannot authenticate to a SOCKS5 proxy; fail here with the reason.
            raise ValueError("Chromium does not support SOCKS5 proxy credentials; use an http(s) proxy")
        proxy["username"] = unquote(parsed.username)
        proxy["password"] = unquote(parsed.password or "")
    return proxy
