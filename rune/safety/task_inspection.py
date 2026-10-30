"""Keep saved-task inspections from changing user work or external state."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_active: ContextVar[bool] = ContextVar("task_inspection", default=False)

# Audited built-ins only; groups and third-party read-only hints are not authority.
_READ_TOOLS = frozenset({
    "think", "ask_user", "task_blocked", "memory_search", "task_list",
    "file_read", "file_list", "file_search", "file_query", "document_read", "document_preview",
    "document_bundle_inspect", "code_analyze", "code_find_def", "code_find_refs", "code_impact",
    "project_map", "skill_search", "skill_load", "connector_list", "service_list", "service_status",
    "managed_service_status", "browser_observe", "browser_extract", "browser_find", "browser_screenshot",
    "desktop_apps", "desktop_observe", "desktop_wait", "web_search", "web_fetch",
})


def inspecting_task() -> bool:
    return _active.get()


def restrict_to_inspection() -> None:
    _active.set(True)


@contextmanager
def inspection_scope() -> Iterator[None]:
    # Nested runs inherit the restriction and cannot clear it for their parent.
    token = _active.set(_active.get())
    try:
        yield
    finally:
        _active.reset(token)


def inspection_blocker(name: str, params: dict[str, Any]) -> str | None:
    if not _active.get():
        return None
    allowed = name in _READ_TOOLS
    if name == "web_fetch":
        method = params.get("method", "GET")
        allowed = isinstance(method, str) and method.upper() in {"GET", "HEAD"} and not params.get("body")
    elif name == "browser_screenshot":
        allowed = not params.get("path")
    if allowed:
        return None
    return (f"{name} was not executed: this turn only inspects saved work. "
            "Use the available read-only tools; changes require a separate task continuation.")
