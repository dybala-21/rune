"""Prevent recursive/nested agent invocations via env var sentinel.

Ported from src/utils/agent-child-guard.ts - uses the
``RUNE_AGENT_CHILD_GUARD_DEPTH`` environment variable to detect
and prevent runaway nested agent launches.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from typing import TypeVar

CHILD_GUARD_ENV_KEY = "RUNE_AGENT_CHILD_GUARD_DEPTH"

T = TypeVar("T")


def _parse_depth(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        val = int(raw)
        return val if val >= 0 else 0
    except (ValueError, TypeError):
        return 0


def get_agent_child_guard_depth(env: dict[str, str] | None = None) -> int:
    """Return the current guard depth (0 means top-level)."""
    store = env if env is not None else os.environ
    return _parse_depth(store.get(CHILD_GUARD_ENV_KEY))


def is_agent_child_guard_active(env: dict[str, str] | None = None) -> bool:
    """Return *True* when running inside a nested agent invocation."""
    return get_agent_child_guard_depth(env) > 0


def push_agent_child_guard(env: dict[str, str] | None = None) -> int:
    """Increment the guard depth and return the new value."""
    store = env if env is not None else os.environ
    next_depth = get_agent_child_guard_depth(store) + 1
    store[CHILD_GUARD_ENV_KEY] = str(next_depth)
    return next_depth


def pop_agent_child_guard(env: dict[str, str] | None = None) -> int:
    """Decrement the guard depth and return the new value."""
    store = env if env is not None else os.environ
    current = get_agent_child_guard_depth(store)
    if current <= 1:
        store.pop(CHILD_GUARD_ENV_KEY, None)
        return 0
    next_depth = current - 1
    store[CHILD_GUARD_ENV_KEY] = str(next_depth)
    return next_depth


@contextmanager
def agent_child_guard(
    env: dict[str, str] | None = None,
) -> Generator[int]:
    """Context manager that pushes the guard on entry and pops on exit.

    Usage::

        with agent_child_guard() as depth:
            print(f"Running at depth {depth}")
            run_sub_agent()
    """
    depth = push_agent_child_guard(env)
    try:
        yield depth
    finally:
        pop_agent_child_guard(env)


async def with_agent_child_guard_async[T](
    fn: Callable[[], Awaitable[T]],
    env: dict[str, str] | None = None,
) -> T:
    """Async wrapper that pushes guard, awaits *fn*, then pops guard."""
    push_agent_child_guard(env)
    try:
        return await fn()
    finally:
        pop_agent_child_guard(env)
