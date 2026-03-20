"""Child Invocation Guard for RUNE CLI.

Ported from src/cli/child-invocation-guard.ts and
src/utils/agent-child-guard.ts -- prevents recursive agent invocations
when the RUNE agent spawns sub-processes that might inadvertently start
another agent loop.

The guard is activated by setting an environment variable
(``RUNE_AGENT_CHILD_GUARD``) to a truthy value.  When active, attempts
to run ``exec`` or any non-builtin subcommand will be blocked.

The ``send`` subcommand is explicitly allowed even when the guard is
active, so that scheduled tasks (e.g., briefing delivery) can still
operate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

_GUARD_ENV_KEY = "RUNE_AGENT_CHILD_GUARD"

# Built-in subcommands that do NOT start an agent loop.
# ``send`` is intentionally included so scheduled tasks can deliver
# messages without being blocked.
_BUILTIN_SUBCOMMANDS = frozenset({"exec", "env", "token", "daemon", "web", "send"})


def is_agent_child_guard_active(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return ``True`` if the child-guard environment variable is set."""
    if env is None:
        env = os.environ
    value = env.get(_GUARD_ENV_KEY, "")
    return value.lower() in ("1", "true", "yes")


def activate_child_guard() -> None:
    """Set the child-guard env var so spawned sub-processes are guarded."""
    os.environ[_GUARD_ENV_KEY] = "1"


def deactivate_child_guard() -> None:
    """Remove the child-guard env var."""
    os.environ.pop(_GUARD_ENV_KEY, None)


def _is_flag(arg: str) -> bool:
    return arg.startswith("-")


def should_block_child_agent_invocation(
    args: list[str],
    env: Mapping[str, str] | None = None,
) -> bool:
    """Determine whether the given CLI *args* should be blocked.

    Returns ``True`` when the child guard is active **and** the invocation
    would start an agent loop (i.e., it targets ``exec`` or a non-builtin
    subcommand).
    """
    if not is_agent_child_guard_active(env):
        return False

    first_non_flag = next((a for a in args if not _is_flag(a)), None)
    if first_non_flag is None:
        return False

    # ``exec`` always starts an agent -- block it.
    if first_non_flag == "exec":
        return True

    # Known builtin subcommands that are safe -- allow them.
    if first_non_flag in _BUILTIN_SUBCOMMANDS:
        return False

    # Unknown subcommand would be promoted to ``exec`` -- block it.
    return True
