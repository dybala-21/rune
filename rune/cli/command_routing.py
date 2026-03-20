"""Command Routing for RUNE CLI.

Ported from src/cli/command-routing.ts -- routes slash commands and
special inputs, determines whether arguments should be promoted to
the ``exec`` subcommand.
"""

from __future__ import annotations

BUILTIN_SUBCOMMANDS = frozenset({"exec", "env", "token", "daemon", "web", "send"})


def _is_flag(arg: str) -> bool:
    """Return True if *arg* looks like a CLI flag (starts with ``-``)."""
    return arg.startswith("-")


def join_non_flag_args(args: list[str]) -> str:
    """Join all non-flag arguments into a single space-separated string."""
    return " ".join(arg for arg in args if not _is_flag(arg)).strip()


def should_promote_to_exec(args: list[str]) -> bool:
    """Determine whether *args* should be wrapped with an ``exec`` prefix.

    Returns ``True`` when the first non-flag argument is not a known
    built-in subcommand (and is not already ``exec``).
    """
    if not args:
        return False
    if args[0] == "exec":
        return False

    first_non_flag = next((a for a in args if not _is_flag(a)), None)
    if first_non_flag is None:
        return False

    return first_non_flag not in BUILTIN_SUBCOMMANDS


def promote_to_exec_args(args: list[str]) -> list[str]:
    """Return *args* with ``exec`` prepended when promotion is needed."""
    if not should_promote_to_exec(args):
        return list(args)
    return ["exec", *args]


def resolve_non_interactive_command(args: list[str]) -> str | None:
    """Extract a non-interactive command string from *args*.

    Returns ``None`` when no meaningful command text is present.
    """
    command = join_non_flag_args(args)
    return command if command else None
