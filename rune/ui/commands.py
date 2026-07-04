"""Backward-compat shim — the slash-command registry moved to
rune.slash_commands so the web/API server does not depend on the TUI package.
"""

from rune.slash_commands import (
    _ALIAS_MAP,
    COMMANDS,
    Command,
    CommandHandler,
    parse_slash_command,
    suggest_command,
)

__all__ = [
    "_ALIAS_MAP",
    "COMMANDS",
    "Command",
    "CommandHandler",
    "parse_slash_command",
    "suggest_command",
]
