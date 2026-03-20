"""Slash command definitions for RUNE TUI.

Provides the built-in slash commands (/help, /exit, etc.) and
a parser to detect them in user input.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

type CommandHandler = Callable[[str], Coroutine[Any, Any, str | None]]


@dataclass(slots=True)
class Command:
    """A registered slash command."""

    name: str
    description: str
    handler: CommandHandler
    aliases: list[str] = field(default_factory=list)
    usage: str = ""
    hidden: bool = False  # Hidden from /help but still callable


# Command handlers

# /help, /exit, /clear, /model are intercepted by RuneApp before reaching
# these handlers. The handlers here exist only for the registry/completer.

async def _noop_handler(args: str) -> str | None:
    """No-op - command is handled directly by RuneApp."""
    return None


async def _export_handler(args: str) -> str:
    """Export conversation."""
    fmt = args.strip().lower()
    if fmt and fmt not in ("markdown", "json", "html"):
        return f"Unknown format: {fmt}. Use: markdown, json, html"
    if not fmt:
        return "__ACTION__:interactive_export"
    return f"__ACTION__:export:{fmt}"


async def _session_handler(args: str) -> str:
    return "__ACTION__:show_session"


async def _status_handler(args: str) -> str:
    return "__ACTION__:show_status"


async def _config_handler(args: str) -> str:
    return "__ACTION__:show_config"


async def _undo_handler(args: str) -> str | None:
    return "__ACTION__:undo"


async def _retry_handler(args: str) -> str | None:
    return "__ACTION__:retry"


async def _style_handler(args: str) -> str | None:
    name = args.strip().lower()
    if name in ("compact", "normal", "verbose"):
        return f"__ACTION__:set_style:{name}"
    return "__ACTION__:cycle_style"


async def _compact_handler(args: str) -> str | None:
    return "__ACTION__:set_style:compact"


async def _normal_handler(args: str) -> str | None:
    return "__ACTION__:set_style:normal"


async def _verbose_handler(args: str) -> str | None:
    return "__ACTION__:set_style:verbose"


async def _files_handler(args: str) -> str | None:
    return "__ACTION__:toggle_files"


async def _diff_handler(args: str) -> str | None:
    return "__ACTION__:toggle_git_diff"


async def _copy_handler(args: str) -> str | None:
    return "__ACTION__:copy_response"


async def _theme_handler(args: str) -> str | None:
    name = args.strip().lower()
    if not name:
        return "__ACTION__:interactive_theme"
    return f"__ACTION__:set_theme:{name}"


async def _save_handler(args: str) -> str | None:
    name = args.strip() or None
    if name:
        return f"__ACTION__:save:{name}"
    return "__ACTION__:save"


async def _sessions_handler(args: str) -> str | None:
    return "__ACTION__:show_sessions"


async def _load_handler(args: str) -> str | None:
    session_id = args.strip()
    if not session_id:
        return "__ACTION__:interactive_load"
    return f"__ACTION__:load:{session_id}"


async def _search_handler(args: str) -> str | None:
    query = args.strip()
    if not query:
        return "Usage: /search <query>"
    return f"__ACTION__:search:{query}"


async def _cost_handler(args: str) -> str | None:
    return "__ACTION__:show_cost"


async def _stats_handler(args: str) -> str | None:
    return "__ACTION__:show_stats"


async def _memory_handler(args: str) -> str | None:
    sub = args.strip().lower()
    if not sub:
        return "__ACTION__:memory:show"
    if sub not in ("show", "add", "clear"):
        return "Usage: /memory [show|add|clear]"
    return f"__ACTION__:memory:{sub}"


# Command registry

COMMANDS: dict[str, Command] = {
    "/help": Command(
        name="/help",
        description="Show available commands",
        handler=_noop_handler,
        aliases=["/h", "/?"],
    ),
    "/exit": Command(
        name="/exit",
        description="Exit RUNE",
        handler=_noop_handler,
        aliases=["/quit", "/q"],
    ),
    "/clear": Command(
        name="/clear",
        description="Clear screen and history",
        handler=_noop_handler,
        aliases=["/cls"],
    ),
    "/model": Command(
        name="/model",
        description="Select or switch LLM model",
        handler=_noop_handler,
        aliases=["/models"],
        usage="/model [provider:model]",
    ),
    "/theme": Command(
        name="/theme",
        description="Switch theme (dark/light/minimal)",
        handler=_theme_handler,
        usage="/theme [name]",
    ),
    "/config": Command(
        name="/config",
        description="Show current configuration",
        handler=_config_handler,
    ),
    "/retry": Command(
        name="/retry",
        description="Re-run the last message",
        handler=_retry_handler,
        aliases=["/r"],
    ),
    "/undo": Command(
        name="/undo",
        description="Undo last file change",
        handler=_undo_handler,
        aliases=["/u"],
    ),
    "/style": Command(
        name="/style",
        description="Set or cycle output style",
        handler=_style_handler,
        usage="/style [compact|normal|verbose]",
    ),
    "/files": Command(
        name="/files",
        description="Show file changes this session",
        handler=_files_handler,
        aliases=["/f"],
    ),
    "/diff": Command(
        name="/diff",
        description="Show git diff",
        handler=_diff_handler,
        aliases=["/d"],
    ),
    "/copy": Command(
        name="/copy",
        description="Copy last response to clipboard",
        handler=_copy_handler,
        aliases=["/cp"],
    ),
    "/export": Command(
        name="/export",
        description="Export conversation (markdown/json/html)",
        handler=_export_handler,
        usage="/export [format]",
    ),
    "/save": Command(
        name="/save",
        description="Save current session",
        handler=_save_handler,
        usage="/save [name]",
    ),
    "/load": Command(
        name="/load",
        description="Load a saved session",
        handler=_load_handler,
        usage="/load [id]",
    ),
    "/sessions": Command(
        name="/sessions",
        description="List saved sessions",
        handler=_sessions_handler,
    ),
    "/search": Command(
        name="/search",
        description="Search message history",
        handler=_search_handler,
        usage="/search <query>",
    ),
    "/memory": Command(
        name="/memory",
        description="Project memory (show/add/clear)",
        handler=_memory_handler,
        usage="/memory [show|add|clear]",
    ),
    "/cost": Command(
        name="/cost",
        description="Show estimated API cost",
        handler=_cost_handler,
    ),
    "/stats": Command(
        name="/stats",
        description="Show session statistics",
        handler=_stats_handler,
    ),
    "/status": Command(
        name="/status",
        description="Show agent status",
        handler=_status_handler,
    ),
    # Hidden shortcuts (not shown in /help, but callable and Tab-completable)
    "/compact": Command(
        name="/compact", description="Compact output", handler=_compact_handler, hidden=True,
    ),
    "/normal": Command(
        name="/normal", description="Normal output", handler=_normal_handler, hidden=True,
    ),
    "/verbose": Command(
        name="/verbose", description="Verbose output", handler=_verbose_handler, hidden=True,
    ),
    "/learning": Command(
        name="/learning",
        description="Toggle background learning (on/off)",
        handler=_noop_handler,  # Handled by app.py directly
    ),
}

# Build alias lookup
_ALIAS_MAP: dict[str, str] = {}
for _cmd in COMMANDS.values():
    for _alias in _cmd.aliases:
        _ALIAS_MAP[_alias] = _cmd.name


# Parser

def parse_slash_command(text: str) -> tuple[str, str] | None:
    """Parse a slash command from text.

    Returns (command_name, args) if the text is a slash command, else None.
    The command_name is the canonical name (aliases resolved).
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # Resolve alias
    canonical = _ALIAS_MAP.get(cmd, cmd)

    if canonical in COMMANDS:
        return (canonical, args)

    return None


def suggest_command(text: str) -> str | None:
    """Suggest closest matching command for a typo."""
    cmd = text.strip().split()[0].lower() if text.strip() else ""
    if not cmd.startswith("/"):
        return None

    all_names = list(COMMANDS.keys()) + list(_ALIAS_MAP.keys())
    matches = get_close_matches(cmd, all_names, n=1, cutoff=0.5)
    if matches:
        canonical = _ALIAS_MAP.get(matches[0], matches[0])
        return canonical
    return None
