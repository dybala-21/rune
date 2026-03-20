"""CLI-accessible project memory operations.

Ported from src/memory/project-memory-command.ts - parse and execute
/memory slash commands (show, add, help) and natural-language equivalents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rune.memory.project_paths import get_project_memory_paths
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

ProjectMemoryAction = Literal["show", "add", "help"]


@dataclass(slots=True)
class ProjectMemoryCommand:
    action: ProjectMemoryAction
    source: Literal["slash", "natural"]
    section: str | None = None
    content: str | None = None
    parse_error: str | None = None


# Section normalization

_SECTION_ALIASES: dict[str, str] = {
    "pref": "Preferences",
    "preference": "Preferences",
    "preferences": "Preferences",
    "env": "Environment",
    "environment": "Environment",
    "decision": "Decisions",
    "decisions": "Decisions",
    "pattern": "Patterns",
    "patterns": "Patterns",
    "note": "Notes",
    "notes": "Notes",
}


def _normalize_section(raw: str | None) -> str:
    trimmed = (raw or "").strip()
    if not trimmed:
        return "Notes"
    return _SECTION_ALIASES.get(trimmed.lower(), trimmed)


# Payload parsing

def _parse_add_payload(payload: str) -> tuple[str, str] | None:
    """Parse section + content from the add payload.

    Supports:
      - ``section | content`` (pipe separator)
      - ``section: content`` (colon separator)
      - ``content`` (defaults to Notes)

    Returns (section, content) or None on failure.
    """
    trimmed = payload.strip()
    if not trimmed:
        return None

    # Pipe separator
    if "|" in trimmed:
        parts = trimmed.split("|", 1)
        section = _normalize_section(parts[0])
        content = parts[1].strip() if len(parts) > 1 else ""
        if not content:
            return None
        return section, content

    # Colon separator: ``SectionName: content``
    colon_match = re.match(r"^([A-Za-z][A-Za-z _-]{1,20})\s*:\s*(.+)$", trimmed)
    if colon_match:
        section = _normalize_section(colon_match.group(1))
        content = colon_match.group(2).strip()
        if not content:
            return None
        return section, content

    # Default: Notes section
    return "Notes", trimmed


# Slash command parsing

def parse_project_memory_slash_args(args: str) -> ProjectMemoryCommand:
    """Parse ``/memory <args>`` arguments into a command object."""
    trimmed = args.strip()

    if not trimmed or trimmed in ("show", "list"):
        return ProjectMemoryCommand(action="show", source="slash")

    if trimmed == "help":
        return ProjectMemoryCommand(action="help", source="slash")

    add_match = re.match(r"^(?:add|append|save)\s+(.+)$", trimmed, re.I)
    if add_match:
        parsed = _parse_add_payload(add_match.group(1))
        if not parsed:
            return ProjectMemoryCommand(
                action="help",
                source="slash",
                parse_error="Usage: /memory add [section] | <content>",
            )
        return ProjectMemoryCommand(
            action="add",
            source="slash",
            section=parsed[0],
            content=parsed[1],
        )

    # Bare text -> Notes append
    fallback = _parse_add_payload(trimmed)
    if fallback:
        return ProjectMemoryCommand(
            action="add",
            source="slash",
            section=fallback[0],
            content=fallback[1],
        )

    return ProjectMemoryCommand(
        action="help",
        source="slash",
        parse_error="Usage: /memory [show|help|add]",
    )


def parse_project_memory_command(
    text: str,
    *,
    allow_natural: bool = False,
) -> ProjectMemoryCommand | None:
    """Parse a user message as a project memory command.

    Recognises ``/memory ...`` slash commands and, if *allow_natural*
    is True, natural-language equivalents.
    """
    trimmed = text.strip()
    if not trimmed:
        return None

    if trimmed.startswith("/memory"):
        args = trimmed[len("/memory"):].strip()
        return parse_project_memory_slash_args(args)

    if not allow_natural:
        return None

    # Natural: "memory show" / "memory list"
    if re.match(r"^(?:memory)\s*(?:show|list)$", trimmed, re.I):
        return ProjectMemoryCommand(action="show", source="natural")

    return None


# Help text

def _build_help_text(parse_error: str | None = None) -> str:
    header = f"{parse_error}\n\n" if parse_error else ""
    return (
        f"{header}Project memory commands:\n"
        "- /memory show\n"
        "- /memory add <content>\n"
        "- /memory add <section> | <content>\n"
        "- /memory help\n"
        "\n"
        "Examples:\n"
        "- /memory add Decisions | Rollback strategy = blue/green\n"
        "- /memory add Notes | Release date moved to Friday"
    )


# Execution

async def execute_project_memory_command(
    command: ProjectMemoryCommand,
    *,
    workspace_path: str | None = None,
    rune_config_dir: str | None = None,
    show_max_lines: int = 120,
    show_max_chars: int = 8000,
) -> dict[str, object]:
    """Execute a parsed project memory command.

    Returns ``{"success": bool, "message": str}``.
    """
    ws = workspace_path or str(Path.cwd())

    if command.action == "help":
        return {"success": True, "message": _build_help_text(command.parse_error)}

    paths = get_project_memory_paths(ws, rune_config_dir)

    if command.action == "show":
        memory_file = paths.memory_file
        if not memory_file.exists():
            return {
                "success": True,
                "message": f"Project Memory ({paths.project_key}) is empty.",
            }
        text = memory_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) > show_max_lines:
            lines = lines[:show_max_lines]
        snippet = "\n".join(lines)
        if len(snippet) > show_max_chars:
            snippet = snippet[:show_max_chars] + "\n..."
        return {
            "success": True,
            "message": f"Project Memory ({paths.project_key}):\n\n{snippet}",
        }

    if command.action == "add":
        section = _normalize_section(command.section)
        content = (command.content or "").strip()
        if not content:
            return {"success": False, "message": _build_help_text("Content is required.")}

        # Ensure directory exists
        paths.memory_dir.mkdir(parents=True, exist_ok=True)

        memory_file = paths.memory_file
        existing = memory_file.read_text(encoding="utf-8") if memory_file.exists() else ""

        # Find or create section header
        section_header = f"## {section}"
        if section_header in existing:
            # Append under existing section
            idx = existing.index(section_header)
            # Find end of section (next ## or EOF)
            next_section = existing.find("\n## ", idx + len(section_header))
            if next_section == -1:
                # Append at end
                updated = existing.rstrip() + f"\n- {content}\n"
            else:
                # Insert before next section
                updated = (
                    existing[:next_section].rstrip()
                    + f"\n- {content}\n"
                    + existing[next_section:]
                )
        else:
            # Create new section at end
            updated = existing.rstrip() + f"\n\n{section_header}\n- {content}\n"

        memory_file.write_text(updated, encoding="utf-8")
        log.info("project_memory_added", section=section, content=content[:80])
        return {
            "success": True,
            "message": f"Saved to project memory [{section}]: {content}",
        }

    return {"success": False, "message": _build_help_text("Unknown memory command.")}
