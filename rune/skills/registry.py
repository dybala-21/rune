"""Skill registry for RUNE.

Loads, stores, and searches skills from SKILL.md files and
programmatic registration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rune.skills.matcher import match_skills
from rune.skills.types import Skill, SkillMatch
from rune.utils.logger import get_logger

log = get_logger(__name__)

_registry: SkillRegistry | None = None


# SKILL.md parser

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)


def _parse_skill_file(path: Path) -> Skill | None:
    """Parse a SKILL.md file into a Skill object.

    Expected format::

        ---
        name: my-skill
        description: Does something useful
        scope: user
        author: someone
        ---

        Body content with instructions...
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("skill_read_error", path=str(path), error=str(exc))
        return None

    metadata: dict[str, Any] = {}
    body = content

    fm_match = _FRONTMATTER_RE.match(content)
    if fm_match:
        frontmatter = fm_match.group(1)
        for kv in _KV_RE.finditer(frontmatter):
            metadata[kv.group(1)] = kv.group(2).strip()
        body = content[fm_match.end():]

    name = metadata.pop("name", path.stem)
    description = metadata.pop("description", "")
    scope = metadata.pop("scope", "user")
    author = metadata.pop("author", "")

    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        scope=scope,
        author=author,
        metadata=metadata,
        file_path=str(path),
    )


# SkillRegistry

class SkillRegistry:
    """Registry for loading, storing, and searching skills."""

    __slots__ = ("_skills",)

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def load_skills(self, directory: str | Path) -> int:
        """Load all SKILL.md files from *directory*.

        Returns the number of skills loaded.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            log.warning("skill_dir_not_found", directory=str(dir_path))
            return 0

        count = 0
        for path in dir_path.rglob("SKILL.md"):
            skill = _parse_skill_file(path)
            if skill:
                self._skills[skill.name] = skill
                count += 1
                log.debug("skill_loaded", name=skill.name, path=str(path))

        # Also load *.skill.md files
        for path in dir_path.rglob("*.skill.md"):
            skill = _parse_skill_file(path)
            if skill:
                self._skills[skill.name] = skill
                count += 1

        log.info("skills_loaded", count=count, directory=str(dir_path))
        return count

    def register(self, skill: Skill) -> None:
        """Register a skill programmatically."""
        self._skills[skill.name] = skill
        log.debug("skill_registered", name=skill.name)

    def unregister(self, name: str) -> None:
        """Remove a skill by name."""
        removed = self._skills.pop(name, None)
        if removed:
            log.debug("skill_unregistered", name=name)

    def get(self, name: str) -> Skill | None:
        """Get a skill by exact name."""
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def search(self, query: str) -> list[SkillMatch]:
        """Search skills by query string. Returns matches sorted by score."""
        return match_skills(query, self.list())


def get_skill_registry() -> SkillRegistry:
    """Get or create the singleton SkillRegistry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
