"""Load, register and search file-backed and programmatic skills."""

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


def _parse_skill_file(path: Path) -> Skill | None:
    """Parse YAML frontmatter and Markdown instructions from SKILL.md."""
    try:
        with path.open(encoding="utf-8") as stream:
            content = stream.read(128_001)
        if len(content) > 128_000:
            raise ValueError("Skill exceeds 128,000 characters")
    except (OSError, UnicodeError, ValueError) as exc:
        log.warning("skill_read_error", path=str(path), error=str(exc))
        return None

    metadata: dict[str, Any] = {}
    body = content

    fm_match = _FRONTMATTER_RE.match(content)
    if fm_match:
        from ruamel.yaml import YAML

        try:
            metadata = YAML(typ="safe").load(fm_match.group(1)) or {}
            if not isinstance(metadata, dict):
                raise ValueError("Skill frontmatter must be a mapping")
            for key in ("name", "description", "scope", "author"):
                if key in ("description", "author") and metadata.get(key) is None:
                    metadata[key] = ""
                if key in metadata and not isinstance(metadata[key], str):
                    raise ValueError(f"Skill {key} must be text")
        except Exception as exc:
            log.warning("skill_metadata_invalid", path=str(path), error=str(exc)[:200])
            return None
        body = content[fm_match.end():]

    name = metadata.pop("name", path.stem)
    description = metadata.pop("description", "").strip()
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


class SkillRegistry:
    """Registry for loading, storing, and searching skills."""

    __slots__ = ("_skills",)

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def load_skills(self, directory: str | Path) -> int:
        """Load skill files from a directory and return the count."""
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


def get_skill_registry(workspace: str | Path | None = None) -> SkillRegistry:
    """Load run-specific skills, or reuse the settings registry when unscoped."""
    global _registry
    if workspace is not None:
        from rune.utils.paths import rune_home

        scoped = SkillRegistry()
        user_skills = rune_home() / "skills"
        if user_skills.is_dir():
            scoped.load_skills(user_skills)
        # Programmatic user skills have no file to reload.
        if _registry is not None:
            for skill in _registry.list():
                if not skill.file_path and skill.scope == "user" and scoped.get(skill.name) is None:
                    scoped.register(skill)
        project_skills = Path(workspace).resolve() / ".rune" / "skills"
        if project_skills.is_dir():
            scoped.load_skills(project_skills)
        return scoped
    if _registry is None:
        _registry = SkillRegistry()
        # Auto-load from standard directories
        from rune.utils.paths import rune_home
        user_skills = rune_home() / "skills"
        if user_skills.is_dir():
            _registry.load_skills(user_skills)
        # Project-level skills
        project_skills = Path.cwd() / ".rune" / "skills"
        if project_skills.is_dir():
            _registry.load_skills(project_skills)
    return _registry
