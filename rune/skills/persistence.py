"""Persist skill instructions, prerequisites and lifecycle state across restarts."""

from __future__ import annotations

from pathlib import Path

from rune.skills.lifecycle import STATE_KEY, get_state
from rune.skills.types import Skill
from rune.utils.logger import get_logger

log = get_logger(__name__)

def _skill_dir(skill: Skill) -> Path:
    from rune.safety.execution_environment import execution_workspace
    from rune.utils.paths import rune_home
    base = (Path(execution_workspace()) / ".rune" / "skills" if skill.scope == "project"
            else rune_home() / "skills")
    target = (base / skill.name).resolve()
    # Reject frontmatter names that could escape the skills directory.
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"skill name escapes the skills directory: {skill.name!r}")
    return target


def _render(skill: Skill) -> str:
    """Preserve structured metadata when a skill's lifecycle changes."""
    import io

    from ruamel.yaml import YAML

    metadata = {**skill.metadata, "name": skill.name, "description": skill.description,
                "scope": skill.scope, "author": skill.author, STATE_KEY: get_state(skill)}
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    stream = io.StringIO()
    yaml.dump(metadata, stream)
    return "---\n" + stream.getvalue() + "---\n" + skill.body.rstrip() + "\n"


def write_skill_to_disk(skill: Skill) -> str | None:
    """Save SKILL.md and record its path; return None on failure without raising."""
    from rune.skills.validator import validate_name

    check = validate_name(skill.name)
    if not check.valid:
        log.warning("skill_persist_rejected", name=skill.name, errors=check.errors)
        return None

    try:
        d = _skill_dir(skill)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "SKILL.md"
        path.write_text(_render(skill), encoding="utf-8")
        skill.file_path = str(path)
        log.info("skill_persisted", name=skill.name, state=get_state(skill))
        return str(path)
    except Exception as exc:
        log.debug("skill_persist_failed", name=skill.name, error=str(exc)[:120])
        return None


def persist_skill_state(skill: Skill) -> bool:
    """Update the existing skill file, preserving its body; return False for in-memory skills."""
    if not skill.file_path:
        return False
    try:
        Path(skill.file_path).write_text(_render(skill), encoding="utf-8")
        log.info("skill_state_persisted", name=skill.name,
                 state=get_state(skill))
        return True
    except Exception as exc:
        log.debug("skill_state_persist_failed", name=skill.name,
                  error=str(exc)[:120])
        return False
