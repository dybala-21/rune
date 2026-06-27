"""Disk persistence for gated skill learning (T1-1).

Distillation registers a skill in-memory, but the daemon that evaluates it runs
in a *separate process*, so a candidate must be on disk for the daemon to see
it. This writes/updates SKILL.md using FLAT frontmatter keys — the format the
registry parser (`rune.skills.registry._parse_skill_file`) reads — so a
persisted ``state:`` round-trips back into ``skill.metadata["state"]``.

Only scalar metadata is written to frontmatter (lists/dicts are skipped); the
lifecycle ``state`` is the field that must survive a restart.
"""

from __future__ import annotations

from pathlib import Path

from rune.skills.lifecycle import STATE_KEY, get_state
from rune.skills.types import Skill
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Frontmatter keys the registry treats specially (not part of free metadata).
_RESERVED = ("name", "description", "scope", "author")


def _skill_dir(skill: Skill) -> Path:
    from rune.utils.paths import rune_home
    base = (Path.cwd() / ".rune" / "skills" if skill.scope == "project"
            else rune_home() / "skills")
    return base / skill.name


def _render(skill: Skill) -> str:
    """Render a SKILL.md with flat frontmatter the registry can parse."""
    lines = ["---", f"name: {skill.name}",
             f"description: {skill.description}", f"scope: {skill.scope}"]
    if skill.author:
        lines.append(f"author: {skill.author}")
    # State first among metadata so it is easy to eyeball.
    lines.append(f"{STATE_KEY}: {get_state(skill)}")
    for k, v in skill.metadata.items():
        if k in (*_RESERVED, STATE_KEY):
            continue
        if isinstance(v, (str, int, float, bool)):  # scalars only — flat format
            lines.append(f"{k}: {v}")
    lines.append("---")
    body = skill.body if skill.body.endswith("\n") else skill.body + "\n"
    return "\n".join(lines) + "\n" + body


def write_skill_to_disk(skill: Skill) -> str | None:
    """Write *skill* to its scope dir as SKILL.md. Returns the path or None.

    Records the path on ``skill.file_path`` so later state updates rewrite the
    same file. Best-effort; never raises.
    """
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
    """Rewrite a skill's on-disk SKILL.md to reflect its current state.

    No-op (returns False) for in-memory skills with no ``file_path``. The body
    and other frontmatter are preserved via :func:`write_skill_to_disk`.
    """
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
