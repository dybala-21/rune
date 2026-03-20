"""Skill file writer for RUNE.

Renders skill definitions as SKILL.md files with YAML frontmatter and
optional HMAC-SHA256 signatures, then writes them to the appropriate
scope directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rune.skills.key_manager import (
    DEFAULT_SKILL_SIGNING_ENV,
    resolve_skill_signing_secret,
)
from rune.skills.signing import sign_skill_payload

try:
    import yaml as _yaml  # PyYAML

    def _dump_yaml(data: dict[str, Any]) -> str:
        return _yaml.dump(data, default_flow_style=False, allow_unicode=True).rstrip()

except ImportError:  # pragma: no cover – fallback without PyYAML
    from rune.utils.fast_serde import json_encode as _json_encode

    def _dump_yaml(data: dict[str, Any]) -> str:  # type: ignore[misc]
        # Simple key: value rendering for flat-ish dicts
        lines: list[str] = []
        for k, v in data.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sk, sv in v.items():
                    lines.append(f"  {sk}: {_json_encode(sv)}")
            else:
                lines.append(f"{k}: {_json_encode(v)}")
        return "\n".join(lines)


@dataclass(slots=True)
class WriteSkillInput:
    """Parameters for writing a skill file."""

    name: str = ""
    description: str = ""
    body: str = ""
    scope: str = "user"  # "user" | "project"
    author: str = ""
    signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    base_dir: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class WriteSkillResult:
    """Outcome of a skill file write."""

    skill_dir: str = ""
    skill_md_path: str = ""
    content: str = ""
    scope: str = "user"
    author: str = ""
    signed: bool = False
    signature: str | None = None


# Helpers

def _render_skill_markdown(
    frontmatter: dict[str, Any], body: str,
) -> str:
    yaml_str = _dump_yaml(frontmatter)
    normalized_body = body if body.endswith("\n") else f"{body}\n"
    return f"---\n{yaml_str}\n---\n\n{normalized_body}"


def _parse_skill_markdown(content: str) -> tuple[dict[str, Any] | None, str]:
    """Parse frontmatter + body from a SKILL.md string."""
    import re
    m = re.match(r"^---\n([\s\S]*?)\n---\n([\s\S]*)$", content)
    if not m:
        return None, content
    try:
        import yaml
        fm = yaml.safe_load(m.group(1))
        return fm, m.group(2).strip()
    except Exception:
        return None, content


def _resolve_base_dir(scope: str, override: str | None = None) -> str:
    if override:
        return override
    if scope == "project":
        return os.path.join(os.getcwd(), ".rune", "skills")
    # User scope
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config = xdg if xdg else os.path.join(Path.home(), ".config", "rune")
    return os.path.join(config, "skills")


# Public API

def write_skill_file(inp: WriteSkillInput) -> WriteSkillResult:
    """Write a skill definition to disk as a SKILL.md file.

    If no signature is provided, the writer will attempt to auto-sign
    the skill using the resolved signing secret.

    Parameters:
        inp: Skill write parameters.

    Returns:
        A :class:`WriteSkillResult` describing what was written.
    """
    base_dir = _resolve_base_dir(inp.scope, inp.base_dir)
    skill_dir = os.path.join(base_dir, inp.name)
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    created_at = (inp.created_at or datetime.now()).isoformat()

    metadata_base: dict[str, Any] = {
        "author": inp.author,
        "version": "1.0.0",
        "created": created_at,
        **inp.metadata,
    }

    effective_name = inp.name
    effective_description = inp.description
    effective_body = inp.body
    effective_signature = inp.signature

    # Auto-sign if no signature provided
    if not effective_signature:
        resolution = resolve_skill_signing_secret(
            env_var_name=DEFAULT_SKILL_SIGNING_ENV,
            create_if_missing=True,
        )
        if resolution.secret:
            effective_signature = sign_skill_payload(
                name=effective_name,
                description=effective_description,
                body=effective_body,
                scope=inp.scope,
                author=inp.author,
                secret=resolution.secret,
            )

    metadata = {
        **metadata_base,
        **({"signature": effective_signature} if effective_signature else {}),
    }

    frontmatter: dict[str, Any] = {
        "name": effective_name,
        "description": effective_description,
        "metadata": metadata,
    }

    content = _render_skill_markdown(frontmatter, effective_body)

    Path(skill_dir).mkdir(parents=True, exist_ok=True)
    Path(skill_md_path).write_text(content, encoding="utf-8")

    return WriteSkillResult(
        skill_dir=skill_dir,
        skill_md_path=skill_md_path,
        content=content,
        scope=inp.scope,
        author=inp.author,
        signed=bool(effective_signature),
        signature=effective_signature,
    )
