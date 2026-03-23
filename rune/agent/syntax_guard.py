"""Syntax validation for file write/edit operations.

Registry-based: add new languages with @register decorator.
Validates content before disk write — rejects invalid syntax
with line number + error message so the agent can self-correct.
"""

from __future__ import annotations

from pathlib import Path

_VALIDATORS: dict[str, object] = {}
_MAX_VALIDATE_SIZE = 512_000  # 500KB — skip large files


def register(extensions: list[str]):
    """Decorator to register a syntax validator for file extensions."""
    def decorator(fn):  # type: ignore[no-untyped-def]
        for ext in extensions:
            _VALIDATORS[ext] = fn
        return fn
    return decorator


def validate(path: str, content: str) -> str | None:
    """Validate syntax of content for the given file path.

    Returns an error string (e.g. 'line 42: unexpected indent')
    if syntax is invalid. Returns None if valid or not validatable.
    """
    if len(content) > _MAX_VALIDATE_SIZE:
        return None
    ext = Path(path).suffix.lower()
    fn = _VALIDATORS.get(ext)
    if fn is None:
        return None
    try:
        return fn(content)  # type: ignore[operator]
    except Exception as e:
        return str(e)[:200]


@register([".py"])
def _python(content: str) -> str | None:
    import ast
    try:
        ast.parse(content)
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    return None


@register([".json"])
def _json(content: str) -> str | None:
    import json
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        return f"line {e.lineno}: {e.msg}"
    return None


@register([".toml"])
def _toml(content: str) -> str | None:
    import tomllib
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        return str(e)[:200]
    return None


@register([".yaml", ".yml"])
def _yaml(content: str) -> str | None:
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")
        yaml.load(content)
    except Exception as e:
        msg = str(e)
        # Extract line info if available
        if "line" in msg.lower():
            return msg[:200]
        return f"YAML error: {msg[:200]}"
    return None
