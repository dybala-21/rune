"""Workspace guard - path scope enforcement for workspace safety.

Ported from src/agent/workspace-guard.ts (627 lines) - prevents agent
from reading/writing/executing outside the designated workspace and
explicit allowlist paths.

findBashScopeViolations(): Check bash command for out-of-scope path access.
findWritePathScopeViolation(): Check a single write path against scope.
extractExplicitPathAllowlist(): Extract path candidates from user goal text.
extractIntentionalPathAllowlist(): Extract workspace-intent paths (with context).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

ScopeViolationSource = Literal[
    "file.path",
    "bash.cwd",
    "bash.cd",
    "bash.arg",
    "bash.redirect",
]


@dataclass(slots=True)
class ScopeViolation:
    """A detected scope violation."""

    source: ScopeViolationSource
    requested: str
    resolved: str


@dataclass(slots=True)
class ScopeGuardOptions:
    """Options for scope guard checks."""

    workspace_root: str
    explicit_allowlist: list[str] = field(default_factory=list)


# Constants

SAFE_REDIRECT_TARGETS = frozenset(["/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"])

LONG_ARG_OPTIONS = ("--cwd", "--prefix", "--manifest-path")

_PATH_DELIMITER_CHARS = frozenset('"\'`,;|)([]{}<>')

_INTENT_CONTEXT_STEMS = [
    "workspace", "workdir", "cwd", "repo", "project",
]

_INTENT_ACTION_STEMS = [
    "continue", "resume", "switch", "follow", "cd",
]


# Path helpers

def _expand_home(raw_path: str) -> str:
    """Expand ~ to HOME directory."""
    home = os.environ.get("HOME", "")
    if not home:
        return raw_path
    if raw_path == "~":
        return home
    if raw_path.startswith("~/"):
        return os.path.join(home, raw_path[2:])
    return raw_path


def _expand_leading_env_var(raw_path: str) -> tuple[str, bool]:
    """Expand leading $VAR or ${VAR}. Returns (expanded, unresolved)."""
    if not raw_path.startswith("$"):
        return raw_path, False

    if raw_path.startswith("${"):
        close_at = raw_path.find("}", 2)
        if close_at <= 2:
            return raw_path, False
        name = raw_path[2:close_at]
        if not name or not (name[0].isalpha() or name[0] == "_"):
            return raw_path, False
        env_value = os.environ.get(name)
        if not env_value:
            return raw_path, True
        return env_value + raw_path[close_at + 1:], False

    # $NAME format
    if len(raw_path) < 2 or not (raw_path[1].isalpha() or raw_path[1] == "_"):
        return raw_path, False
    end = 2
    while end < len(raw_path) and (raw_path[end].isalnum() or raw_path[end] == "_"):
        end += 1
    name = raw_path[1:end]
    env_value = os.environ.get(name)
    if not env_value:
        return raw_path, True
    return env_value + raw_path[end:], False


def _trim_path_token(token: str) -> str:
    """Strip trailing delimiters from a path token."""
    out = token.strip()
    while out:
        tail = out[-1]
        if tail in ")]},.;:":
            out = out[:-1]
            continue
        break
    return out


def _normalize_path(raw_path: str, workspace_root: str) -> str:
    """Normalize a path: trim, expand env/home, resolve relative."""
    trimmed = _trim_path_token(raw_path.strip())
    expanded_env, _ = _expand_leading_env_var(trimmed)
    expanded = _expand_home(expanded_env)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(workspace_root, expanded))


def _is_within(base_dir: str, target_path: str) -> bool:
    """Check if target_path is same as or under base_dir."""
    rel = os.path.relpath(target_path, base_dir)
    return rel == "" or (not rel.startswith("..") and not os.path.isabs(rel))


def _is_allowed_by_scope(target_path: str, options: ScopeGuardOptions) -> bool:
    """Check if target is within workspace or explicit allowlist."""
    workspace_root = os.path.normpath(os.path.abspath(options.workspace_root))
    if _is_within(workspace_root, target_path):
        return True

    return any(
        _is_within(_normalize_path(allowed, workspace_root), target_path)
        for allowed in options.explicit_allowlist
    )


# Shell tokenizer (minimal)

@dataclass(slots=True)
class _ShellToken:
    kind: Literal["word", "op"]
    value: str


def _tokenize_shell_command(command: str) -> list[_ShellToken]:
    """Minimal shell command tokenizer for scope checking."""
    tokens: list[_ShellToken] = []
    current = ""
    quote: str | None = None
    escaped = False

    def flush_word() -> None:
        nonlocal current
        if current:
            tokens.append(_ShellToken(kind="word", value=current))
            current = ""

    for i, ch in enumerate(command):
        if escaped:
            current += ch
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue

        if quote:
            if ch == quote:
                quote = None
            else:
                current += ch
            continue

        if ch in ("'", '"'):
            quote = ch
            continue

        if ch in (" ", "\t", "\n", "\r"):
            flush_word()
            continue

        if ch in (";", "|", "&", ">", "<"):
            flush_word()
            op = ch
            if i + 1 < len(command) and command[i + 1] == ch and ch in ("|", "&", ">", "<"):
                op += ch
                # Skip next char handled by enumerate offset
            tokens.append(_ShellToken(kind="op", value=op))
            continue

        current += ch

    flush_word()
    return tokens


def _next_word_token(tokens: list[_ShellToken], start: int) -> str | None:
    """Find the next word token after start index."""
    for i in range(start, len(tokens)):
        t = tokens[i]
        if t.kind == "word" and t.value.strip():
            return t.value
        if t.kind == "op":
            return None
    return None


def _extract_long_option_value(word: str) -> str | None:
    """Extract value from --option=value for known long options."""
    for option in LONG_ARG_OPTIONS:
        prefix = f"{option}="
        if word.lower().startswith(prefix.lower()):
            return word[len(prefix):]
    return None


# Path candidate extraction

def _is_path_start(s: str, idx: int) -> bool:
    """Check if a path-like token starts at index."""
    ch = s[idx]
    if ch == "/":
        return True
    if ch == "~" and idx + 1 < len(s) and s[idx + 1] == "/":
        return True
    if ch == ".":
        if idx + 1 < len(s) and s[idx + 1] == "/":
            return True
        if idx + 2 < len(s) and s[idx + 1] == "." and s[idx + 2] == "/":
            return True
    return False


def _normalize_extracted_path_token(raw: str) -> str:
    """Normalize a raw path token extracted from text."""
    token = _trim_path_token(raw)
    if not token:
        return ""
    if token.startswith("//"):
        return ""
    if not (
        token.startswith("/")
        or token.startswith("./")
        or token.startswith("../")
        or token.startswith("~/")
    ):
        return ""
    return token


def _scan_path_candidates(text: str) -> list[tuple[str, int, int]]:
    """Scan text for path-like candidates. Returns [(token, start, end)]."""
    candidates: list[tuple[str, int, int]] = []
    i = 0
    while i < len(text):
        if not _is_path_start(text, i):
            i += 1
            continue
        # Check boundary before path
        if i > 0 and text[i - 1] not in (" ", "\t", "\n", "\r", "(", "[", "{", '"', "'", "`"):
            i += 1
            continue

        end = i
        while end < len(text) and text[end] not in (" ", "\t", "\n", "\r") and text[end] not in _PATH_DELIMITER_CHARS:
            end += 1

        raw_token = text[i:end]
        token = _normalize_extracted_path_token(raw_token)
        if not token:
            i = max(end, i + 1)
            continue

        candidates.append((token, i, i + len(token)))
        i = max(end, i + 1)

    return candidates


# Public API

def extract_explicit_path_allowlist(goal: str, workspace_root: str) -> list[str]:
    """Extract all path candidates from goal text, normalized to absolute paths."""
    allowlist: set[str] = set()
    for token, _, _ in _scan_path_candidates(goal):
        allowlist.add(_normalize_path(token, workspace_root))
    return list(allowlist)


def extract_intentional_path_allowlist(goal: str, workspace_root: str) -> list[str]:
    """Extract workspace-intent paths (with context analysis).

    Filters path candidates to only those with surrounding workspace/action
    intent signals, excluding paths in code fences and report lines.
    """
    allowlist: set[str] = set()
    # Simple filtering: skip code fence content
    in_code_fence = False
    for line in goal.split("\n"):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        if not trimmed:
            continue

        for token, start, end in _scan_path_candidates(line):
            # Check for workspace intent signals in surrounding context (outside the path)
            surrounding = (line[:start] + " " + line[end:]).lower()
            has_context = any(s in surrounding for s in _INTENT_CONTEXT_STEMS)
            has_action = any(s in surrounding for s in _INTENT_ACTION_STEMS)
            if has_context or has_action:
                allowlist.add(_normalize_path(token, workspace_root))

    return list(allowlist)


def find_write_path_scope_violation(
    raw_path: str | None,
    options: ScopeGuardOptions,
) -> ScopeViolation | None:
    """Check if a write path is outside the allowed scope.

    Returns a ScopeViolation if the path is out of scope, otherwise None.
    """
    if not raw_path or not isinstance(raw_path, str) or not raw_path.strip():
        return None

    resolved = _normalize_path(raw_path, options.workspace_root)
    if _is_allowed_by_scope(resolved, options):
        return None

    return ScopeViolation(
        source="file.path",
        requested=raw_path,
        resolved=resolved,
    )


def find_bash_scope_violations(
    *,
    cwd: str | None = None,
    command: str | None = None,
    workspace_root: str,
    explicit_allowlist: list[str] | None = None,
) -> list[ScopeViolation]:
    """Check a bash command for out-of-scope path access.

    Inspects:
    - cwd parameter
    - cd targets
    - redirect targets (>, >>)
    - --cwd/--prefix/--manifest-path arguments
    - -C arguments
    """
    options = ScopeGuardOptions(
        workspace_root=workspace_root,
        explicit_allowlist=explicit_allowlist or [],
    )
    violations: list[ScopeViolation] = []

    def check_path(raw: str, source: ScopeViolationSource) -> None:
        token = raw.strip()
        if not token or token == "-":
            return
        if source == "bash.redirect" and token in SAFE_REDIRECT_TARGETS:
            return

        expanded, unresolved = _expand_leading_env_var(token)
        if unresolved:
            violations.append(ScopeViolation(source=source, requested=token, resolved=token))
            return

        resolved = _normalize_path(expanded, options.workspace_root)
        if not _is_allowed_by_scope(resolved, options):
            violations.append(ScopeViolation(source=source, requested=token, resolved=resolved))

    # Check cwd
    if cwd and isinstance(cwd, str) and cwd.strip():
        check_path(cwd, "bash.cwd")

    # Check command
    if not command or not isinstance(command, str) or not command.strip():
        return violations

    tokens = _tokenize_shell_command(command)
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.kind == "op":
            if token.value in (">", ">>"):
                target = _next_word_token(tokens, i + 1)
                if target:
                    check_path(target, "bash.redirect")
            i += 1
            continue

        value = token.value

        if value == "cd":
            target = _next_word_token(tokens, i + 1)
            if target:
                check_path(target, "bash.cd")
            i += 1
            continue

        if value == "-C":
            target = _next_word_token(tokens, i + 1)
            if target:
                check_path(target, "bash.arg")
            i += 1
            continue

        if value.startswith("-C") and len(value) > 2:
            check_path(value[2:], "bash.arg")
            i += 1
            continue

        long_option_value = _extract_long_option_value(value)
        if long_option_value:
            check_path(long_option_value, "bash.arg")
            i += 1
            continue

        lower = value.lower()
        if lower in ("--cwd", "--prefix", "--manifest-path"):
            target = _next_word_token(tokens, i + 1)
            if target:
                check_path(target, "bash.arg")

        i += 1

    return violations
