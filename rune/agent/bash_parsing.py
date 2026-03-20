"""Bash command parsing utilities for RUNE.

Ported from src/agent/loop.ts lines 2461-2672 - shell tokenization,
command classification (verification, load-test, service start/probe/cleanup).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TypedDict

# Constants

VERIFICATION_SINGLE_WORD_COMMANDS: frozenset[str] = frozenset({
    "pytest", "vitest", "jest",
})

LOAD_TEST_COMMAND_HEADS: frozenset[str] = frozenset({
    "loadtest", "k6", "wrk", "hey", "vegeta", "ab",
})

_SHELL_SEPARATORS = frozenset({";", "|", "&", "\n"})


# Shell splitting / tokenization

def split_shell_segments(command: str) -> list[str]:
    """Split a command string by shell separators (``;``, ``|``, ``&&``, ``||``).

    Respects single and double quoting so that separators inside quoted
    strings are not treated as boundaries.
    """
    segments: list[str] = []
    quote: str | None = None
    current: list[str] = []
    idx = 0
    length = len(command)

    while idx < length:
        ch = command[idx]

        # Inside a quoted region - consume until matching close quote.
        if quote is not None:
            current.append(ch)
            if ch == quote:
                quote = None
            idx += 1
            continue

        # Opening quote.
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            idx += 1
            continue

        # Shell separator.
        if ch in _SHELL_SEPARATORS:
            trimmed = "".join(current).strip()
            if trimmed:
                segments.append(trimmed)
            current = []
            # Consume doubled separators (``&&``, ``||``).
            if (ch in ("&", "|")) and idx + 1 < length and command[idx + 1] == ch:
                idx += 2
            else:
                idx += 1
            continue

        current.append(ch)
        idx += 1

    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments


def split_shell_tokens(segment: str) -> list[str]:
    """Tokenize a single shell segment by whitespace, respecting quotes.

    Quotes are stripped from the resulting tokens (content only).
    """
    tokens: list[str] = []
    quote: str | None = None
    current: list[str] = []

    for ch in segment:
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
            continue

        if ch in ("'", '"'):
            quote = ch
            continue

        if ch.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue

        current.append(ch)

    if current:
        tokens.append("".join(current))
    return tokens


def strip_leading_env_assignments(tokens: list[str]) -> list[str]:
    """Remove leading ``VAR=value`` tokens from a token list.

    Stops at the first token that does not look like an environment
    variable assignment (i.e. does not contain ``=`` or starts with ``-``).
    """
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        eq_at = token.find("=")
        is_assignment = eq_at > 0 and not token.startswith("-")
        if not is_assignment:
            break
        idx += 1
    return tokens[idx:]


class CommandSegment(TypedDict):
    """Parsed representation of one shell segment."""
    segment: str
    tokens: list[str]
    executable: str


def parse_command_segments(command: str) -> list[CommandSegment]:
    """Fully parse a command string into classified segments.

    Each returned dict contains:
    - ``segment`` - the raw segment text
    - ``tokens`` - lowercase tokens with env assignments stripped
    - ``executable`` - the first token (i.e. the command name)
    """
    result: list[CommandSegment] = []
    for segment in split_shell_segments(command):
        tokens = strip_leading_env_assignments(split_shell_tokens(segment))
        if tokens:
            lower_tokens = [t.lower() for t in tokens]
            result.append(CommandSegment(
                segment=segment,
                tokens=lower_tokens,
                executable=lower_tokens[0],
            ))
    return result


# Internal helper - cached segment parsing for classification functions

@lru_cache(maxsize=256)
def _parsed_tokens(command: str) -> tuple[tuple[str, ...], ...]:
    """Return a hashable tuple-of-tuples of lowercase tokens per segment."""
    result: list[tuple[str, ...]] = []
    for seg in split_shell_segments(command):
        tokens = strip_leading_env_assignments(split_shell_tokens(seg))
        if tokens:
            result.append(tuple(t.lower() for t in tokens))
    return tuple(result)


# Command classification

def is_verification_command(command: str) -> bool:
    """Detect test / build / check / lint commands."""
    for tokens in _parsed_tokens(command):
        first = tokens[0] if len(tokens) > 0 else ""
        second = tokens[1] if len(tokens) > 1 else ""
        third = tokens[2] if len(tokens) > 2 else ""

        if first in VERIFICATION_SINGLE_WORD_COMMANDS:
            return True
        if first == "go" and second in ("test", "build"):
            return True
        if first == "cargo" and second in ("test", "build", "check", "clippy"):
            return True
        if first == "npm" and second in ("test", "build"):
            return True
        if first == "npm" and second == "run" and third in ("test", "build"):
            return True
        if first in ("pnpm", "yarn") and second in ("test", "build"):
            return True
        if first == "make" and second in ("test", "build", "check"):
            return True
        # Python-specific linters / type checkers
        if first in ("mypy", "ruff", "flake8", "pylint", "eslint", "tsc"):
            return True
    return False


def is_load_test_command(command: str) -> bool:
    """Detect load / stress test tools (k6, locust, jmeter, ab, wrk, etc.)."""
    for tokens in _parsed_tokens(command):
        first = tokens[0] if len(tokens) > 0 else ""
        second = tokens[1] if len(tokens) > 1 else ""

        if first in LOAD_TEST_COMMAND_HEADS:
            return True
        # Extra tools not in the TS set but mentioned in spec.
        if first in ("locust", "jmeter", "artillery", "gatling"):
            return True
        if first == "go" and second == "run" and any("loadtest" in t for t in tokens[2:]):
            return True
        if first in ("npm", "pnpm", "yarn") and "loadtest" in tokens:
            return True
    return False


def is_service_start_command(command: str) -> bool:
    """Detect service startup commands (go run, npm run dev, uvicorn, etc.)."""
    for tokens in _parsed_tokens(command):
        first = tokens[0] if len(tokens) > 0 else ""
        second = tokens[1] if len(tokens) > 1 else ""
        third = tokens[2] if len(tokens) > 2 else ""

        if "--help" in tokens or "-h" in tokens:
            continue

        if first == "go" and second == "run":
            return True
        if first == "npm" and second == "run" and third in ("dev", "start", "serve", "preview"):
            return True
        if first in ("pnpm", "yarn") and second in ("dev", "start", "serve", "preview"):
            return True

        # Docker compose up (foreground only)
        if first == "docker" and second == "compose" and third == "up":
            if "-d" not in tokens and "--detach" not in tokens:
                return True
        if first == "docker-compose" and second == "up":
            if "-d" not in tokens and "--detach" not in tokens:
                return True
        if first == "docker" and second == "run":
            return True

        # Python servers
        if first.startswith("python") and second == "-m":
            if third in ("uvicorn", "http.server", "flask"):
                return True
        if first in ("uvicorn", "gunicorn"):
            return True

        # Rails / Flask
        if first == "rails" and second == "server":
            return True
        if first == "flask" and second == "run":
            return True
    return False


def is_service_runtime_probe_command(command: str) -> bool:
    """Detect runtime probes (curl/wget localhost, health checks)."""
    lower = command.lower()
    has_health_path = bool(re.search(r"/(health|healthz|ready|readyz|live|livez)\b", lower))
    has_localhost = bool(re.search(r"127\.0\.0\.1|localhost|0\.0\.0\.0", lower))

    for tokens in _parsed_tokens(command):
        first = tokens[0] if len(tokens) > 0 else ""
        if first in ("curl", "wget") and (has_health_path or has_localhost):
            return True
        if first in ("nc", "netcat") and ("-z" in tokens or "-vz" in tokens):
            return True

    # Node inline probe
    if re.search(r"\bnode\b", lower) and re.search(r"\bhttp\.(get|request)\b|\bfetch\(", lower):
        if has_health_path or has_localhost:
            return True

    # Python inline probe
    if re.search(r"\bpython\d*(\.\d+)?\b", lower):
        python_http = (
            re.search(r"\burllib\.request\.urlopen\s*\(", lower)
            or re.search(r"\bhttp\.client\.", lower)
            or re.search(r"\brequests\.(get|head)\s*\(", lower)
            or re.search(r"\bsocket\.create_connection\s*\(", lower)
        )
        if python_http and (has_health_path or has_localhost):
            return True

    return False


def is_service_cleanup_command(command: str) -> bool:
    """Detect cleanup commands (docker compose down, kill, pkill, lsof -ti)."""
    for tokens in _parsed_tokens(command):
        first = tokens[0] if len(tokens) > 0 else ""
        second = tokens[1] if len(tokens) > 1 else ""
        third = tokens[2] if len(tokens) > 2 else ""

        if first in ("kill", "pkill", "killall"):
            return True
        if first == "docker" and second == "compose" and third in ("down", "stop"):
            return True
        if first == "docker-compose" and second in ("down", "stop"):
            return True
        if first == "npm" and second == "run" and third == "stop":
            return True
        if first in ("pnpm", "yarn") and second == "stop":
            return True
        # lsof -ti :PORT | xargs kill  - the lsof part is in a pipe segment
        if first == "lsof" and "-ti" in tokens:
            return True
    return False
