"""Command analysis engine for RUNE.

Ported 1:1 from src/safety/analyzer.ts - normalization, risk classification,
danger pattern database (40+ patterns), command parser & tokenizer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Types

FindingType = Literal["critical", "high", "medium", "low", "info"]


@dataclass(slots=True)
class AnalysisFinding:
    type: FindingType
    category: str
    description: str
    evidence: str


@dataclass(slots=True)
class ParsedCommand:
    executable: str
    args: list[str]
    has_pipeline: bool
    has_redirection: bool
    has_substitution: bool
    has_background_job: bool
    chained_commands: list[str]


@dataclass(slots=True)
class AnalysisResult:
    safe: bool
    risk_score: int  # 0-100
    findings: list[AnalysisFinding]
    command: str
    parsed: ParsedCommand


# Danger Pattern Database (ported 1:1 from TS)

@dataclass(slots=True, frozen=True)
class DangerPattern:
    name: str
    category: str
    pattern: re.Pattern[str]
    risk_level: FindingType
    description: str


# All patterns compiled once at module load
DANGER_PATTERNS: list[DangerPattern] = [
    # System destruction
    DangerPattern("force-delete-all", "destruction",
                  re.compile(r"rm\s+(-rf?|--force|--recursive)\s*\*"),
                  "high", "Force delete with wildcard"),
    DangerPattern("disk-format", "destruction",
                  re.compile(r"mkfs\.|mkswap|fdisk|parted"),
                  "critical", "Disk formatting command"),
    DangerPattern("direct-disk-write", "destruction",
                  re.compile(r"dd\s+.*of=/dev"),
                  "critical", "Direct disk write"),

    # Remote code execution
    DangerPattern("curl-pipe-shell", "rce",
                  re.compile(r"curl[^|]*\|\s*(ba)?sh"),
                  "critical", "Remote code execution via curl"),
    DangerPattern("wget-pipe-shell", "rce",
                  re.compile(r"wget[^|]*\|\s*(ba)?sh"),
                  "critical", "Remote code execution via wget"),
    DangerPattern("pipe-to-shell", "rce",
                  re.compile(r"\|\s*(ba)?sh\s*$"),
                  "high", "Piping to shell"),
    DangerPattern("eval-usage", "rce",
                  re.compile(r"\beval\s+"),
                  "high", "Eval command usage"),

    # Encoding bypass
    DangerPattern("base64-decode-execute", "obfuscation",
                  re.compile(r"base64\s+(-d|--decode)[^|]*\|\s*(ba)?sh"),
                  "critical", "Base64 decode and execute"),
    DangerPattern("hex-decode", "obfuscation",
                  re.compile(r"xxd\s+-r[^|]*\|\s*(ba)?sh"),
                  "critical", "Hex decode and execute"),
    DangerPattern("python-exec", "obfuscation",
                  re.compile(r"python[23]?\s+-c\s+[\"'].*exec"),
                  "high", "Python exec usage"),

    # Fork bomb
    DangerPattern("fork-bomb", "dos",
                  re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;|\.\/:\(\)"),
                  "critical", "Fork bomb detected"),

    # Permission manipulation
    DangerPattern("dangerous-chmod", "permissions",
                  re.compile(r"chmod\s+(-R\s+)?777\s+/"),
                  "critical", "Dangerous permission change"),
    DangerPattern("setuid", "permissions",
                  re.compile(r"chmod\s+[0-7]*[4-7][0-7]{2}"),
                  "high", "Setuid/setgid bit manipulation"),
    DangerPattern("chown-root", "permissions",
                  re.compile(r"chown\s+(-R\s+)?root"),
                  "high", "Changing ownership to root"),

    # System commands
    DangerPattern("shutdown", "system",
                  re.compile(r"\b(shutdown|reboot|halt|init\s+0)\b"),
                  "critical", "System shutdown/reboot"),

    # Environment manipulation
    DangerPattern("path-hijack", "environment",
                  re.compile(r"export\s+PATH=.*:"),
                  "medium", "PATH manipulation"),
    DangerPattern("ld-preload", "environment",
                  re.compile(r"LD_PRELOAD|LD_LIBRARY_PATH"),
                  "high", "Library path manipulation"),

    # Network exfiltration
    DangerPattern("reverse-shell", "network",
                  re.compile(r"nc\s+-[elp]|ncat.*-e|/dev/(tcp|udp)/"),
                  "critical", "Potential reverse shell"),
    DangerPattern("data-exfil", "network",
                  re.compile(r"curl\s+.*-d\s+.*@|wget\s+.*--post-file"),
                  "high", "Potential data exfiltration"),

    # Argument injection
    DangerPattern("git-arg-injection", "injection",
                  re.compile(r"git\s+(pull|clone|fetch)\s+.*--upload-pack"),
                  "high", "Git argument injection"),
    DangerPattern("tar-extraction", "injection",
                  re.compile(r"tar\s+.*--to-command"),
                  "high", "Tar extraction command injection"),

    # Encoding/escape bypass
    DangerPattern("ansi-c-quoting", "obfuscation",
                  re.compile(r"\$'[^']*\\x[0-9a-fA-F]{2}"),
                  "high", "ANSI-C quoting with hex escape (potential command obfuscation)"),
    DangerPattern("octal-escape", "obfuscation",
                  re.compile(r"\$'[^']*\\[0-7]{1,3}"),
                  "high", "ANSI-C quoting with octal escape (potential command obfuscation)"),
    DangerPattern("unicode-escape", "obfuscation",
                  re.compile(r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}"),
                  "medium", "Unicode escape sequence detected"),
    DangerPattern("env-var-command", "obfuscation",
                  re.compile(r"\bcmd=.*;\s*\$cmd\b|\$\{[A-Z_]+\}\s+"),
                  "high", "Command execution via variable expansion"),

    # Language runtime RCE
    DangerPattern("python-system", "rce",
                  re.compile(r"python[23]?\s+-c\s+[\"'].*(?:os\.system|subprocess|os\.popen)"),
                  "high", "Python system command execution"),
    DangerPattern("perl-exec", "rce",
                  re.compile(r"perl\s+-e\s+[\"'].*(?:system|exec|`)"),
                  "high", "Perl system command execution"),
    DangerPattern("ruby-exec", "rce",
                  re.compile(r"ruby\s+-e\s+[\"'].*(?:system|exec|`)"),
                  "high", "Ruby system command execution"),
    DangerPattern("node-exec", "rce",
                  re.compile(r"node\s+-e\s+[\"'].*(?:child_process|execSync|exec\()"),
                  "high", "Node.js child process execution"),

    # Destructive find
    DangerPattern("find-delete", "destruction",
                  re.compile(r"find\s+[/~]\S*\s+.*-delete"),
                  "critical", "Recursive find with delete from root or home"),
    DangerPattern("find-exec-rm", "destruction",
                  re.compile(r"find\s+[/~]\S*\s+.*-exec\s+rm"),
                  "critical", "Recursive find with exec rm"),
    DangerPattern("xargs-rm", "destruction",
                  re.compile(r"\|\s*xargs\s+.*rm"),
                  "high", "Piped xargs with rm"),

    # Symlink attacks
    DangerPattern("symlink-attack", "injection",
                  re.compile(r"ln\s+-[sf]+\s+/(etc|sys|proc|bin|sbin|usr)"),
                  "high", "Symlink to system directory (potential traversal attack)"),

    # Crontab manipulation
    DangerPattern("crontab-modify", "persistence",
                  re.compile(r"crontab\s+-[elr]|echo.*>.*crontab"),
                  "high", "Crontab modification"),

    # Docker/Container destructive
    DangerPattern("docker-prune-all", "destruction",
                  re.compile(r"docker\s+system\s+prune\s+(-a|--all|-f|--force)"),
                  "high", "Docker system prune"),
    DangerPattern("docker-rm-all-containers", "destruction",
                  re.compile(r"docker\s+rm\s+(-f\s+)?\$\(docker\s+ps"),
                  "high", "Force remove all Docker containers"),
    DangerPattern("docker-rmi-all", "destruction",
                  re.compile(r"docker\s+rmi\s+(-f\s+)?\$\(docker\s+images"),
                  "high", "Remove all Docker images"),
    DangerPattern("docker-volume-rm-all", "destruction",
                  re.compile(r"docker\s+volume\s+(rm|prune)\s+(-f|--force|\$\()"),
                  "high", "Remove Docker volumes (potential data loss)"),
]

# Risk multipliers for scoring
_RISK_MULTIPLIER: dict[str, int] = {
    "critical": 50,
    "high": 30,
    "medium": 15,
    "low": 5,
}

# Critical delete paths for rm -rf classification
_CRITICAL_DELETE_PATHS = [
    "/etc", "/bin", "/sbin", "/usr", "/lib", "/lib64",
    "/System", "/Library", "/var/log",
    # macOS symlink targets
    "/private/etc", "/private/var/log",
]

_CRITICAL_DOTFILES = [".ssh", ".aws", ".config", ".gnupg"]


# Command Tokenizer (shell-aware)

def tokenize(command: str) -> list[str]:
    """Tokenize a shell command respecting quotes and escapes."""
    tokens: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            current.append(char)
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if char == " " and not in_single_quote and not in_double_quote:
            if current:
                tokens.append("".join(current))
                current.clear()
            continue

        current.append(char)

    if current:
        tokens.append("".join(current))

    return tokens


# Command Parser

_PIPELINE_RE = re.compile(r"\|(?!\|)")
_REDIRECTION_RE = re.compile(r"[<>]")
_SUBSTITUTION_RE = re.compile(r"\$\(|`")
_BACKGROUND_RE = re.compile(r"&\s*$")
_CHAIN_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")


def parse_command(command: str) -> ParsedCommand:
    """Parse a shell command into its structural components."""
    trimmed = command.strip()

    has_pipeline = bool(_PIPELINE_RE.search(trimmed))
    has_redirection = bool(_REDIRECTION_RE.search(trimmed))
    has_substitution = bool(_SUBSTITUTION_RE.search(trimmed))
    has_background_job = bool(_BACKGROUND_RE.search(trimmed))
    chained_commands = [c for c in _CHAIN_SPLIT_RE.split(trimmed) if c]

    first_command = chained_commands[0] if chained_commands else trimmed
    tokens = tokenize(first_command)
    executable = tokens[0] if tokens else ""
    args = tokens[1:] if len(tokens) > 1 else []

    return ParsedCommand(
        executable=executable,
        args=args,
        has_pipeline=has_pipeline,
        has_redirection=has_redirection,
        has_substitution=has_substitution,
        has_background_job=has_background_job,
        chained_commands=chained_commands,
    )


# rm -rf Risk Classification

_RM_RF_RE = re.compile(
    r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r?|--recursive)\s+(\S+)"
)


def classify_rm_rf_risk(command: str) -> Literal["critical", "high"] | None:
    """Classify the risk of an rm -rf command based on its target path."""
    match = _RM_RF_RE.search(command)
    if not match:
        return None

    raw_path = re.sub(r"""["';)]+$""", "", match.group(2))
    home = os.environ.get("HOME", "/home")

    expanded = raw_path.replace("~", home, 1) if raw_path.startswith("~") else raw_path
    expanded = expanded.replace("$HOME", home).replace("${HOME}", home)
    resolved = str(Path(expanded).resolve())

    if resolved == "/" or resolved == home:
        return "critical"

    for sys_path in _CRITICAL_DELETE_PATHS:
        if resolved == sys_path or resolved.startswith(sys_path + "/"):
            return "critical"

    for dot in _CRITICAL_DOTFILES:
        dot_path = str(Path(home) / dot)
        if resolved == dot_path or resolved.startswith(dot_path + "/"):
            return "critical"

    return "high"


# Command Normalization (all encoding bypass handling)

_ANSI_C_RE = re.compile(r"\$'([^']*)'")
_HEX_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_OCTAL_ESCAPE_RE = re.compile(r"\\([0-7]{1,3})")
_WHITESPACE_RE = re.compile(r"\s+")
_LINE_CONTINUATION_RE = re.compile(r"\\$")


def normalize_command(command: str) -> str:
    """Normalize a command by decoding ANSI-C escapes, expanding env vars, etc."""
    normalized = command

    # ANSI-C escape interpretation: $'\x72\x6d' → rm
    def _decode_ansi_c(m: re.Match[str]) -> str:
        content = m.group(1)
        # Hex escapes: \x72 → r
        content = _HEX_ESCAPE_RE.sub(
            lambda hm: chr(int(hm.group(1), 16)), content
        )
        # Octal escapes: \162 → r
        content = _OCTAL_ESCAPE_RE.sub(
            lambda om: chr(int(om.group(1), 8)), content
        )
        return content

    normalized = _ANSI_C_RE.sub(_decode_ansi_c, normalized)

    # Environment variable expansion
    home = os.environ.get("HOME", "/home")
    normalized = normalized.replace("$HOME", home).replace("${HOME}", home)
    normalized = re.sub(r"~(?=/|$)", home, normalized)

    # Normalize whitespace
    normalized = _WHITESPACE_RE.sub(" ", normalized)

    # Remove line continuation escapes
    normalized = _LINE_CONTINUATION_RE.sub("", normalized)

    # IFS variable substitution
    normalized = normalized.replace("$IFS", " ")

    return normalized.strip()


# Core Analysis Function

def analyze_command(command: str) -> AnalysisResult:
    """Analyze a shell command for security risks.

    Returns an AnalysisResult with risk score (0-100), findings, and parsed structure.
    """
    findings: list[AnalysisFinding] = []
    risk_score = 0

    parsed = parse_command(command)

    # rm -rf path-based classification
    rm_rf_risk = classify_rm_rf_risk(command)
    if rm_rf_risk is not None:
        rm_rf_score = 50 if rm_rf_risk == "critical" else 30
        findings.append(AnalysisFinding(
            type=rm_rf_risk,
            category="destruction",
            description=(
                "Recursive deletion targeting system or critical path"
                if rm_rf_risk == "critical"
                else "Recursive deletion with absolute path (requires approval)"
            ),
            evidence=(_RM_RF_RE.search(command) or _placeholder_match()).group(0),
        ))
        risk_score += rm_rf_score

    # Pattern checking
    for dp in DANGER_PATTERNS:
        m = dp.pattern.search(command)
        if m:
            findings.append(AnalysisFinding(
                type=dp.risk_level,
                category=dp.category,
                description=dp.description,
                evidence=m.group(0),
            ))
            risk_score += _RISK_MULTIPLIER.get(dp.risk_level, 0)

    # Structural risk assessment
    if parsed.has_pipeline:
        findings.append(AnalysisFinding(
            type="info", category="structure",
            description="Command contains pipeline", evidence="|",
        ))
        risk_score += 5

    if parsed.has_substitution:
        findings.append(AnalysisFinding(
            type="medium", category="structure",
            description="Command contains command substitution",
            evidence="$() or ``",
        ))
        risk_score += 15

    if len(parsed.chained_commands) > 3:
        findings.append(AnalysisFinding(
            type="low", category="complexity",
            description="Complex command chain",
            evidence=f"{len(parsed.chained_commands)} chained commands",
        ))
        risk_score += 10

    # Clamp 0-100
    risk_score = min(100, risk_score)

    safe = risk_score < 50 and not any(f.type == "critical" for f in findings)

    return AnalysisResult(
        safe=safe,
        risk_score=risk_score,
        findings=findings,
        command=command,
        parsed=parsed,
    )


class _PlaceholderMatch:
    """Fallback for regex match .group(0)."""
    def group(self, _: int) -> str:
        return ""


def _placeholder_match() -> _PlaceholderMatch:
    return _PlaceholderMatch()
