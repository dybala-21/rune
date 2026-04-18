"""Guardian - real-time safety validation for RUNE.

Ported 1:1 from src/safety/guardian.ts - risk scoring, path validation,
dangerous bash pattern detection, dual-pass analysis (original + normalized).
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rune.safety.analyzer import analyze_command, classify_rm_rf_risk, normalize_command

# Types

RiskLevel = Literal["safe", "low", "medium", "high", "critical"]


@dataclass(slots=True)
class ValidationResult:
    allowed: bool
    risk_level: RiskLevel
    reason: str = ""
    suggestions: list[str] = field(default_factory=list)
    requires_approval: bool = False


# Dangerous Bash Patterns (Guardian-level, ported 1:1)

@dataclass(slots=True, frozen=True)
class _DangerRule:
    pattern: re.Pattern[str]
    risk: RiskLevel
    reason: str


_DANGEROUS_BASH_PATTERNS: list[_DangerRule] = [
    _DangerRule(re.compile(r"rm\s+-rf?\s*$"), "high", "rm -rf without target"),
    _DangerRule(re.compile(r"mkfs\."), "critical", "Disk formatting command"),
    _DangerRule(re.compile(r"dd\s+.*of=/dev"), "critical", "Direct disk write"),
    # Permissions
    _DangerRule(re.compile(r"chmod\s+(-R\s+)?777\s+/"), "critical", "Dangerous permission change"),
    _DangerRule(re.compile(r"chown\s+-R?\s+root"), "high", "Ownership change to root"),
    # Network RCE
    _DangerRule(re.compile(r"curl.*\|\s*bash"), "critical", "Remote code execution via curl"),
    _DangerRule(re.compile(r"wget.*\|\s*bash"), "critical", "Remote code execution via wget"),
    _DangerRule(re.compile(r"\|\s*sh\s*$"), "high", "Piping to shell"),
    # Fork bomb
    _DangerRule(re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"), "critical", "Fork bomb detected"),
    # System
    _DangerRule(re.compile(r"shutdown"), "critical", "System shutdown"),
    _DangerRule(re.compile(r"reboot"), "critical", "System reboot"),
    _DangerRule(re.compile(r"init\s+0"), "critical", "System halt"),
    # Password/auth
    _DangerRule(re.compile(r"passwd"), "high", "Password change attempt"),
    _DangerRule(re.compile(r"sudo\s+-S"), "high", "Sudo with password from stdin"),
    # Environment
    _DangerRule(re.compile(r"export\s+PATH="), "medium", "PATH modification"),
    _DangerRule(re.compile(r"export\s+LD_"), "high", "Library path modification"),
    # Docker
    _DangerRule(re.compile(r"docker\s+system\s+prune\s+(-a|--all)"), "high",
                "Docker system prune all"),
    _DangerRule(re.compile(r"docker\s+rm\s+(-f\s+)?\$\(docker\s+ps"), "high",
                "Remove all Docker containers"),
    _DangerRule(re.compile(r"docker\s+rmi\s+(-f\s+)?\$\(docker\s+images"), "high",
                "Remove all Docker images"),
    # Git
    _DangerRule(re.compile(r"git\s+push\s+.*--force(?!-with-lease)"), "high",
                "Git force push can overwrite remote history"),
    _DangerRule(re.compile(r"git\s+push\s+.*--force-with-lease"), "medium",
                "Git force push with lease (safer but still risky)"),
    _DangerRule(re.compile(r"git\s+reset\s+--hard"), "high",
                "Git hard reset discards uncommitted changes"),
    _DangerRule(re.compile(r"git\s+branch\s+-D"), "high", "Force delete git branch"),
    _DangerRule(re.compile(r"git\s+clean\s+-[fd]+"), "medium",
                "Git clean removes untracked files/directories"),
    _DangerRule(re.compile(r"git\s+checkout\s+--\s+\."), "medium",
                "Discard all uncommitted changes"),
    # SQL
    _DangerRule(re.compile(r"\bDROP\s+(TABLE|DATABASE|INDEX|SCHEMA)\b", re.I), "high",
                "SQL DROP statement — destructive data operation"),
    _DangerRule(re.compile(r"\bTRUNCATE\s+TABLE\b", re.I), "high",
                "SQL TRUNCATE — removes all table data"),
    _DangerRule(re.compile(r"\bDELETE\s+FROM\b", re.I), "medium",
                "SQL DELETE — bulk data removal"),
    # Network access
    _DangerRule(re.compile(r"\bcurl\s+"), "medium", "Network request via curl"),
    _DangerRule(re.compile(r"\bwget\s+"), "medium", "Network request via wget"),
    _DangerRule(re.compile(r"\bssh\s+"), "medium", "SSH connection"),
    _DangerRule(re.compile(r"\bscp\s+"), "medium", "SCP file transfer"),
    _DangerRule(re.compile(r"\bsftp\s+"), "medium", "SFTP file transfer"),
    # nc/netcat
    _DangerRule(re.compile(r"\bnc\s+"), "medium", "Netcat connection"),
    _DangerRule(re.compile(r"\bnetcat\s+"), "medium", "Netcat connection"),
    # nmap
    _DangerRule(re.compile(r"\bnmap\s+"), "medium", "Network scanning"),
    # Process management
    _DangerRule(re.compile(r"\bkill\s+-9\b"), "high", "Force kill process (SIGKILL)"),
    _DangerRule(re.compile(r"\bkillall\s+"), "high", "Kill all processes by name"),
    _DangerRule(re.compile(r"\bpkill\s+"), "high", "Kill processes by pattern"),
    # Bash file read on sensitive paths
    _DangerRule(re.compile(r"\b(cat|head|tail|less|more)\s+.*(\.ssh|\.aws|\.gnupg|/etc/shadow|/etc/sudoers)"), "high",
                "Reading sensitive file via bash command"),
    # General file read via bash (lower priority than sensitive path rule above)
    _DangerRule(re.compile(r"\b(cat|head|tail|less|more)\s+\S"), "low",
                "File read via bash — prefer file.read capability"),
    # Inline Python file read
    _DangerRule(re.compile(r"python[23]?\s+-c\s+.*open\s*\("), "medium",
                "Inline Python file read — prefer file.read capability"),
    # Config redirect bypass
    _DangerRule(re.compile(r">\s*~?/?\.rune/(config\.ya?ml|\.env)\b"), "high",
                "Redirect to RUNE config file — use file.write with approval instead"),
]

# Protected Paths

PROTECTED_PATHS = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/hosts", "/etc/ssh",
    "~/.ssh", "~/.aws", "~/.config",
    "/System", "/Library",
    "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/opt", "/var",
    "/private/etc", "/private/var",
]

# Paths that must never be targets of write/delete operations.
# Checked both as exact match AND as ancestor (i.e. deleting "/" is blocked
# because it is a parent of every protected path).
_CRITICAL_ROOT_PATHS = [
    "/",
    "~",  # expanded at runtime
]

# Minimum path depth for write/delete operations.
# Depth 1 = "/Users", "/tmp", etc.  Depth 2 = "/Users/foo".
# Anything with fewer parts than this is too close to the filesystem root.
_MIN_WRITABLE_DEPTH = 3

CONFIG_APPROVAL_PATHS = [
    "~/.rune/config.yaml",
    "~/.rune/config.yml",
    "~/.rune/.env",
]

READ_BLOCKED_PATHS = [
    "~/.ssh", "~/.aws", "~/.npmrc", "~/.netrc", "~/.gnupg",
    "/etc/shadow", "/etc/sudoers",
]


# Helper: standalone path match

_PATH_BOUNDARY = re.compile(r"""[\s"'`(=;|&<>]""")


def _is_standalone_path_match(input_str: str, protected_path: str) -> bool:
    """Check if *protected_path* appears as a standalone token in *input_str*."""
    search_from = 0
    while search_from < len(input_str):
        idx = input_str.find(protected_path, search_from)
        if idx == -1:
            return False
        if idx == 0:
            return True
        if _PATH_BOUNDARY.match(input_str[idx - 1]):
            return True
        search_from = idx + 1
    return False


# Risk conversion

_RISK_NUMERIC: dict[RiskLevel, int] = {
    "safe": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}


def risk_to_number(risk: RiskLevel) -> int:
    return _RISK_NUMERIC.get(risk, 0)


# Guardian

class Guardian:
    """Real-time safety validator for bash commands and file paths."""

    def __init__(self) -> None:
        self._home = os.environ.get("HOME", str(Path.home()))
        self._approval_callback: Callable[[str], Awaitable[bool]] | None = None

    def _expand(self, p: str) -> str:
        return p.replace("~", self._home, 1) if p.startswith("~") else p

    def validate(self, command: str, _context: str | None = None) -> ValidationResult:
        """Validate a bash command for safety risks.

        Uses dual-pass analysis: original command + normalized (decoded) command.
        The higher risk wins.
        """
        normalized = normalize_command(command)
        analysis = analyze_command(command)
        normalized_analysis = (
            analyze_command(normalized) if normalized != command else analysis
        )
        effective = (
            normalized_analysis
            if normalized_analysis.risk_score > analysis.risk_score
            else analysis
        )

        # rm -rf path-based classification (direct, before pattern matching)
        rm_rf_risk = classify_rm_rf_risk(command)
        if rm_rf_risk is None and normalized != command:
            rm_rf_risk = classify_rm_rf_risk(normalized)
        if rm_rf_risk == "critical":
            return ValidationResult(
                allowed=False,
                risk_level="critical",
                reason="Recursive deletion targeting system or critical path",
            )

        # Critical finding → hard block
        critical = next((f for f in effective.findings if f.type == "critical"), None)
        if critical:
            return ValidationResult(
                allowed=False,
                risk_level="critical",
                reason=critical.description,
            )

        # High risk score (50+) → requires approval
        if effective.risk_score >= 50:
            high_findings = [f for f in effective.findings if f.type == "high"]
            return ValidationResult(
                allowed=True,
                risk_level="high",
                reason=", ".join(f.description for f in high_findings) or "High risk score",
                requires_approval=True,
            )

        # Evaluate Guardian-specific rules (pattern checks)
        worst_result: ValidationResult | None = None
        worst_risk_num = -1

        for rule in _DANGEROUS_BASH_PATTERNS:
            for cmd in (command, normalized) if normalized != command else (command,):
                if rule.pattern.search(cmd):
                    risk_num = risk_to_number(rule.risk)
                    if risk_num > worst_risk_num:
                        worst_risk_num = risk_num
                        if rule.risk == "critical":
                            worst_result = ValidationResult(
                                allowed=False,
                                risk_level="critical",
                                reason=rule.reason,
                            )
                        elif rule.risk == "high":
                            worst_result = ValidationResult(
                                allowed=True,
                                risk_level="high",
                                reason=rule.reason,
                                requires_approval=True,
                            )
                        else:
                            worst_result = ValidationResult(
                                allowed=True,
                                risk_level=rule.risk,
                                reason=rule.reason,
                            )
                    break  # one match per rule is enough

        if worst_result is not None:
            return worst_result

        # Bash command referencing protected/blocked paths
        for pp in PROTECTED_PATHS + READ_BLOCKED_PATHS:
            expanded_pp = self._expand(pp)
            for cmd in (command, normalized) if normalized != command else (command,):
                if _is_standalone_path_match(cmd, expanded_pp):
                    return ValidationResult(
                        allowed=False,
                        risk_level="high",
                        reason=f"Bash command references sensitive path: {pp}",
                    )

        # Medium risk score (30-49)
        if effective.risk_score >= 30:
            return ValidationResult(
                allowed=True,
                risk_level="medium",
                reason=", ".join(f.description for f in effective.findings),
            )

        return ValidationResult(
            allowed=True,
            risk_level="low" if effective.risk_score >= 15 else "safe",
        )

    def set_approval_callback(self, callback: Callable[[str], Awaitable[bool]]) -> None:
        """Register a callback for interactive approval workflows."""
        self._approval_callback = callback

    async def execute_with_approval(self, action: str, executor: Callable[[], Awaitable[None]]) -> dict[str, Any]:
        """Validate, prompt for approval if needed, then execute."""
        result = self.validate(action)
        if not result.allowed:
            return {"executed": False, "reason": result.reason}
        if result.requires_approval:
            if self._approval_callback is None:
                return {"executed": False, "reason": "No approval callback registered"}
            approved = await self._approval_callback(f"{result.reason}: {action}")
            if not approved:
                self.log_audit("denied", "bash_execute", {"command": action})
                return {"executed": False, "reason": "User denied approval"}
            self.log_audit("approved", "bash_execute", {"command": action})
        await executor()
        return {"executed": True}

    def add_rule(self, pattern: str, risk: RiskLevel, reason: str) -> None:
        """Dynamically add a danger rule."""
        _DANGEROUS_BASH_PATTERNS.append(_DangerRule(re.compile(pattern), risk, reason))

    # Parameter names that should be redacted in audit logs.
    _SENSITIVE_PARAM_RE = re.compile(
        r"(key|token|secret|password|passwd|credential|auth|bearer)",
        re.IGNORECASE,
    )

    def log_audit(self, action: str, capability: str, params: dict[str, Any]) -> None:
        """Log a safety audit event to ~/.rune/audit.jsonl."""
        import time

        from rune.utils.fast_serde import json_encode
        from rune.utils.paths import rune_home

        audit_file = rune_home() / "audit.jsonl"

        # Redact values whose parameter names suggest sensitive content.
        safe_params: dict[str, str] = {}
        for k, v in params.items():
            if self._SENSITIVE_PARAM_RE.search(k):
                safe_params[k] = "***REDACTED***"
            else:
                safe_params[k] = str(v)[:200]

        entry = {
            "timestamp": time.time(),
            "action": action,
            "capability": capability,
            "params": safe_params,
        }
        try:
            with open(audit_file, "a") as f:
                f.write(json_encode(entry) + "\n")
            # Restrict file permissions (owner read/write only)
            audit_file.chmod(0o600)
        except OSError:
            pass

    def is_command_safe(self, command: str) -> bool:
        """Quick boolean check -- True if command is safe to execute without approval."""
        result = self.validate(command)
        return result.allowed and not result.requires_approval

    def analyze_command(self, command: str) -> Any:
        """Analyze a command and return findings. Delegates to analyzer module."""
        return analyze_command(command)

    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate a write path against protected paths.

        Checks:
        1. Empty / blank path → reject
        2. Critical root paths (``/``, ``~``) → reject
        3. Minimum depth check (must be ≥ _MIN_WRITABLE_DEPTH parts)
        4. Protected-path containment - both directions:
           a. requested path IS or is INSIDE a protected path → reject
           b. requested path is a PARENT of a protected path → reject
        5. Config-file approval gate
        """
        # -- empty path guard --------------------------------------------------
        if not file_path or not file_path.strip():
            return ValidationResult(
                allowed=False,
                risk_level="critical",
                reason="Empty file path",
            )

        expanded = file_path.replace("~", self._home, 1) if file_path.startswith("~") else file_path
        normalized = str(Path(expanded).resolve())

        # Resolve symlinks
        real_path = normalized
        try:
            p = Path(normalized)
            if p.exists():
                real_path = str(p.resolve(strict=True))
        except OSError:
            pass

        # -- critical root paths -----------------------------------------------
        for crp in _CRITICAL_ROOT_PATHS:
            expanded_crp = self._expand(crp)
            norm_crp = str(Path(expanded_crp).resolve())
            if normalized == norm_crp or real_path == norm_crp:
                return ValidationResult(
                    allowed=False,
                    risk_level="critical",
                    reason=f"Write/delete to critical root path blocked: {crp}",
                )

        # minimum depth check
        if len(Path(normalized).parts) < _MIN_WRITABLE_DEPTH:
            return ValidationResult(
                allowed=False,
                risk_level="critical",
                reason=f"Path too close to filesystem root: {normalized}",
            )

        # protected path containment (both directions)
        for pp in PROTECTED_PATHS:
            expanded_pp = self._expand(pp)
            norm_pp = str(Path(expanded_pp).resolve())

            # (a) requested path is inside (or equal to) a protected path
            if (
                normalized == norm_pp
                or normalized.startswith(norm_pp + "/")
                or real_path == norm_pp
                or real_path.startswith(norm_pp + "/")
            ):
                return ValidationResult(
                    allowed=False,
                    risk_level="high",
                    reason=f"Protected path: {pp}",
                )

            # (b) requested path is a PARENT of a protected path
            #     e.g. deleting "/usr" when "/usr/bin" is protected
            if (
                norm_pp.startswith(normalized + "/")
                or norm_pp.startswith(real_path + "/")
            ):
                return ValidationResult(
                    allowed=False,
                    risk_level="critical",
                    reason=f"Path is ancestor of protected path {pp}: {file_path}",
                )

        # Config file approval gate
        for cp in CONFIG_APPROVAL_PATHS:
            expanded_cp = self._expand(cp)
            norm_cp = str(Path(expanded_cp).resolve())
            if real_path == norm_cp:
                return ValidationResult(
                    allowed=True,
                    risk_level="high",
                    reason=f"Config file modification requires approval: {cp}",
                )

        return ValidationResult(allowed=True, risk_level="safe")

    def validate_file_read_path(self, file_path: str) -> ValidationResult:
        """Validate a read path against blocked paths."""
        expanded = file_path.replace("~", self._home, 1) if file_path.startswith("~") else file_path
        normalized = str(Path(expanded).resolve())

        real_path = normalized
        try:
            p = Path(normalized)
            if p.exists():
                real_path = str(p.resolve(strict=True))
        except OSError:
            pass

        for bp in READ_BLOCKED_PATHS:
            expanded_bp = self._expand(bp)
            norm_bp = str(Path(expanded_bp).resolve())

            if (
                normalized == norm_bp
                or normalized.startswith(norm_bp + "/")
                or real_path == norm_bp
                or real_path.startswith(norm_bp + "/")
            ):
                return ValidationResult(
                    allowed=False,
                    risk_level="high",
                    reason=f"Reading sensitive path blocked: {bp}",
                )

        return ValidationResult(allowed=True, risk_level="safe")


# Module-level singleton

_guardian: Guardian | None = None


def get_guardian() -> Guardian:
    """Get the singleton Guardian instance."""
    global _guardian
    if _guardian is None:
        _guardian = Guardian()
    return _guardian
