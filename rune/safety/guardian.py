"""Validate shell commands and file paths against risk and approval policies."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rune.safety.analyzer import analyze_command, classify_rm_rf_risk, normalize_command
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

RiskLevel = Literal["safe", "low", "medium", "high", "critical"]


@dataclass(slots=True)
class ValidationResult:
    allowed: bool
    risk_level: RiskLevel
    reason: str = ""
    suggestions: list[str] = field(default_factory=list)
    requires_approval: bool = False


# Dangerous Bash Patterns

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

# Protect these paths and their ancestors from writes and deletions.
_CRITICAL_ROOT_PATHS = [
    "/",
    "~",  # expanded at runtime
]

# Reject writes too close to the filesystem root.
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


def _is_rune_secret_file(path: str) -> bool:
    """Match Rune's credential file and backups, excluding project .env and .env.example files."""
    try:
        from rune.cloud.store import cloud_home
        from rune.connectors.store import broker_home
        from rune.utils.paths import rune_home

        candidate = Path(path).resolve()
        if any(candidate.is_relative_to(root.resolve()) for root in (broker_home(), cloud_home())):
            return True
        return (
            candidate.name.startswith(".env")
            and candidate.parent.resolve() == rune_home().resolve()
        )
    except Exception:  # pragma: no cover - path or home resolution failure
        return False


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


def _milder(a: ValidationResult, b: ValidationResult) -> bool:
    """Is *a* a softer answer than *b* on any axis that matters?"""
    return (risk_to_number(a.risk_level) < risk_to_number(b.risk_level)
            or (a.allowed and not b.allowed)
            or (b.requires_approval and not a.requires_approval))


def _raised(base: ValidationResult, level: RiskLevel) -> ValidationResult:
    """Raise risk without relaxing approval or blocks; critical risk always blocks."""
    if risk_to_number(level) <= risk_to_number(base.risk_level):
        return base
    out = ValidationResult(
        allowed=base.allowed and level != "critical",
        risk_level=level,
        reason=(base.reason if level != "critical" else
                "Recursive deletion targeting system or critical path")
        or "Deletion seen in the parsed command",
        suggestions=base.suggestions,
        requires_approval=base.requires_approval,
    )
    if _milder(out, base):      # unreachable by construction
        log.error("guardian_escalation_would_soften",
                  base=base.risk_level, raised=level)
        return base
    log.debug("guardian_escalation", from_level=base.risk_level, to=level)
    return out


# Guardian

class Guardian:
    """Real-time safety validator for bash commands and file paths."""

    def __init__(self) -> None:
        self._home = os.environ.get("HOME", str(Path.home()))
        self._approval_callback: Callable[[str], Awaitable[bool]] | None = None

    def _expand(self, p: str) -> str:
        return p.replace("~", self._home, 1) if p.startswith("~") else p

    def validate(self, command: str, _context: str | None = None, *, cwd: str | None = None) -> ValidationResult:
        """Check commands and write targets without weakening an existing safety verdict."""
        base = self._validate_patterns(command, _context)
        from rune.safety.shell_ast import worst_deletion

        seen = worst_deletion(command)
        if seen is not None:
            base = _raised(base, seen)
        if not base.allowed:
            return base
        from rune.safety.shell_writes import is_raw_python, shell_write_targets

        if is_raw_python(command):
            return ValidationResult(
                allowed=False, risk_level=base.risk_level,
                reason=("Python source was passed as a shell command; nothing was executed. "
                        "For a one-off calculation or file transformation, pass the code directly "
                        "through a quoted shell heredoc: python3 - <<'PY'\n<code>\nPY. "
                        "Approval cannot correct this command format."),
            )

        targets = shell_write_targets(command, cwd or "", home=self._home)
        for path in sorted(targets.paths):
            check = self.validate_file_path(path)
            if not check.allowed or check.requires_approval:
                base = ValidationResult(
                    allowed=base.allowed and check.allowed,
                    risk_level=max((base.risk_level, check.risk_level), key=risk_to_number),
                    reason=check.reason,
                    requires_approval=base.requires_approval or check.requires_approval,
                )
            if not base.allowed:
                return base
        if targets.uncertain:
            return ValidationResult(
                allowed=True,
                risk_level=max((base.risk_level, "high"), key=risk_to_number),
                reason="Shell write targets cannot be resolved before execution; approval is required.",
                requires_approval=True,
            )
        return base

    def _validate_patterns(self, command: str,
                           _context: str | None = None) -> ValidationResult:
        """Check original and normalized commands, keeping the higher risk."""
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

        # Only critical risk can short-circuit; weaker matches must not bypass stricter checks.
        if worst_result is not None and worst_result.risk_level == "critical":
            return worst_result

        # Bash command referencing protected/blocked paths
        if self._command_reads_rune_secret(command, analysis.parsed) or (
            normalized != command
            and self._command_reads_rune_secret(
                normalized, normalized_analysis.parsed
            )
        ):
            return ValidationResult(
                allowed=False,
                risk_level="high",
                reason="Reading RUNE's credential store is not permitted",
            )

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
            by_score = ValidationResult(
                allowed=True,
                risk_level="medium",
                reason=", ".join(f.description for f in effective.findings),
            )
        else:
            by_score = ValidationResult(
                allowed=True,
                risk_level="low" if effective.risk_score >= 15 else "safe",
            )

        # Whichever of the two saw more, decides.
        if worst_result is not None and risk_to_number(
                worst_result.risk_level) >= risk_to_number(by_score.risk_level):
            return worst_result
        return by_score

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
        except OSError as exc:
            # Log audit failures without interrupting the safety check.
            log.debug("guardian_audit_write_failed", error=str(exc))

    def is_command_safe(self, command: str) -> bool:
        """Quick boolean check -- True if command is safe to execute without approval."""
        result = self.validate(command)
        return result.allowed and not result.requires_approval

    def analyze_command(self, command: str) -> Any:
        """Analyze a command and return findings. Delegates to analyzer module."""
        return analyze_command(command)

    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Reject shallow or protected paths and require approval for configuration writes."""
        # empty path guard
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
        except OSError as exc:
            # Check the unresolved path when resolution fails rather than skipping protection.
            log.debug("guardian_symlink_unresolvable", path=normalized,
                      error=str(exc))

        from rune.cloud.store import cloud_home
        from rune.connectors.store import broker_home
        target = Path(real_path)
        if any(target.is_relative_to(root.resolve()) or root.resolve().is_relative_to(target)
               for root in (broker_home(), cloud_home())):
            return ValidationResult(allowed=False, risk_level="critical",
                                    reason="Broker and hosting state may only be changed by their management commands")

        # critical root paths
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

            # Reject ancestors of protected paths, such as /usr when /usr/bin is protected.
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
                    requires_approval=True,
                )

        return ValidationResult(allowed=True, risk_level="safe")

    def _command_reads_rune_secret(self, command: str, parsed: Any = None) -> bool:
        """Resolve shell arguments to detect access to Rune's credential store.

        Runtime-built paths, including variables and substitutions, are not visible here; this
        is not a security boundary against a caller who controls the shell.
        """
        try:
            if parsed is None:
                from rune.safety.analyzer import analyze_command

                parsed = analyze_command(command).parsed
            chain = parsed.chained_commands or [command]
        except Exception as exc:  # pragma: no cover - parser failure
            log.debug("guardian_parse_failed", error=str(exc))
            return False

        cwd: Path | None = None
        for part in chain:
            tokens = part.split()
            if not tokens:
                continue
            # "cd <dir> && cat .env": later relative args resolve against <dir>.
            if tokens[0] == "cd" and len(tokens) > 1:
                cwd = self._resolve_arg(tokens[1], None)
                continue
            for token in tokens[1:]:
                resolved = self._resolve_arg(token, cwd)
                if resolved is not None and _is_rune_secret_file(str(resolved)):
                    return True
        return False

    def _resolve_arg(self, token: str, cwd: Path | None) -> Path | None:
        """Strip shell quoting from *token* and resolve it to a real path."""
        cleaned = token.replace('"', "").replace("'", "").strip()
        if not cleaned or cleaned.startswith("-"):
            return None
        cleaned = cleaned.replace("$HOME", self._home).replace("${HOME}", self._home)
        if cleaned.startswith("~"):
            cleaned = cleaned.replace("~", self._home, 1)
        try:
            base = Path(cleaned)
            if not base.is_absolute() and cwd is not None:
                base = cwd / base
            return Path(os.path.normpath(str(base)))
        except (OSError, ValueError):
            return None

    def validate_file_read_path(self, file_path: str) -> ValidationResult:
        """Validate a read path against blocked paths."""
        return self.read_path_validator()(file_path)

    def read_path_validator(self):
        """Resolve policy roots once when scanning many files; resolve each target afresh."""
        from rune.cloud.boundary import hosted, workspace_root
        from rune.cloud.store import cloud_home
        from rune.connectors.store import broker_home
        from rune.utils.paths import rune_home

        private = rune_home().resolve()
        roots = tuple(Path(self._expand(path)).resolve() for path in READ_BLOCKED_PATHS)
        roots += (broker_home().resolve(), cloud_home().resolve())
        workspace = workspace_root() if hosted() else None
        boundary = Path(workspace).resolve() if workspace else None
        hosted_mode = hosted()

        def validate(file_path: str) -> ValidationResult:
            expanded = file_path.replace("~", self._home, 1) if file_path.startswith("~") else file_path
            try:
                target = Path(expanded).resolve()
            except (OSError, ValueError):
                return ValidationResult(allowed=False, risk_level="high", reason="Cannot resolve file read path")
            if hosted_mode and (boundary is None or not target.is_relative_to(boundary)):
                return ValidationResult(allowed=False, risk_level="high", reason="Hosted file access is restricted to the owner's workspace")
            if target.name.startswith(".env") and target.parent == private:
                return ValidationResult(allowed=False, risk_level="high", reason="Reading RUNE's credential store is not permitted")
            for root in roots:
                if target.is_relative_to(root):
                    return ValidationResult(allowed=False, risk_level="high", reason=f"Reading sensitive path blocked: {root}")
            return ValidationResult(allowed=True, risk_level="safe")

        return validate


# Module-level singleton

_guardian: Guardian | None = None


def get_guardian() -> Guardian:
    """Get the singleton Guardian instance."""
    global _guardian
    if _guardian is None:
        _guardian = Guardian()
    return _guardian
