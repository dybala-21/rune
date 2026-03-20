"""Skill security gate - security checks for skill creation.

Ported from src/agent/hooks/skill-security-gate.ts (195 lines) - pre-tool-use
hook that validates skill creation requests against security policies.

Checks:
- Project scope policy
- Body size limits
- Author allowlist
- Signature verification
- Suspicious pattern detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rune.agent.hooks.runner import HookHandler, HookResult, PreToolUseContext
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

SkillGateMode = str  # "off" | "advisory" | "required"


@dataclass(slots=True)
class SkillSecurityGateConfig:
    """Configuration for the skill security gate."""

    mode: SkillGateMode = "advisory"
    auto_harden_on_code_tasks: bool = True
    allowed_authors: list[str] = field(default_factory=lambda: ["rune-agent"])
    require_signature: bool = False
    allow_auto_sign_when_missing: bool = True
    signature_secret_env: str = "RUNE_SKILL_SIGNING_KEY"
    block_project_scope: bool = True
    project_scope_allowed_name_prefixes: list[str] = field(default_factory=list)
    max_body_chars: int = 12_000
    suspicious_patterns: list[str] = field(default_factory=lambda: [
        r"curl\s+.*\|\s*(bash|sh)",
        r"wget\s+.*\|\s*(bash|sh)",
        r"rm\s+-rf\s+/",
        r"chmod\s+777",
        r"\bsudo\b",
        r"export\s+(OPENAI|ANTHROPIC|AWS|GITHUB)_[A-Z_]*\s*=",
        r"(api[_-]?key|secret|token|password)\s*[:=]",
        r"\.env",
    ])


DEFAULT_SKILL_SECURITY_GATE_CONFIG = SkillSecurityGateConfig()


# Finding detection

async def _detect_findings(
    *,
    name: str,
    description: str,
    body: str,
    scope: str,
    author: str,
    signature: str | None,
    config: SkillSecurityGateConfig,
) -> list[str]:
    """Detect security findings for a skill creation request."""
    findings: list[str] = []

    # Project scope check
    if config.block_project_scope and scope == "project":
        findings.append("project scope skill creation is blocked by policy")

    if (
        scope == "project"
        and not config.block_project_scope
        and config.project_scope_allowed_name_prefixes
        and not any(name.startswith(prefix) for prefix in config.project_scope_allowed_name_prefixes)
    ):
        findings.append(
            f'project scope skill name "{name}" is outside allowed prefixes'
        )

    # Body size check
    if len(body) > config.max_body_chars:
        findings.append(f"skill body too large ({len(body)} > {config.max_body_chars})")

    # Author allowlist check
    if config.allowed_authors and author not in config.allowed_authors:
        findings.append(f'author "{author}" is not in allowedAuthors')

    # Signature verification
    if config.require_signature:
        import os

        secret = os.environ.get(config.signature_secret_env, "")
        has_signature = bool(signature and signature.strip())

        if has_signature:
            if not secret:
                findings.append(
                    f'signature secret env "{config.signature_secret_env}" is not set'
                )
            else:
                # Attempt signature verification
                try:
                    from rune.skills.signing import verify_skill_signature

                    verified = verify_skill_signature(
                        name=name,
                        description=description,
                        body=body,
                        scope=scope,
                        author=author,
                        secret=secret,
                        signature=signature,
                    )
                    if not verified:
                        findings.append("signature verification failed")
                except ImportError:
                    findings.append("signature verification module not available")
        elif config.allow_auto_sign_when_missing:
            if not secret:
                findings.append(
                    f'signature is required by policy and auto-sign secret env '
                    f'"{config.signature_secret_env}" is not set'
                )
        else:
            findings.append("signature is required by policy")

    # Suspicious pattern detection
    for pattern_text in config.suspicious_patterns:
        try:
            pattern = re.compile(pattern_text, re.IGNORECASE)
            if pattern.search(body):
                findings.append(f"matched suspicious pattern: {pattern_text}")
        except re.error:
            pass  # Invalid user pattern should not crash hooks

    return findings


# Config resolution

def resolve_skill_security_gate_config_for_goal(
    config: SkillSecurityGateConfig,
    *,
    requires_code: bool,
) -> SkillSecurityGateConfig:
    """Auto-harden advisory mode to required for code tasks."""
    if config.mode != "advisory":
        return config
    if not config.auto_harden_on_code_tasks:
        return config
    if not requires_code:
        return config

    return SkillSecurityGateConfig(
        mode="required",
        auto_harden_on_code_tasks=config.auto_harden_on_code_tasks,
        allowed_authors=config.allowed_authors,
        require_signature=config.require_signature,
        allow_auto_sign_when_missing=config.allow_auto_sign_when_missing,
        signature_secret_env=config.signature_secret_env,
        block_project_scope=config.block_project_scope,
        project_scope_allowed_name_prefixes=config.project_scope_allowed_name_prefixes,
        max_body_chars=config.max_body_chars,
        suspicious_patterns=config.suspicious_patterns,
    )


# Hook factory

def create_skill_security_gate_hook(
    config: SkillSecurityGateConfig | None = None,
) -> HookHandler:
    """Create a pre_tool_use hook for skill security gating.

    Returns a handler that checks skill.create capabilities against
    the configured security policy.
    """
    cfg = config or DEFAULT_SKILL_SECURITY_GATE_CONFIG

    async def handler(context: PreToolUseContext) -> HookResult:
        if cfg.mode == "off":
            return HookResult(decision="pass")
        if context.capability != "skill.create":
            return HookResult(decision="pass")

        params = context.params
        scope_raw = params.get("scope", "user")
        scope = "project" if scope_raw == "project" else "user"
        name = str(params.get("name", ""))
        description = str(params.get("description", ""))
        body = str(params.get("body", ""))
        author_raw = params.get("author", "")
        author = str(author_raw).strip() if author_raw else "rune-agent"
        signature = str(params.get("signature", "")) or None

        findings = await _detect_findings(
            name=name,
            description=description,
            body=body,
            scope=scope,
            author=author,
            signature=signature,
            config=cfg,
        )

        if not findings:
            return HookResult(
                decision="pass",
                metadata={"checked": True, "findings": 0},
            )

        message = f"Skill security gate findings:\n- {chr(10).join('- ' + f for f in findings)}"

        if cfg.mode == "required":
            return HookResult(
                decision="block",
                message=message,
                metadata={"findings": findings},
            )

        return HookResult(
            decision="warn",
            message=message,
            metadata={"findings": findings},
        )

    return handler
