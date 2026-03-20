"""Execution policy with progressive rollout for RUNE.

Ported 1:1 from src/safety/execution-policy.ts - 4 rollout modes
(shadow/balanced/strict/legacy), deny-by-default allowlist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rune.safety.guardian import ValidationResult

# Types

SafetyRolloutMode = Literal["shadow", "balanced", "strict", "legacy"]
Decision = Literal["allow", "ask", "deny"]


@dataclass(slots=True)
class BashDecision:
    decision: Decision
    reason: str
    use_sandbox: bool = False
    shadow: BashDecisionShadow | None = None


@dataclass(slots=True)
class BashDecisionShadow:
    decision: Decision
    reason: str
    use_sandbox: bool = False


@dataclass(slots=True)
class ExecutionPolicyConfig:
    rollout_mode: SafetyRolloutMode = "shadow"
    sandbox_enabled: bool = True
    sandbox_on_unavailable: Literal["deny", "ask", "allow"] = "deny"
    sandbox_on_execution_failure: Literal["deny", "ask", "allow"] = "deny"
    deny_by_default_enabled: bool = True
    allowed_executables: list[str] = field(default_factory=list)


# Default Allowlist

DEFAULT_ALLOWED_EXECUTABLES = [
    "ls", "pwd", "cd", "echo", "cat", "head", "tail", "sed", "awk", "rg",
    "find", "wc", "sort", "uniq",
    "git", "npm", "pnpm", "yarn", "node", "npx",
    "python", "python3", "pip", "pip3", "pytest", "vitest", "jest",
    "go", "cargo", "rustc", "make",
    "docker", "docker-compose", "uv",
]

_ENV_PREFIX_RE = re.compile(r"^(\w+=\S+\s+)+")


# Helpers

def _normalize_executable(command: str) -> str:
    """Extract the executable name, stripping env-prefix assignments."""
    trimmed = command.strip()
    if not trimmed:
        return ""
    without_env = _ENV_PREFIX_RE.sub("", trimmed)
    first_token = without_env.split()[0] if without_env.split() else ""
    return first_token.lower()


@dataclass(slots=True)
class _BashDecisionInput:
    command: str
    validation: ValidationResult
    config: ExecutionPolicyConfig
    has_sandbox_support: bool = False
    interactive_approval: bool = True


def _to_final_decision(
    decision: BashDecision,
    interactive_approval: bool,
    fail_closed: bool = True,
) -> BashDecision:
    if decision.decision == "ask" and not interactive_approval and fail_closed:
        return BashDecision(
            decision="deny",
            reason=f"{decision.reason} (approval unavailable in this channel)",
            use_sandbox=False,
        )
    return decision


def _with_sandbox_availability(
    decision: BashDecision, inp: _BashDecisionInput
) -> BashDecision:
    if not decision.use_sandbox:
        return decision
    if inp.has_sandbox_support:
        return decision

    action = inp.config.sandbox_on_unavailable
    if action == "deny":
        return BashDecision(
            decision="deny",
            reason=f"{decision.reason} (sandbox unavailable)",
            use_sandbox=False,
        )
    if action == "ask":
        return _to_final_decision(
            BashDecision(
                decision="ask",
                reason=f"{decision.reason} (sandbox unavailable: explicit approval required)",
                use_sandbox=False,
            ),
            inp.interactive_approval,
        )
    # allow
    return BashDecision(
        decision="allow",
        reason=f"{decision.reason} (sandbox unavailable: allowed by policy)",
        use_sandbox=False,
    )


# Mode evaluators

def _evaluate_mode(mode: SafetyRolloutMode, inp: _BashDecisionInput) -> BashDecision:
    validation = inp.validation
    config = inp.config
    executable = _normalize_executable(inp.command)
    allowed_set = {x.lower() for x in config.allowed_executables}
    is_known = executable in allowed_set
    risk = validation.risk_level

    # Hard stop from Guardian
    if not validation.allowed:
        return BashDecision(
            decision="deny",
            reason=validation.reason or "Blocked by Guardian",
        )

    # Legacy mode
    if mode == "legacy":
        requires_approval = validation.requires_approval
        return BashDecision(
            decision="ask" if requires_approval else "allow",
            reason=(
                validation.reason or "Guardian requires approval"
                if requires_approval
                else "Legacy allow"
            ),
            use_sandbox=config.sandbox_enabled and risk in ("medium", "high"),
        )

    # Critical always denied
    if risk == "critical":
        return BashDecision(
            decision="deny",
            reason="Critical risk command denied",
        )

    # Balanced mode
    if mode == "balanced":
        if risk == "high" or validation.requires_approval:
            return BashDecision(
                decision="ask",
                reason=validation.reason or "High-risk command requires approval",
                use_sandbox=config.sandbox_enabled,
            )
        return BashDecision(
            decision="allow",
            reason=(
                "Medium-risk command allowed with sandbox"
                if risk == "medium"
                else "Low-risk command allowed"
            ),
            use_sandbox=config.sandbox_enabled and risk == "medium",
        )

    # Strict mode
    if config.deny_by_default_enabled and not is_known:
        return BashDecision(
            decision="ask",
            reason=f'Executable "{executable or "(unknown)"}" is not allowlisted (deny-by-default)',
            use_sandbox=config.sandbox_enabled,
        )

    if risk in ("high", "medium") or validation.requires_approval:
        return BashDecision(
            decision="ask",
            reason=validation.reason or f"{risk} risk command requires explicit approval",
            use_sandbox=config.sandbox_enabled,
        )

    return BashDecision(
        decision="allow",
        reason="Allowlisted low-risk command",
    )


# Public API

def decide_bash_execution(
    command: str,
    validation: ValidationResult,
    config: ExecutionPolicyConfig | None = None,
    *,
    has_sandbox_support: bool = False,
    interactive_approval: bool = True,
) -> BashDecision:
    """Decide whether a bash command should be allowed, asked, or denied."""
    if config is None:
        config = ExecutionPolicyConfig(
            allowed_executables=list(DEFAULT_ALLOWED_EXECUTABLES),
        )

    inp = _BashDecisionInput(
        command=command,
        validation=validation,
        config=config,
        has_sandbox_support=has_sandbox_support,
        interactive_approval=interactive_approval,
    )
    mode = config.rollout_mode

    if mode == "shadow":
        legacy_decision = _evaluate_mode("legacy", inp)
        strict_decision = _evaluate_mode("strict", inp)
        resolved_legacy = _to_final_decision(
            _with_sandbox_availability(legacy_decision, inp),
            inp.interactive_approval,
            fail_closed=False,
        )
        resolved_strict = _to_final_decision(
            _with_sandbox_availability(strict_decision, inp),
            inp.interactive_approval,
        )
        resolved_legacy.shadow = BashDecisionShadow(
            decision=resolved_strict.decision,
            reason=resolved_strict.reason,
            use_sandbox=resolved_strict.use_sandbox,
        )
        return resolved_legacy

    decision = _evaluate_mode(mode, inp)
    with_sandbox = _with_sandbox_availability(decision, inp)
    return _to_final_decision(with_sandbox, inp.interactive_approval)
