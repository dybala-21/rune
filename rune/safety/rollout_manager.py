"""Safety rollout manager with automatic mode progression.

Ported from src/safety/rollout-manager.ts - reads audit logs and
persisted state to automatically progress through safety rollout modes
(shadow -> balanced -> strict) based on observed metrics.

Supports five modes:
  - auto: automatically progresses through shadow/balanced/strict
  - shadow: legacy behavior with strict shadow logging
  - balanced: medium-risk commands sandboxed, high-risk requires approval
  - strict: deny-by-default with allowlist
  - legacy: original behavior, no new safety checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rune.safety.execution_policy import ExecutionPolicyConfig, SafetyRolloutMode
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

# Types

AutoManagedMode = Literal["shadow", "balanced", "strict"]


@dataclass(slots=True)
class RolloutMetrics:
    bash_samples: int = 0
    actual_blocked: int = 0
    actual_blocked_rate: float = 0.0
    shadow_samples: int = 0
    shadow_deny: int = 0
    shadow_deny_rate: float = 0.0


@dataclass(slots=True)
class RolloutResolution:
    policy: ExecutionPolicyConfig
    mode: SafetyRolloutMode
    changed: bool = False
    previous_mode: SafetyRolloutMode | None = None
    reason: str | None = None
    metrics: RolloutMetrics = field(default_factory=RolloutMetrics)


@dataclass(slots=True)
class _RolloutState:
    mode: SafetyRolloutMode
    changed_at: str
    reason: str | None = None


@dataclass(slots=True)
class _AuditEntry:
    timestamp: str | None = None
    capability: str | None = None
    action: str | None = None
    shadow_decision: str | None = None


@dataclass(slots=True)
class AutoRolloutConfig:
    """Thresholds for automatic mode progression."""

    enabled: bool = True
    observation_window_days: int = 7
    min_shadow_samples: int = 50
    promote_shadow_max_deny_rate: float = 0.05
    min_balanced_samples: int = 100
    promote_balanced_max_blocked_rate: float = 0.02
    rollback_min_samples: int = 20
    rollback_blocked_rate: float = 0.15


# File I/O helpers

def _default_audit_path() -> Path:
    return rune_home() / "audit.jsonl"


def _default_state_path() -> Path:
    return rune_home() / "safety-rollout-state.json"


def _is_rollout_mode(value: str) -> bool:
    return value in ("shadow", "balanced", "strict", "legacy")


def _is_auto_managed(mode: SafetyRolloutMode) -> bool:
    return mode in ("shadow", "balanced", "strict")


def _read_audit_entries(file_path: Path) -> list[_AuditEntry]:
    try:
        raw = file_path.read_text(encoding="utf-8")
        entries: list[_AuditEntry] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json_decode(line)
                entries.append(_AuditEntry(
                    timestamp=data.get("timestamp"),
                    capability=data.get("capability"),
                    action=data.get("action"),
                    shadow_decision=data.get("shadowDecision"),
                ))
            except (json.JSONDecodeError, TypeError):
                continue
        return entries
    except OSError:
        return []


def _read_rollout_state(file_path: Path) -> _RolloutState | None:
    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json_decode(raw)
        mode = data.get("mode", "")
        if not _is_rollout_mode(mode):
            return None
        return _RolloutState(
            mode=mode,
            changed_at=data.get("changedAt", ""),
            reason=data.get("reason"),
        )
    except (OSError, json.JSONDecodeError):
        return None


def _write_rollout_state(file_path: Path, state: _RolloutState) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps({
                "mode": state.mode,
                "changedAt": state.changed_at,
                "reason": state.reason,
            }, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("rollout_state_write_failed", path=str(file_path), error=str(exc))


# Metrics computation

def _compute_metrics(
    entries: list[_AuditEntry],
    observation_window_days: int,
    now: datetime,
) -> RolloutMetrics:
    cutoff = now.timestamp() - observation_window_days * 86400

    recent: list[_AuditEntry] = []
    for e in entries:
        if not e.timestamp:
            continue
        try:
            ts = datetime.fromisoformat(e.timestamp)
            if ts.timestamp() >= cutoff:
                recent.append(e)
        except (ValueError, TypeError):
            continue

    bash_entries = [e for e in recent if e.capability == "bash"]
    shadow_entries = [e for e in bash_entries if e.shadow_decision]
    actual_blocked = sum(
        1 for e in bash_entries if e.action in ("blocked", "denied")
    )
    shadow_deny = sum(
        1 for e in shadow_entries if e.shadow_decision == "deny"
    )

    def safe_rate(num: int, den: int) -> float:
        return num / den if den > 0 else 0.0

    return RolloutMetrics(
        bash_samples=len(bash_entries),
        actual_blocked=actual_blocked,
        actual_blocked_rate=safe_rate(actual_blocked, len(bash_entries)),
        shadow_samples=len(shadow_entries),
        shadow_deny=shadow_deny,
        shadow_deny_rate=safe_rate(shadow_deny, len(shadow_entries)),
    )


# Auto mode resolution

def _resolve_auto_mode(
    current_mode: AutoManagedMode,
    metrics: RolloutMetrics,
    auto_config: AutoRolloutConfig,
) -> tuple[AutoManagedMode, str | None]:
    """Determine next mode based on metrics and thresholds.

    Returns (next_mode, reason_or_None).
    """
    # Rollback first (fail-safe)
    if current_mode in ("balanced", "strict") and (
        metrics.bash_samples >= auto_config.rollback_min_samples
        and metrics.actual_blocked_rate >= auto_config.rollback_blocked_rate
    ):
        new_mode: AutoManagedMode = "balanced" if current_mode == "strict" else "shadow"
        return new_mode, (
            f"Rollback due to high blocked rate "
            f"({metrics.actual_blocked_rate * 100:.1f}%)"
        )

    # Promotions
    if current_mode == "shadow" and (
        metrics.shadow_samples >= auto_config.min_shadow_samples
        and metrics.shadow_deny_rate <= auto_config.promote_shadow_max_deny_rate
    ):
        return "balanced", (
            f"Promoted: shadow deny rate "
            f"{metrics.shadow_deny_rate * 100:.1f}%"
        )

    if current_mode == "balanced" and (
        metrics.bash_samples >= auto_config.min_balanced_samples
        and metrics.actual_blocked_rate <= auto_config.promote_balanced_max_blocked_rate
    ):
        return "strict", (
            f"Promoted: blocked rate "
            f"{metrics.actual_blocked_rate * 100:.1f}%"
        )

    return current_mode, None


# Public API

def resolve_execution_policy(
    *,
    rollout_mode: SafetyRolloutMode | Literal["auto"] = "auto",
    auto_config: AutoRolloutConfig | None = None,
    audit_path: Path | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> RolloutResolution:
    """Resolve the current execution policy based on rollout state.

    When *rollout_mode* is ``"auto"`` and auto is enabled, reads audit
    logs and persisted state to automatically progress through modes.
    Otherwise returns a static resolution for the given mode.
    """
    if auto_config is None:
        auto_config = AutoRolloutConfig()

    if rollout_mode != "auto" or not auto_config.enabled:
        mode: SafetyRolloutMode = "shadow" if rollout_mode == "auto" else rollout_mode
        return RolloutResolution(
            policy=ExecutionPolicyConfig(rollout_mode=mode),
            mode=mode,
            changed=False,
            metrics=RolloutMetrics(),
        )

    if now is None:
        now = datetime.now(UTC)
    resolved_audit = audit_path or _default_audit_path()
    resolved_state = state_path or _default_state_path()

    entries = _read_audit_entries(resolved_audit)
    state = _read_rollout_state(resolved_state)

    metrics = _compute_metrics(entries, auto_config.observation_window_days, now)

    current_mode: AutoManagedMode = (
        state.mode
        if state and _is_auto_managed(state.mode)
        else "shadow"
    )
    next_mode, reason = _resolve_auto_mode(current_mode, metrics, auto_config)

    changed = next_mode != current_mode
    if changed:
        _write_rollout_state(resolved_state, _RolloutState(
            mode=next_mode,
            changed_at=now.isoformat(),
            reason=reason,
        ))
        log.info(
            "safety_rollout_mode_changed",
            from_mode=current_mode,
            to_mode=next_mode,
            reason=reason,
        )
    elif state is None:
        _write_rollout_state(resolved_state, _RolloutState(
            mode=current_mode,
            changed_at=now.isoformat(),
            reason="Initialized auto rollout state",
        ))

    return RolloutResolution(
        policy=ExecutionPolicyConfig(rollout_mode=next_mode),
        mode=next_mode,
        changed=changed,
        previous_mode=current_mode if changed else None,
        reason=reason,
        metrics=metrics,
    )
