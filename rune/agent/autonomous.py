"""Autonomous executor - adaptive autonomy engine for RUNE.

Ported from src/agent/autonomous.ts (991 lines).
Manages a learning autonomy system where the agent can escalate from
SUGGEST to INFORM_DO to JUST_DO based on user feedback patterns, with
demotion via a sliding window of recent executions.

Governance flags (shadow_mode, kill_switch) allow operators to disable
autonomous execution globally.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Enums & type aliases

class AutonomyLevel(IntEnum):
    """Autonomy escalation levels. Higher values = more autonomous."""

    SUGGEST = 0
    INFORM_DO = 1
    JUST_DO = 2


TaskDomain = Literal[
    "git", "build", "file", "browser", "system", "notify", "cleanup", "unknown"
]

ExecutionFeedback = Literal[
    "approved", "full_revert", "partial_revert", "manual_correction"
]


# Data classes

@dataclass(slots=True)
class PatternStats:
    """Tracks historical execution statistics for a single command pattern."""

    total_executions: int = 0
    approved: int = 0
    reverted: int = 0
    manual_corrections: int = 0
    avg_risk: float = 0.0
    last_used: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class AutonomyDecision:
    """The result of an autonomy-level decision for a given command."""

    level: AutonomyLevel
    domain: TaskDomain
    risk_score: float
    reason: str
    pattern_key: str


@dataclass(slots=True)
class AutonomousExecution:
    """Full execution record matching TS AutonomousExecution (12 fields).

    Used by ``AutonomousExecutor.record_execution()`` to track individual
    autonomous actions with complete audit context.
    """

    id: str
    timestamp: float = field(default_factory=time.monotonic)
    level: AutonomyLevel = AutonomyLevel.SUGGEST
    domain: TaskDomain = "unknown"
    description: str = ""
    action: str = ""
    success: bool = False
    result_summary: str = ""
    duration_ms: float = 0.0
    reversible: bool = False
    user_feedback: ExecutionFeedback | None = None
    rollback_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AutonomyPolicy:
    """Configuration governing autonomy promotion/demotion thresholds."""

    domain_levels: dict[TaskDomain, AutonomyLevel] = field(default_factory=dict)
    risk_thresholds: dict[AutonomyLevel, float] = field(default_factory=dict)
    promotion_conditions: dict[str, Any] = field(default_factory=dict)
    demotion_conditions: dict[str, Any] = field(default_factory=dict)


# Default policy

DEFAULT_AUTONOMY_POLICY = AutonomyPolicy(
    domain_levels={
        "git": AutonomyLevel.SUGGEST,
        "build": AutonomyLevel.SUGGEST,
        "file": AutonomyLevel.SUGGEST,
        "browser": AutonomyLevel.SUGGEST,
        "system": AutonomyLevel.SUGGEST,
        "notify": AutonomyLevel.JUST_DO,
        "cleanup": AutonomyLevel.INFORM_DO,
        "unknown": AutonomyLevel.SUGGEST,
    },
    risk_thresholds={
        AutonomyLevel.JUST_DO: 0.3,
        AutonomyLevel.INFORM_DO: 0.6,
        AutonomyLevel.SUGGEST: 1.0,
    },
    promotion_conditions={
        # Minimum approved executions before promotion is considered
        "min_approved": 5,
        # Minimum approval rate (approved / total)
        "min_approval_rate": 0.9,
        # Maximum average risk score
        "max_avg_risk": 0.4,
    },
    demotion_conditions={
        # Sliding window size
        "window_size": 20,
        # Maximum reverts allowed in window before demotion
        "max_reverts_in_window": 2,
        # Maximum manual corrections in window before demotion
        "max_corrections_in_window": 3,
    },
)


# AutonomousExecutor

# Domain classifier regexes. Order matters: first match wins.
_DOMAIN_PATTERNS: list[tuple[TaskDomain, re.Pattern[str]]] = [
    ("git", re.compile(r"\bgit\b")),
    ("build", re.compile(r"\b(?:npm|yarn|pnpm|pip|cargo|make|gradle|mvn)\b")),
    ("browser", re.compile(r"\b(?:open|xdg-open|start|browse)\b")),
    ("cleanup", re.compile(r"\b(?:rm|clean|prune|purge|cache\s+clear)\b")),
    ("notify", re.compile(r"\b(?:notify|alert|toast|message)\b")),
    ("system", re.compile(r"\b(?:sudo|systemctl|service|launchctl|chmod|chown)\b")),
    ("file", re.compile(r"\b(?:cp|mv|mkdir|touch|cat|echo|write|read|edit)\b")),
]


class AutonomousExecutor:
    """Adaptive autonomy engine.

    Learns from user feedback to escalate or reduce autonomy per-command
    pattern.  Supports serialisation for persistence across sessions.
    """

    __slots__ = (
        "_policy",
        "_patterns",
        "_history",
        "_execution_history",
        "_shadow_mode",
        "_kill_switch",
    )

    def __init__(self, policy: AutonomyPolicy | None = None) -> None:
        self._policy = policy or DEFAULT_AUTONOMY_POLICY
        self._patterns: dict[str, PatternStats] = {}
        # Sliding window of (pattern_key, feedback) tuples, most recent last.
        self._history: deque[tuple[str, ExecutionFeedback]] = deque(maxlen=100)
        # Full execution records for audit trail (capped at 100).
        self._execution_history: deque[AutonomousExecution] = deque(maxlen=100)
        self._shadow_mode: bool = False
        self._kill_switch: bool = False

    # -- Properties ---------------------------------------------------------

    @property
    def shadow_mode(self) -> bool:
        """When ``True``, decisions are logged but never executed autonomously.
        All commands fall back to SUGGEST."""
        return self._shadow_mode

    @shadow_mode.setter
    def shadow_mode(self, value: bool) -> None:
        self._shadow_mode = value
        log.info("autonomy_shadow_mode", enabled=value)

    @property
    def kill_switch(self) -> bool:
        """Emergency stop. Forces all decisions to SUGGEST."""
        return self._kill_switch

    @kill_switch.setter
    def kill_switch(self, value: bool) -> None:
        self._kill_switch = value
        log.warning("autonomy_kill_switch", enabled=value)

    @property
    def pattern_stats(self) -> dict[str, PatternStats]:
        return dict(self._patterns)

    @property
    def feedback_history(self) -> list[tuple[str, ExecutionFeedback]]:
        """Sliding window of (pattern_key, feedback) tuples."""
        return list(self._history)

    # -- Core decision logic ------------------------------------------------

    def decide(
        self,
        command: str,
        domain: TaskDomain | None = None,
        risk_score: float = 0.5,
    ) -> AutonomyDecision:
        """Determine the autonomy level for *command*.

        Parameters:
            command: The command string being evaluated.
            domain: Explicit domain override; auto-classified if ``None``.
            risk_score: Risk score in [0, 1].

        Returns:
            An :class:`AutonomyDecision` with the resolved level.
        """
        resolved_domain = domain or self._classify_domain(command)
        pattern_key = self._pattern_key(command, resolved_domain)

        # Governance overrides
        if self._kill_switch or self._shadow_mode:
            return AutonomyDecision(
                level=AutonomyLevel.SUGGEST,
                domain=resolved_domain,
                risk_score=risk_score,
                reason="kill_switch" if self._kill_switch else "shadow_mode",
                pattern_key=pattern_key,
            )

        # Base level from policy
        base_level = self._policy.domain_levels.get(
            resolved_domain, AutonomyLevel.SUGGEST
        )

        # Risk gate: if risk exceeds the threshold for the base level,
        # downgrade to a safer level.
        level = base_level
        for check_level in (AutonomyLevel.JUST_DO, AutonomyLevel.INFORM_DO):
            threshold = self._policy.risk_thresholds.get(check_level, 0.0)
            if risk_score > threshold and level >= check_level:
                level = AutonomyLevel(max(0, check_level - 1))

        # Track risk score via EMA so avg_risk reflects actual usage.
        stats = self._patterns.get(pattern_key)
        if stats is not None:
            alpha = 0.3
            stats.avg_risk = alpha * risk_score + (1 - alpha) * stats.avg_risk

        # Pattern-based promotion: if the user has repeatedly approved this
        # pattern, the level may be promoted above the base.
        if stats is not None:
            promoted = self._check_promotion(pattern_key, stats)
            if promoted is not None and promoted > level:
                level = promoted

        # Demotion check: recent failures in the sliding window pull
        # everything down.
        if self._check_demotion_window():
            level = min(level, AutonomyLevel.SUGGEST)

        reason = (
            f"domain={resolved_domain}, risk={risk_score:.2f}, "
            f"base={base_level.name}, resolved={level.name}"
        )
        log.debug("autonomy_decision", pattern_key=pattern_key, reason=reason)

        return AutonomyDecision(
            level=level,
            domain=resolved_domain,
            risk_score=risk_score,
            reason=reason,
            pattern_key=pattern_key,
        )

    # -- Feedback -----------------------------------------------------------

    def record_feedback(
        self,
        pattern_key: str,
        feedback: ExecutionFeedback,
        risk_score: float | None = None,
    ) -> None:
        """Record user feedback for a completed execution.

        Parameters:
            pattern_key: The pattern key identifying the command.
            feedback: The user's feedback on the execution.
            risk_score: Optional risk score from the decision that triggered
                this execution.  Used to maintain a rolling average via EMA.
        """
        stats = self._patterns.setdefault(pattern_key, PatternStats())
        stats.total_executions += 1
        stats.last_used = time.monotonic()

        match feedback:
            case "approved":
                stats.approved += 1
            case "full_revert" | "partial_revert":
                stats.reverted += 1
            case "manual_correction":
                stats.manual_corrections += 1

        # Update rolling average risk via EMA (α = 0.3)
        if risk_score is not None:
            alpha = 0.3
            stats.avg_risk = alpha * risk_score + (1 - alpha) * stats.avg_risk
        self._history.append((pattern_key, feedback))
        log.debug(
            "autonomy_feedback",
            pattern_key=pattern_key,
            feedback=feedback,
            total=stats.total_executions,
        )

    def record_execution(self, execution: AutonomousExecution) -> None:
        """Record a full autonomous execution with all context fields.

        Mirrors TS ``autonomy.recordExecution(execution)``. Stores the
        execution in history and delegates to ``record_feedback()`` for
        pattern stats updates.
        """
        self._execution_history.append(execution)

        # Derive pattern_key and feedback, then delegate to record_feedback
        pattern_key = f"{execution.domain}:{execution.action[:60]}"
        if execution.user_feedback is not None:
            feedback = execution.user_feedback
        else:
            feedback: ExecutionFeedback = "approved" if execution.success else "full_revert"

        self.record_feedback(pattern_key, feedback)

        log.debug(
            "autonomy_execution_recorded",
            execution_id=execution.id,
            domain=execution.domain,
            success=execution.success,
            duration_ms=execution.duration_ms,
        )

    @property
    def execution_history(self) -> list[AutonomousExecution]:
        """Return a copy of the execution history."""
        return list(self._execution_history)

    # -- Promotion / demotion -----------------------------------------------

    def _check_promotion(
        self,
        pattern_key: str,
        stats: PatternStats,
    ) -> AutonomyLevel | None:
        """Check if *pattern_key* qualifies for promotion.

        Returns the promoted level or ``None`` if no change.
        """
        cond = self._policy.promotion_conditions
        min_approved: int = cond.get("min_approved", 5)
        min_rate: float = cond.get("min_approval_rate", 0.9)
        max_risk: float = cond.get("max_avg_risk", 0.4)

        if stats.total_executions < min_approved:
            return None

        approval_rate = stats.approved / stats.total_executions
        if approval_rate < min_rate:
            return None

        if stats.avg_risk > max_risk:
            return None

        # Promote one level above what the domain default would give
        domain_default = AutonomyLevel.SUGGEST
        # Infer domain from pattern_key
        for d, lvl in self._policy.domain_levels.items():
            if d in pattern_key:
                domain_default = lvl
                break

        promoted = AutonomyLevel(min(domain_default + 1, AutonomyLevel.JUST_DO))
        log.info(
            "autonomy_promotion",
            pattern_key=pattern_key,
            new_level=promoted.name,
            approval_rate=f"{approval_rate:.2%}",
        )

        # Record promotion in learned.md so the user sees it in session
        # briefing. This makes self-improving visible.
        try:
            from rune.memory.markdown_store import save_learned_fact

            # Human-readable command name: "unknown:web_search python" -> "web_search"
            cmd = pattern_key.split(":")[-1].split()[0] if ":" in pattern_key else pattern_key[:20]
            level_desc = {
                AutonomyLevel.INFORM_DO: f"{cmd}은(는) 이제 확인 없이 자동 실행합니다",
                AutonomyLevel.JUST_DO: f"{cmd}은(는) 이제 알림 없이 자동 실행합니다",
            }.get(promoted, f"{cmd} 승격됨")
            save_learned_fact(
                category="autonomy",
                key=f"auto_{cmd[:20]}",
                value=f"{level_desc} ({stats.approved}회 승인 학습)",
                confidence=min(approval_rate, 0.95),
            )
        except Exception:
            pass  # Must never block promotion logic

        return promoted

    def _check_demotion_window(self) -> bool:
        """Inspect the sliding window of recent executions for demotion triggers.

        Returns ``True`` if demotion is warranted (too many reverts or
        manual corrections in the window).
        """
        cond = self._policy.demotion_conditions
        window_size: int = cond.get("window_size", 20)
        max_reverts: int = cond.get("max_reverts_in_window", 2)
        max_corrections: int = cond.get("max_corrections_in_window", 3)

        recent = list(self._history)[-window_size:]
        if len(recent) < 3:
            # Not enough data to demote
            return False

        reverts = sum(
            1 for _, fb in recent if fb in ("full_revert", "partial_revert")
        )
        corrections = sum(
            1 for _, fb in recent if fb == "manual_correction"
        )

        if reverts > max_reverts or corrections > max_corrections:
            log.warning(
                "autonomy_demotion",
                reverts=reverts,
                corrections=corrections,
                window=len(recent),
            )
            return True
        return False

    # -- Domain classification ----------------------------------------------

    def _classify_domain(self, command: str) -> TaskDomain:
        """Classify *command* into a :type:`TaskDomain` using regex heuristics."""
        lower = command.lower()
        for domain, pattern in _DOMAIN_PATTERNS:
            if pattern.search(lower):
                return domain
        return "unknown"

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _pattern_key(command: str, domain: TaskDomain) -> str:
        """Derive a stable pattern key from the command and domain.

        Strips arguments / paths to group similar commands together.
        """
        # Normalise: take the first two tokens of the command
        tokens = command.strip().split()
        base = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
        return f"{domain}:{base}"

    # -- Serialisation ------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialise internal state for persistence."""
        return {
            "patterns": {
                key: {
                    "total_executions": s.total_executions,
                    "approved": s.approved,
                    "reverted": s.reverted,
                    "manual_corrections": s.manual_corrections,
                    "avg_risk": s.avg_risk,
                    "last_used": s.last_used,
                }
                for key, s in self._patterns.items()
            },
            "history": list(self._history),
            "shadow_mode": self._shadow_mode,
            "kill_switch": self._kill_switch,
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> AutonomousExecutor:
        """Restore an executor from previously serialised *data*."""
        executor = cls()
        for key, raw in data.get("patterns", {}).items():
            executor._patterns[key] = PatternStats(
                total_executions=raw.get("total_executions", 0),
                approved=raw.get("approved", 0),
                reverted=raw.get("reverted", 0),
                manual_corrections=raw.get("manual_corrections", 0),
                avg_risk=raw.get("avg_risk", 0.0),
                last_used=raw.get("last_used", 0.0),
            )
        for entry in data.get("history", []):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                executor._history.append((entry[0], entry[1]))
        executor._shadow_mode = data.get("shadow_mode", False)
        executor._kill_switch = data.get("kill_switch", False)
        return executor


# Singleton

_executor_instance: AutonomousExecutor | None = None


def _load_policy_from_config() -> AutonomyPolicy | None:
    """Attempt to load an AutonomyPolicy from the RUNE config."""
    try:
        from rune.config import get_config

        config = get_config()
        proactive = getattr(config, "proactive", None)
        if proactive is None:
            return None
        # Start with defaults and override with config values
        policy = AutonomyPolicy(
            domain_levels=dict(DEFAULT_AUTONOMY_POLICY.domain_levels),
            risk_thresholds=dict(DEFAULT_AUTONOMY_POLICY.risk_thresholds),
            promotion_conditions={
                "min_approved": getattr(proactive, "autonomy_promotion_accepts", 3),
                "min_approval_rate": getattr(proactive, "autonomy_promotion_confidence", 0.7),
                "max_avg_risk": DEFAULT_AUTONOMY_POLICY.promotion_conditions.get("max_avg_risk", 0.4),
            },
            demotion_conditions=dict(DEFAULT_AUTONOMY_POLICY.demotion_conditions),
        )
        return policy
    except Exception:
        return None


def get_autonomous_executor(
    policy: AutonomyPolicy | None = None,
) -> AutonomousExecutor:
    """Return the global :class:`AutonomousExecutor` singleton."""
    global _executor_instance
    if _executor_instance is None:
        if policy is None:
            policy = _load_policy_from_config()
        _executor_instance = AutonomousExecutor(policy)
    return _executor_instance
