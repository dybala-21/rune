"""Memory policy rollout manager.

Ported from src/memory/rollout-manager.ts - controls memory system
rollout phases (legacy/shadow/balanced/strict) and collects metrics for
automated mode promotion decisions.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


# Types

MemoryPolicyMode = Literal["legacy", "shadow", "balanced", "strict"]

_VALID_MODES: frozenset[str] = frozenset({"legacy", "shadow", "balanced", "strict"})

_MODE_ORDER: list[MemoryPolicyMode] = ["legacy", "shadow", "balanced", "strict"]

_MODE_ENV_VAR = "RUNE_MEMORY_MODE"


@dataclass(slots=True)
class RolloutMetric:
    """A single recorded event for rollout analysis."""

    event: str = ""
    mode: str = ""
    intent_domain: str = ""
    conversation_id: str = ""
    success: bool = False
    iterations: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    # Additional metric fields (aligned with TS MemoryPolicyMetricEntry)
    baseline_count: int = 0
    selected_count: int = 0
    shadow_agreement: float = 0.0
    compression_gain: float = 0.0
    retention_rate: float = 0.0


@dataclass(slots=True)
class RolloutAutoConfig:
    """Auto-promotion configuration parameters (from TS MemoryRolloutAutoConfig)."""

    observation_window_days: int = 7
    min_shadow_samples: int = 50
    promote_shadow_min_agreement: float = 0.85
    promote_balanced_min_retention: float = 0.7


# Serialisation helpers

def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp string to a timezone-aware datetime."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _metric_to_dict(m: RolloutMetric) -> dict[str, Any]:
    return asdict(m)


def _dict_to_metric(d: dict[str, Any]) -> RolloutMetric:
    return RolloutMetric(
        event=d.get("event", ""),
        mode=d.get("mode", ""),
        intent_domain=d.get("intent_domain", ""),
        conversation_id=d.get("conversation_id", ""),
        success=d.get("success", False),
        iterations=d.get("iterations", 0),
        timestamp=d.get("timestamp", ""),
        baseline_count=d.get("baseline_count", 0),
        selected_count=d.get("selected_count", 0),
        shadow_agreement=d.get("shadow_agreement", 0.0),
        compression_gain=d.get("compression_gain", 0.0),
        retention_rate=d.get("retention_rate", 0.0),
    )


# RolloutManager

class RolloutManager:
    """Controls memory policy mode and collects rollout metrics."""

    _DEFAULT_MODE: MemoryPolicyMode = "balanced"

    def __init__(
        self,
        config_path: str | Path | None = None,
        auto_config: RolloutAutoConfig | None = None,
    ) -> None:
        if config_path is None:
            config_path = rune_home() / "rollout.json"
        self._config_path = Path(config_path)
        self._lock = threading.Lock()
        self._config: dict[str, Any] | None = None
        self._auto_config = auto_config or RolloutAutoConfig()

    @property
    def auto_config(self) -> RolloutAutoConfig:
        return self._auto_config

    def _load_config(self) -> dict[str, Any]:
        """Load config from disk, with caching."""
        if self._config is not None:
            return self._config

        if self._config_path.exists():
            try:
                raw = self._config_path.read_text(encoding="utf-8")
                self._config = json_decode(raw)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("rollout_config_load_failed", error=str(exc))
                self._config = {}
        else:
            self._config = {}

        return self._config

    def _save_config(self, config: dict[str, Any]) -> None:
        """Persist config to disk."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._config_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._config_path)
            self._config = config
        except OSError as exc:
            log.error("rollout_config_save_failed", error=str(exc))
            tmp.unlink(missing_ok=True)
            raise

    def get_mode(self) -> MemoryPolicyMode:
        """Determine the current memory policy mode.

        Priority: env var > config file > default ("balanced").
        """
        # 1. Environment variable override
        env_mode = os.environ.get(_MODE_ENV_VAR, "").strip().lower()
        if env_mode in _VALID_MODES:
            return env_mode  # type: ignore[return-value]

        # 2. Persisted config
        with self._lock:
            config = self._load_config()
            config_mode = config.get("mode", "").strip().lower()
            if config_mode in _VALID_MODES:
                return config_mode  # type: ignore[return-value]

        # 3. Default
        return self._DEFAULT_MODE

    def set_mode(self, mode: MemoryPolicyMode) -> None:
        """Persist a new memory policy mode."""
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid memory policy mode: {mode!r}")

        with self._lock:
            config = self._load_config()
            config["mode"] = mode
            self._save_config(config)
            log.info("rollout_mode_set", mode=mode)

    def record_metric(self, metric: RolloutMetric) -> None:
        """Append a metric to the rollout log."""
        if not metric.timestamp:
            metric.timestamp = datetime.now(UTC).isoformat()

        with self._lock:
            config = self._load_config()
            metrics_list: list[dict[str, Any]] = config.get("metrics", [])
            metrics_list.append(_metric_to_dict(metric))
            config["metrics"] = metrics_list
            self._save_config(config)
            log.debug("rollout_metric_recorded", metric_event=metric.event)

    def get_metrics(self, since: str = "") -> list[RolloutMetric]:
        """Retrieve recorded metrics, optionally filtered by timestamp.

        *since* is an ISO timestamp string; only metrics at or after that
        time are returned.
        """
        with self._lock:
            config = self._load_config()

        raw_metrics: list[dict[str, Any]] = config.get("metrics", [])
        metrics = [_dict_to_metric(m) for m in raw_metrics]

        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None
            if since_dt:
                metrics = [m for m in metrics if _parse_ts(m.timestamp) >= since_dt]

        return metrics

    def _get_windowed_metrics(self) -> list[RolloutMetric]:
        """Return metrics within the observation window."""
        cutoff = datetime.now(UTC) - timedelta(
            days=self._auto_config.observation_window_days
        )
        return self.get_metrics(since=cutoff.isoformat())

    def should_auto_promote(self) -> bool:
        """Decide whether the current mode should be auto-promoted.

        Delegates to :meth:`evaluate_promotion` and returns ``True``
        if a promotion target exists.
        """
        return self.evaluate_promotion() is not None

    def evaluate_promotion(self) -> MemoryPolicyMode | None:
        """Check if metrics justify upgrading from current mode to the next.

        Promotion chain: legacy -> shadow -> balanced -> strict.

        Returns the target mode if promotion is warranted, or ``None``.
        """
        current_mode = self.get_mode()

        # Already at max
        if current_mode == "strict":
            return None

        current_idx = _MODE_ORDER.index(current_mode)
        next_mode = _MODE_ORDER[current_idx + 1]

        metrics = self._get_windowed_metrics()
        mode_metrics = [m for m in metrics if m.mode == current_mode]

        if not mode_metrics:
            return None

        # --- legacy -> shadow ---
        # Promotion from legacy requires a minimum number of samples.
        if current_mode == "legacy":
            if len(mode_metrics) >= self._auto_config.min_shadow_samples:
                success_count = sum(1 for m in mode_metrics if m.success)
                success_rate = success_count / len(mode_metrics)
                if success_rate >= self._auto_config.promote_shadow_min_agreement:
                    log.info(
                        "evaluate_promotion",
                        current=current_mode,
                        target=next_mode,
                        samples=len(mode_metrics),
                        success_rate=round(success_rate, 3),
                    )
                    return next_mode
            return None

        # --- shadow -> balanced ---
        # Requires min_shadow_samples and shadow_agreement >= threshold.
        if current_mode == "shadow":
            if len(mode_metrics) < self._auto_config.min_shadow_samples:
                return None
            agreement_values = [
                m.shadow_agreement for m in mode_metrics if m.shadow_agreement > 0
            ]
            if not agreement_values:
                return None
            avg_agreement = sum(agreement_values) / len(agreement_values)
            if avg_agreement >= self._auto_config.promote_shadow_min_agreement:
                log.info(
                    "evaluate_promotion",
                    current=current_mode,
                    target=next_mode,
                    samples=len(mode_metrics),
                    avg_agreement=round(avg_agreement, 3),
                )
                return next_mode
            return None

        # --- balanced -> strict ---
        # Requires sufficient samples and retention_rate >= threshold.
        if current_mode == "balanced":
            if len(mode_metrics) < self._auto_config.min_shadow_samples:
                return None
            retention_values = [
                m.retention_rate for m in mode_metrics if m.retention_rate > 0
            ]
            if not retention_values:
                return None
            avg_retention = sum(retention_values) / len(retention_values)
            if avg_retention >= self._auto_config.promote_balanced_min_retention:
                log.info(
                    "evaluate_promotion",
                    current=current_mode,
                    target=next_mode,
                    samples=len(mode_metrics),
                    avg_retention=round(avg_retention, 3),
                )
                return next_mode
            return None

        return None


# Convenience functions

def append_memory_policy_metric(metric: RolloutMetric) -> None:
    """Convenience: record a metric via the singleton manager."""
    get_rollout_manager().record_metric(metric)


# Module singleton

_manager: RolloutManager | None = None


def get_rollout_manager() -> RolloutManager:
    global _manager
    if _manager is None:
        _manager = RolloutManager()
    return _manager
