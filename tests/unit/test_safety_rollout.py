"""Tests for rune.memory.rollout_manager — memory policy rollout management."""


import pytest

from rune.memory.rollout_manager import (
    RolloutAutoConfig,
    RolloutManager,
    RolloutMetric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    """Provide a temp config path for RolloutManager."""
    return tmp_path / "rollout.json"


def make_manager(config_path, **kwargs) -> RolloutManager:
    return RolloutManager(config_path=config_path, **kwargs)


def make_metric(**kwargs) -> RolloutMetric:
    defaults = dict(
        event="task_completed",
        mode="balanced",
        intent_domain="code",
        conversation_id="conv_1",
        success=True,
        iterations=5,
    )
    defaults.update(kwargs)
    return RolloutMetric(**defaults)


# ---------------------------------------------------------------------------
# Tests: Mode management
# ---------------------------------------------------------------------------

class TestRolloutManagerMode:
    def test_default_mode_is_balanced(self, tmp_config):
        mgr = make_manager(tmp_config)
        assert mgr.get_mode() == "balanced"

    def test_set_and_get_mode(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.set_mode("strict")
        assert mgr.get_mode() == "strict"

    def test_set_mode_persists_to_disk(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.set_mode("shadow")
        # Create new manager reading same file
        mgr2 = make_manager(tmp_config)
        assert mgr2.get_mode() == "shadow"

    def test_env_var_overrides_config(self, tmp_config, monkeypatch):
        mgr = make_manager(tmp_config)
        mgr.set_mode("balanced")
        monkeypatch.setenv("RUNE_MEMORY_MODE", "legacy")
        assert mgr.get_mode() == "legacy"

    def test_invalid_env_var_falls_back_to_config(self, tmp_config, monkeypatch):
        mgr = make_manager(tmp_config)
        mgr.set_mode("strict")
        monkeypatch.setenv("RUNE_MEMORY_MODE", "invalid_mode")
        assert mgr.get_mode() == "strict"

    def test_set_invalid_mode_raises(self, tmp_config):
        mgr = make_manager(tmp_config)
        with pytest.raises(ValueError):
            mgr.set_mode("invalid")  # type: ignore


# ---------------------------------------------------------------------------
# Tests: Metrics (including new fields)
# ---------------------------------------------------------------------------

class TestRolloutManagerMetrics:
    def test_record_and_retrieve_metrics(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.record_metric(make_metric(event="e1"))
        mgr.record_metric(make_metric(event="e2"))
        metrics = mgr.get_metrics()
        assert len(metrics) == 2

    def test_metrics_filtered_by_since(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.record_metric(make_metric(
            event="old",
            timestamp="2025-01-01T00:00:00Z",
        ))
        mgr.record_metric(make_metric(
            event="new",
            timestamp="2026-06-01T00:00:00Z",
        ))
        metrics = mgr.get_metrics(since="2026-01-01T00:00:00Z")
        assert len(metrics) == 1
        assert metrics[0].event == "new"

    def test_new_metric_fields_roundtrip(self, tmp_config):
        """New fields (baseline_count, selected_count, shadow_agreement,
        compression_gain, retention_rate) persist and reload correctly."""
        mgr = make_manager(tmp_config)
        mgr.record_metric(RolloutMetric(
            event="retrieval",
            mode="shadow",
            baseline_count=100,
            selected_count=42,
            shadow_agreement=0.92,
            compression_gain=0.35,
            retention_rate=0.78,
        ))
        # Reload from disk
        mgr2 = make_manager(tmp_config)
        metrics = mgr2.get_metrics()
        assert len(metrics) == 1
        m = metrics[0]
        assert m.baseline_count == 100
        assert m.selected_count == 42
        assert m.shadow_agreement == pytest.approx(0.92)
        assert m.compression_gain == pytest.approx(0.35)
        assert m.retention_rate == pytest.approx(0.78)


# ---------------------------------------------------------------------------
# Tests: Auto-promotion & evaluate_promotion
# ---------------------------------------------------------------------------

class TestAutoPromotion:
    def test_does_not_promote_when_already_strict(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.set_mode("strict")
        assert mgr.should_auto_promote() is False
        assert mgr.evaluate_promotion() is None

    def test_does_not_promote_with_insufficient_events(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.set_mode("legacy")
        for _i in range(10):
            mgr.record_metric(make_metric(mode="legacy", success=True))
        assert mgr.should_auto_promote() is False

    def test_legacy_to_shadow_promotion(self, tmp_config):
        """legacy -> shadow when enough samples with high success rate."""
        mgr = make_manager(tmp_config)
        mgr.set_mode("legacy")
        for i in range(50):
            mgr.record_metric(make_metric(
                mode="legacy",
                success=(i < 45),  # 45/50 = 90% > 85%
            ))
        assert mgr.evaluate_promotion() == "shadow"
        assert mgr.should_auto_promote() is True

    def test_legacy_no_promotion_low_success(self, tmp_config):
        mgr = make_manager(tmp_config)
        mgr.set_mode("legacy")
        for i in range(50):
            mgr.record_metric(make_metric(
                mode="legacy",
                success=(i < 30),  # 30/50 = 60% < 85%
            ))
        assert mgr.evaluate_promotion() is None

    def test_shadow_to_balanced_promotion(self, tmp_config):
        """shadow -> balanced when shadow_agreement is high enough."""
        mgr = make_manager(tmp_config)
        mgr.set_mode("shadow")
        for _i in range(50):
            mgr.record_metric(RolloutMetric(
                event="retrieval",
                mode="shadow",
                shadow_agreement=0.90,
                success=True,
            ))
        assert mgr.evaluate_promotion() == "balanced"

    def test_shadow_no_promotion_low_agreement(self, tmp_config):
        """shadow does not promote with low shadow_agreement."""
        mgr = make_manager(tmp_config)
        mgr.set_mode("shadow")
        for _i in range(50):
            mgr.record_metric(RolloutMetric(
                event="retrieval",
                mode="shadow",
                shadow_agreement=0.50,
                success=True,
            ))
        assert mgr.evaluate_promotion() is None

    def test_balanced_to_strict_promotion(self, tmp_config):
        """balanced -> strict when retention_rate is high enough."""
        mgr = make_manager(tmp_config)
        mgr.set_mode("balanced")
        for _i in range(50):
            mgr.record_metric(RolloutMetric(
                event="outcome",
                mode="balanced",
                retention_rate=0.80,
                success=True,
            ))
        assert mgr.evaluate_promotion() == "strict"

    def test_balanced_no_promotion_low_retention(self, tmp_config):
        """balanced does not promote with low retention_rate."""
        mgr = make_manager(tmp_config)
        mgr.set_mode("balanced")
        for _i in range(50):
            mgr.record_metric(RolloutMetric(
                event="outcome",
                mode="balanced",
                retention_rate=0.50,
                success=True,
            ))
        assert mgr.evaluate_promotion() is None


# ---------------------------------------------------------------------------
# Tests: Auto-config parameters
# ---------------------------------------------------------------------------

class TestAutoConfig:
    def test_default_auto_config(self, tmp_config):
        mgr = make_manager(tmp_config)
        cfg = mgr.auto_config
        assert cfg.observation_window_days == 7
        assert cfg.min_shadow_samples == 50
        assert cfg.promote_shadow_min_agreement == pytest.approx(0.85)
        assert cfg.promote_balanced_min_retention == pytest.approx(0.7)

    def test_custom_auto_config(self, tmp_config):
        custom = RolloutAutoConfig(
            observation_window_days=14,
            min_shadow_samples=20,
            promote_shadow_min_agreement=0.90,
            promote_balanced_min_retention=0.80,
        )
        mgr = make_manager(tmp_config, auto_config=custom)
        cfg = mgr.auto_config
        assert cfg.observation_window_days == 14
        assert cfg.min_shadow_samples == 20
        assert cfg.promote_shadow_min_agreement == pytest.approx(0.90)
        assert cfg.promote_balanced_min_retention == pytest.approx(0.80)
