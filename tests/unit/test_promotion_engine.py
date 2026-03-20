"""Tests for rune.memory.promotion_engine — episode promotion and quality scoring."""



from rune.memory.promotion_engine import (
    EpisodeCandidate,
    calculate_episode_quality,
    compact_episode_for_mode,
    promote_episode_candidates,
)
from rune.memory.store import Episode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_episode(id: str, **kwargs) -> Episode:
    defaults = dict(
        id=id,
        timestamp="2026-02-18T00:00:00+00:00",
        task_summary=f"Episode {id} summary",
        intent="process",
        plan=f"Plan {id}",
        result="success completed",
        lessons="lesson1\nlesson2\nlesson3",
        importance=0.7,
    )
    defaults.update(kwargs)
    return Episode(**defaults)


# ---------------------------------------------------------------------------
# Tests: promote_episode_candidates
# ---------------------------------------------------------------------------

class TestPromoteEpisodeCandidates:
    def test_shadow_mode_keeps_baseline_unchanged(self):
        candidates = [
            EpisodeCandidate(episode=make_episode("a"), score=0.86),
            EpisodeCandidate(episode=make_episode("b"), score=0.63),
            EpisodeCandidate(episode=make_episode("c"), score=0.44),
        ]
        result = promote_episode_candidates(candidates, "shadow", 3)
        ids = [c.episode.id for c in result.selected]
        assert ids == ["a", "b", "c"]
        assert result.diagnostics.shadow_agreement == 1.0
        assert result.diagnostics.compression_gain == 0.0

    def test_legacy_mode_keeps_baseline_unchanged(self):
        candidates = [
            EpisodeCandidate(episode=make_episode("a"), score=0.9),
            EpisodeCandidate(episode=make_episode("b"), score=0.5),
        ]
        result = promote_episode_candidates(candidates, "legacy", 2)
        ids = [c.episode.id for c in result.selected]
        assert ids == ["a", "b"]

    def test_balanced_mode_preserves_high_quality(self):
        candidates = [
            EpisodeCandidate(
                episode=make_episode("core", importance=0.9, lessons="fix\nverify\ntest\nregression\ndocs"),
                score=0.9,
            ),
            EpisodeCandidate(
                episode=make_episode("secondary", importance=0.75, lessons="module\nrefactor\nqa"),
                score=0.72,
            ),
            EpisodeCandidate(
                episode=make_episode("noise", importance=0.95, result="failed", lessons=""),
                score=0.46,
            ),
        ]
        result = promote_episode_candidates(candidates, "balanced", 3)
        ids = [c.episode.id for c in result.selected]
        assert ids[0] == "core"

    def test_strict_mode_drops_low_quality(self):
        candidates = [
            EpisodeCandidate(
                episode=make_episode("core", importance=0.85, lessons="e1\ne2\ne3"),
                score=0.88,
            ),
            EpisodeCandidate(
                episode=make_episode("borderline", importance=1.0, result="failed", lessons=""),
                score=0.46,
            ),
            EpisodeCandidate(
                episode=make_episode("too-low", importance=0.8, result="success completed", lessons="x"),
                score=0.41,
            ),
        ]
        result = promote_episode_candidates(candidates, "strict", 3)
        ids = [c.episode.id for c in result.selected]
        assert "core" in ids
        assert result.diagnostics.compression_gain > 0.6

    def test_empty_baseline_returns_empty(self):
        result = promote_episode_candidates([], "balanced", 5)
        assert len(result.selected) == 0

    def test_limit_caps_results(self):
        candidates = [
            EpisodeCandidate(episode=make_episode(f"e{i}"), score=0.9 - i * 0.1)
            for i in range(10)
        ]
        result = promote_episode_candidates(candidates, "shadow", 3)
        assert len(result.selected) <= 3


# ---------------------------------------------------------------------------
# Tests: compact_episode_for_mode
# ---------------------------------------------------------------------------

class TestCompactEpisodeForMode:
    def test_legacy_mode_unchanged(self):
        ep = make_episode("long", task_summary="x" * 500)
        compact = compact_episode_for_mode(ep, "legacy")
        assert compact.task_summary == ep.task_summary

    def test_shadow_mode_unchanged(self):
        ep = make_episode("long", task_summary="x" * 500)
        compact = compact_episode_for_mode(ep, "shadow")
        assert compact.task_summary == ep.task_summary

    def test_strict_mode_truncates_summary(self):
        ep = make_episode("long", task_summary="x" * 500)
        compact = compact_episode_for_mode(ep, "strict")
        assert len(compact.task_summary) <= 120

    def test_balanced_mode_truncates_summary(self):
        ep = make_episode("long", task_summary="x" * 500)
        compact = compact_episode_for_mode(ep, "balanced")
        assert len(compact.task_summary) <= 180

    def test_short_summary_unchanged_in_strict(self):
        ep = make_episode("short", task_summary="short summary")
        compact = compact_episode_for_mode(ep, "strict")
        assert compact.task_summary == "short summary"


# ---------------------------------------------------------------------------
# Tests: calculate_episode_quality
# ---------------------------------------------------------------------------

class TestCalculateEpisodeQuality:
    def test_quality_between_0_and_1(self):
        ep = make_episode("test")
        quality = calculate_episode_quality(ep)
        assert 0.0 <= quality <= 1.0

    def test_successful_episode_higher_quality(self):
        robust = make_episode("robust", importance=0.8, result="success completed", lessons="l1\nl2\nl3")
        noisy = make_episode("noisy", importance=1.0, result="failed error", lessons="")
        assert calculate_episode_quality(robust) > calculate_episode_quality(noisy)

    def test_high_importance_contributes_to_quality(self):
        high_imp = make_episode("hi", importance=1.0)
        low_imp = make_episode("lo", importance=0.1)
        assert calculate_episode_quality(high_imp) > calculate_episode_quality(low_imp)

    def test_more_lessons_contribute_to_quality(self):
        many_lessons = make_episode("many", lessons="a\nb\nc\nd\ne")
        no_lessons = make_episode("none", lessons="")
        assert calculate_episode_quality(many_lessons) > calculate_episode_quality(no_lessons)

    def test_empty_episode_has_low_quality(self):
        ep = Episode(importance=0.0, result="", lessons="", task_summary="")
        quality = calculate_episode_quality(ep)
        assert quality < 0.3
