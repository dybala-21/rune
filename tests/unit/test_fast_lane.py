"""Simple-query fast lane entry rules.

Only high-confidence chat/web goals may enter; everything else stays on
the primary model. See docs/design/simple-query-fast-path.md.
"""

from __future__ import annotations

import pytest

from rune.agent.fast_lane import (
    FAST_LANE_TOOL_ROUNDS,
    decide_fast_lane,
)
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import _compute_tool_rounds
from rune.config.loader import get_config


def _classification(goal_type: str, confidence: float) -> ClassificationResult:
    return ClassificationResult(
        goal_type=goal_type,  # type: ignore[arg-type]
        confidence=confidence,
        tier=2,
        reason="test",
    )


@pytest.fixture()
def anthropic_session(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "active_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "active_model", "claude-opus-4-6")
    monkeypatch.setattr(cfg.llm, "route_simple_queries", True)
    monkeypatch.setattr(cfg.llm, "simple_query_tier", "fast")
    monkeypatch.setattr(cfg.llm, "simple_query_confidence", 0.8)
    return cfg


def test_simple_web_goal_enters_lane_with_fast_model(anthropic_session):
    decision = decide_fast_lane(_classification("web", 0.9))
    assert decision.active
    # Derived from config, not hardcoded — CI configs may override tiers.
    assert decision.model == f"anthropic/{get_config().llm.models.anthropic.fast}"


def test_chat_goal_enters_lane(anthropic_session):
    assert decide_fast_lane(_classification("chat", 0.95)).active


@pytest.mark.parametrize(
    "goal_type", ["code_modify", "execution", "research", "browser", "full"]
)
def test_non_simple_goals_never_enter(anthropic_session, goal_type):
    decision = decide_fast_lane(_classification(goal_type, 0.99))
    assert not decision.active
    assert decision.reason.startswith("goal_type:")


def test_low_confidence_stays_on_primary(anthropic_session):
    decision = decide_fast_lane(_classification("web", 0.79))
    assert not decision.active
    assert decision.reason.startswith("confidence:")


def test_kill_switch_disables_lane(anthropic_session, monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "route_simple_queries", False)
    decision = decide_fast_lane(_classification("web", 0.99))
    assert not decision.active
    assert decision.reason == "disabled"


def test_openai_model_string_has_no_provider_prefix(anthropic_session, monkeypatch):
    # The loop uses bare model ids for openai and provider/model otherwise;
    # the lane must produce the same format or LiteLLM routing breaks.
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "active_provider", "openai")
    monkeypatch.setattr(cfg.llm, "active_model", None)
    decision = decide_fast_lane(_classification("web", 0.9))
    assert decision.active
    assert "/" not in decision.model
    assert decision.model == cfg.llm.models.openai.fast


def test_invalid_tier_falls_back_to_fast(anthropic_session, monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "simple_query_tier", "not-a-tier")
    decision = decide_fast_lane(_classification("web", 0.9))
    assert decision.active
    assert decision.model.endswith(cfg.llm.models.anthropic.fast)


def test_fast_lane_tool_rounds_bypass_the_floor():
    # _compute_tool_rounds has a 12-round floor; the lane needs ~3.
    classification = _classification("web", 0.9)
    normal = _compute_tool_rounds(classification, "anthropic/claude-opus-4-6", False)
    lane = _compute_tool_rounds(
        classification, "anthropic/claude-haiku-4-5-20251001", False, fast_lane=True
    )
    assert normal >= 12
    assert lane == FAST_LANE_TOOL_ROUNDS


def test_boundary_confidence_exactly_at_threshold_enters(anthropic_session):
    # The threshold is inclusive.
    assert decide_fast_lane(_classification("web", 0.8)).active


def test_classifier_fallback_confidences_stay_out(anthropic_session):
    # 0.7 = missing-confidence default, 0.5 = exception fallback,
    # 0.75 = LLM-intent fallback (loop.py). All below the threshold by design.
    for conf in (0.7, 0.5, 0.75):
        assert not decide_fast_lane(_classification("chat", conf)).active


def test_broken_config_fails_open_to_primary(anthropic_session, monkeypatch):
    # A config failure must not kill the run — no lane, primary model.
    import rune.agent.fast_lane as fl

    def _boom():
        raise RuntimeError("config exploded")

    monkeypatch.setattr("rune.config.loader.get_config", _boom)
    decision = fl.decide_fast_lane(_classification("web", 0.95))
    assert not decision.active
    assert decision.reason == "error"


def test_garbage_classification_object_is_rejected(anthropic_session):
    # Duck-typed classification with missing/absurd fields must not crash.
    class _Garbage:
        pass

    assert not decide_fast_lane(_Garbage()).active
    assert not decide_fast_lane(_classification("", 0.99)).active


def test_lane_fetch_cap_shrinks_default_and_respects_smaller():
    from rune.agent.tool_adapter import (
        _FAST_LANE_FETCH_MAX_CHARS,
        ToolAdapterOptions,
        _cap_fast_lane_fetch,
    )

    # Default 20k page dump shrinks to the lane cap.
    capped = _cap_fast_lane_fetch({"url": "https://x.test"})
    assert capped["maxLength"] == _FAST_LANE_FETCH_MAX_CHARS
    assert capped["max_length"] == _FAST_LANE_FETCH_MAX_CHARS
    # An explicit smaller request is left alone.
    small = _cap_fast_lane_fetch({"url": "https://x.test", "maxLength": 3000})
    assert small["maxLength"] == 3000
    # The option is off by default — no lane, no capping call.
    assert ToolAdapterOptions().fast_lane_active is None


def test_web_fetch_cache_key_varies_by_maxlength():
    # A lane-capped (8k) fetch must not be served to a later full-length
    # request for the same URL.
    from rune.agent.cognitive_cache import SessionToolCache

    cache = SessionToolCache()
    capped = cache.generate_key("web_fetch", {"url": "https://x.test", "maxLength": 8000})
    full = cache.generate_key("web_fetch", {"url": "https://x.test"})
    assert capped != full
    assert cache.generate_key(
        "web_fetch", {"url": "https://x.test", "maxLength": 8000}
    ) == capped


def test_web_fetch_cache_key_varies_by_selector():
    from rune.agent.cognitive_cache import SessionToolCache

    cache = SessionToolCache()
    page = cache.generate_key("web_fetch", {"url": "https://x.test"})
    part = cache.generate_key("web_fetch", {"url": "https://x.test", "selector": ".price"})
    assert page != part


def test_web_fetch_cache_key_default_matches_capability():
    # The key's implicit default must track WebFetchParams.max_length, or
    # default-length fetches collide with a different explicit length.
    from rune.agent.cognitive_cache import (
        WEB_FETCH_DEFAULT_MAX_LENGTH,
        SessionToolCache,
    )
    from rune.capabilities.web import WebFetchParams

    assert WebFetchParams.model_fields["max_length"].default == WEB_FETCH_DEFAULT_MAX_LENGTH
    cache = SessionToolCache()
    implicit = cache.generate_key("web_fetch", {"url": "https://x.test"})
    explicit = cache.generate_key(
        "web_fetch", {"url": "https://x.test", "maxLength": WEB_FETCH_DEFAULT_MAX_LENGTH}
    )
    assert implicit == explicit
    # Falsy zero is a real value, not a missing one.
    zero = cache.generate_key("web_fetch", {"url": "https://x.test", "maxLength": 0})
    assert zero != implicit


def test_lane_fetch_cap_respects_explicit_zero():
    from rune.agent.tool_adapter import _cap_fast_lane_fetch

    assert _cap_fast_lane_fetch({"url": "https://x.test", "maxLength": 0})["maxLength"] == 0


def test_cache_hit_hint_matches_capability_params():
    # The web_fetch hint must only suggest parameters that change its key.
    from rune.agent.cognitive_cache import SessionToolCache

    cache = SessionToolCache()
    params = {"url": "https://x.test"}
    key = cache.generate_key("web_fetch", params)

    class _R:
        output = "content " * 50
        success = True

    cache.set(key, "web_fetch", params, _R(), step_number=1)
    hit = cache.get(key, "web_fetch", params)
    assert "maxLength" in hit.output
    assert "offset" not in hit.output


def test_invalid_tier_rejected_at_config_boundary():
    import pydantic
    import pytest as _pytest

    from rune.config.schema import LLMConfig

    with _pytest.raises(pydantic.ValidationError):
        LLMConfig.model_validate({"simpleQueryTier": "fastest"})


def test_confidence_out_of_range_is_rejected():
    import pydantic
    import pytest as _pytest

    from rune.config.schema import LLMConfig

    with _pytest.raises(pydantic.ValidationError):
        LLMConfig.model_validate({"simpleQueryConfidence": 1.5})


def test_config_defaults_are_on_and_reversible():
    from rune.config.schema import LLMConfig

    llm = LLMConfig()
    assert llm.route_simple_queries is True
    assert llm.simple_query_tier == "fast"
    assert llm.simple_query_confidence == 0.8
    # camelCase aliases must round-trip (user config files are camelCase).
    llm2 = LLMConfig.model_validate(
        {"routeSimpleQueries": False, "simpleQueryTier": "best", "simpleQueryConfidence": 0.9}
    )
    assert llm2.route_simple_queries is False
    assert llm2.simple_query_tier == "best"
    assert llm2.simple_query_confidence == 0.9
