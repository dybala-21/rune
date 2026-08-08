"""The model-traits table: one place for what a model family accepts.

These tests pin the matching rules — first row wins, specific families
above general ones — so a future row edit can't silently reorder them.
"""

from __future__ import annotations

from rune.agent.model_traits import (
    _TEMPERATURE_REJECTED,
    ModelTraits,
    note_temperature_rejected,
    traits,
)


def test_unknown_models_get_bare_defaults():
    assert traits("gemini/gemini-2.5-pro") == ModelTraits()
    assert traits("ollama/qwen2.5-coder:7b") == ModelTraits()
    assert traits("") == ModelTraits()


def test_specific_family_wins_over_general():
    # opus rows sit above the bare claude row; both must keep anthropic_wire.
    opus = traits("anthropic/claude-opus-5")
    assert opus.speed_param and opus.anthropic_wire
    haiku = traits("claude-haiku-4-5")
    assert haiku.anthropic_wire and not haiku.speed_param


def test_opus_requires_the_anthropic_family():
    # A non-Anthropic model with "opus" in its name must not inherit
    # Anthropic-only parameters.
    assert not traits("magnum-opus-7b").speed_param


def test_gpt5_family_never_gets_temperature():
    assert not traits("gpt-5.5").temperature
    assert not traits("openai/gpt-5-mini").temperature


def test_learned_overlay_matches_exact_id_only():
    _TEMPERATURE_REJECTED.discard("claude-opus-4-8")
    try:
        assert traits("claude-opus-4-8").temperature
        note_temperature_rejected("claude-opus-4-8")
        assert not traits("claude-opus-4-8").temperature
        # The overlay is per resolved id, not per family.
        assert traits("claude-opus-4-6").temperature
    finally:
        _TEMPERATURE_REJECTED.discard("claude-opus-4-8")
