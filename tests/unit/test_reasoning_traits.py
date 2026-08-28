"""reasoning_effort gate follows litellm's per-model capability, not a hand-list.

The point of using litellm.supports_reasoning is that support differs per model
in ways a static list gets wrong: o1 takes an effort but o1-mini doesn't,
opus-4-5 reasons while opus-4 doesn't. These assert the distinctions that a
naive substring list would miss.
"""
from rune.agent.model_traits import supports_reasoning_effort


def test_reasoning_on_for_reasoning_models():
    for m in ["gpt-5.6-sol", "gpt-5.4", "o1", "o3", "o4-mini",
              "claude-opus-4-6", "claude-opus-5", "claude-opus-4-5"]:
        assert supports_reasoning_effort(m) is True, m


def test_reasoning_off_where_a_hand_list_would_get_it_wrong():
    # o1-mini takes no effort though it shares the "o1" prefix; opus-4 predates
    # thinking; plain chat models don't reason.
    for m in ["o1-mini", "claude-opus-4", "gpt-4o", "gpt-4o-mini", "ollama/llama3.2"]:
        assert supports_reasoning_effort(m) is False, m


def test_unknown_model_defaults_false():
    assert supports_reasoning_effort("totally-made-up-model-xyz") is False
