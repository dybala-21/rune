"""reasoning_effort trait must be on only for models that accept it."""
from rune.agent.model_traits import traits


def test_reasoning_on_for_gpt5_and_o_series():
    for m in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.4", "gpt-5-mini", "o1", "o3", "o4-mini"]:
        assert traits(m).reasoning_effort is True, m


def test_reasoning_on_for_claude_46_and_5():
    for m in ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-5", "claude-sonnet-5"]:
        assert traits(m).reasoning_effort is True, m


def test_reasoning_off_for_older_claude_and_nonreasoning():
    # opus-4-5 / opus-4 predate adaptive thinking; gpt-4o and local models don't reason.
    for m in ["claude-opus-4-5-20251101", "claude-opus-4", "claude-sonnet-4-20250514",
              "gpt-4o", "gpt-4o-mini", "llama3.2"]:
        assert traits(m).reasoning_effort is False, m
