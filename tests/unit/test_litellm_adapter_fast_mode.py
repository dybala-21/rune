"""Fast mode must never be asked of a model that cannot do it.

`speed` is not advisory. A model without support rejects the entire request
with a 400, so a flag meant to make runs quicker instead makes every round
fail — and because the run then ends early having done nothing, the
wall-clock reads like a large improvement.
"""

from __future__ import annotations

import pytest

from rune.agent.litellm_adapter import _supports_speed


class TestSupportsSpeed:
    @pytest.mark.parametrize("model", [
        "claude-opus-5", "claude-opus-4-8", "anthropic/claude-opus-5",
    ])
    def test_opus_class_models_may_be_asked(self, model):
        assert _supports_speed(model)

    @pytest.mark.parametrize("model", [
        "claude-haiku-4-5-20251001",   # measured: 400 on every round
        "claude-sonnet-5",
        "claude-3-5-haiku-20241022",
    ])
    def test_everything_else_is_left_alone(self, model):
        assert not _supports_speed(model)

    @pytest.mark.parametrize("model", [
        "gpt-4o", "gemini-2.5-pro", "qwen2.5-coder:32b", "",
    ])
    def test_other_providers_never_get_an_anthropic_parameter(self, model):
        assert not _supports_speed(model)
