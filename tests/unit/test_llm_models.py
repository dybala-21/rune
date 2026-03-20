"""Tests for rune.llm.models — model filtering, sorting, and registry."""

from __future__ import annotations

from rune.llm.models import (
    ANTHROPIC_MODELS,
    FALLBACK_OPENAI_MODELS,
    _is_chat_model,
    _model_sort_key,
    invalidate_cache,
)

# ---------------------------------------------------------------------------
# _is_chat_model filtering
# ---------------------------------------------------------------------------


class TestIsChatModel:
    def test_includes_gpt5(self):
        assert _is_chat_model("gpt-5.2") is True

    def test_includes_gpt4(self):
        assert _is_chat_model("gpt-4o-2024") is True

    def test_includes_gpt41(self):
        assert _is_chat_model("gpt-4.1-nano") is True

    def test_includes_o1(self):
        assert _is_chat_model("o1-preview") is True

    def test_includes_o3(self):
        assert _is_chat_model("o3-mini") is True

    def test_includes_chatgpt(self):
        assert _is_chat_model("chatgpt-4o") is True

    def test_excludes_instruct(self):
        assert _is_chat_model("gpt-4-instruct") is False

    def test_excludes_realtime(self):
        assert _is_chat_model("gpt-4o-realtime-preview") is False

    def test_excludes_audio(self):
        assert _is_chat_model("gpt-4o-audio") is False

    def test_excludes_tts(self):
        assert _is_chat_model("tts-1") is False

    def test_excludes_dalle(self):
        assert _is_chat_model("dall-e-3") is False

    def test_excludes_embedding(self):
        assert _is_chat_model("text-embedding-3-small") is False

    def test_excludes_whisper(self):
        assert _is_chat_model("whisper-1") is False

    def test_excludes_search(self):
        assert _is_chat_model("gpt-4-search") is False

    def test_excludes_transcription(self):
        assert _is_chat_model("gpt-4-transcription") is False

    def test_excludes_unrecognized(self):
        assert _is_chat_model("text-davinci-003") is False
        assert _is_chat_model("babbage-002") is False


# ---------------------------------------------------------------------------
# _model_sort_key ordering
# ---------------------------------------------------------------------------


class TestModelSortKey:
    def test_gpt54_highest_priority(self):
        assert _model_sort_key("gpt-5.4") == 0

    def test_gpt52_variant(self):
        assert _model_sort_key("gpt-5.2") == 1

    def test_gpt51_variant(self):
        assert _model_sort_key("gpt-5.1-codex") == 2

    def test_gpt5_variant(self):
        assert _model_sort_key("gpt-5-turbo") == 3

    def test_o4_mini(self):
        assert _model_sort_key("o4-mini") == 6

    def test_o3_pro(self):
        assert _model_sort_key("o3-pro") == 7

    def test_o3_mini(self):
        assert _model_sort_key("o3-mini") == 8

    def test_o1(self):
        assert _model_sort_key("o1") == 9

    def test_gpt41(self):
        assert _model_sort_key("gpt-4.1-nano") == 10

    def test_gpt4o(self):
        assert _model_sort_key("gpt-4o-latest") == 11

    def test_gpt4(self):
        assert _model_sort_key("gpt-4-turbo") == 12

    def test_gpt35(self):
        assert _model_sort_key("gpt-3.5-turbo") == 15

    def test_sort_order(self):
        model_ids = [
            "gpt-3.5-turbo", "o4-mini", "gpt-4o-latest", "gpt-4.1-nano",
            "o3-mini", "o3-pro", "gpt-5-turbo", "gpt-5.1-codex",
            "gpt-5.2", "gpt-5.4",
        ]
        sorted_ids = sorted(model_ids, key=_model_sort_key)
        assert sorted_ids == [
            "gpt-5.4",
            "gpt-5.2",
            "gpt-5.1-codex",
            "gpt-5-turbo",
            "o4-mini",
            "o3-pro",
            "o3-mini",
            "gpt-4.1-nano",
            "gpt-4o-latest",
            "gpt-3.5-turbo",
        ]


# ---------------------------------------------------------------------------
# Fallback / Anthropic model lists
# ---------------------------------------------------------------------------


class TestModelLists:
    def test_fallback_openai_has_entries(self):
        assert len(FALLBACK_OPENAI_MODELS) >= 10
        assert FALLBACK_OPENAI_MODELS[0].id == "gpt-5.4"
        assert all(m.provider == "openai" for m in FALLBACK_OPENAI_MODELS)

    def test_anthropic_models_have_entries(self):
        assert len(ANTHROPIC_MODELS) >= 3
        assert all(m.provider == "anthropic" for m in ANTHROPIC_MODELS)

    def test_invalidate_cache_does_not_raise(self):
        invalidate_cache()  # should not raise
