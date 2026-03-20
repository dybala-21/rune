"""Tests for rune.voice.availability — voice input capability probing."""

from __future__ import annotations

import os
from unittest.mock import patch

from rune.voice.availability import (
    _reset_cache,
    get_voice_availability,
    get_voice_install_hint,
    is_voice_input_available,
)


class TestVoiceAvailability:
    def setup_method(self):
        _reset_cache()

    def teardown_method(self):
        _reset_cache()
        os.environ.pop("DEEPGRAM_API_KEY", None)

    def test_reset_cache_returns_consistent_state(self):
        info = get_voice_availability()
        assert isinstance(info.mic_available, bool)
        assert isinstance(info.provider_available, bool)
        assert isinstance(info.available, bool)

    def test_deepgram_detected_when_api_key_set(self):
        os.environ["DEEPGRAM_API_KEY"] = "test-key-123"
        _reset_cache()
        info = get_voice_availability()
        assert info.provider_available is True
        assert info.active_provider == "deepgram"

    def test_no_provider_when_no_key(self):
        os.environ.pop("DEEPGRAM_API_KEY", None)
        _reset_cache()

        # Also mock sherpa_onnx to be unavailable
        with patch("rune.voice.availability.importlib.import_module", side_effect=ImportError):
            _reset_cache()
            info = get_voice_availability()
            assert info.active_provider != "deepgram" or info.active_provider is None

    def test_install_hint_matches_availability(self):
        hint = get_voice_install_hint()
        info = get_voice_availability()
        assert hint == info.install_hint

    def test_is_voice_input_available_consistent(self):
        available = is_voice_input_available()
        info = get_voice_availability()
        assert available == info.available

    def test_install_hint_mentions_dependencies(self):
        os.environ.pop("DEEPGRAM_API_KEY", None)
        with patch("rune.voice.availability.importlib.import_module", side_effect=ImportError):
            _reset_cache()
            info = get_voice_availability()
            if info.install_hint:
                assert (
                    "sounddevice" in info.install_hint
                    or "pyaudio" in info.install_hint
                    or "DEEPGRAM_API_KEY" in info.install_hint
                    or "sherpa" in info.install_hint
                )
