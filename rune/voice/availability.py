"""Voice capability availability checking for RUNE.

Caches three-state availability for microphone capture and STT providers
(Deepgram via API key, Sherpa-ONNX via local package).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Literal

_ToolState = Literal["unknown", "available", "unavailable"]

_mic_state: _ToolState = "unknown"
_deepgram_state: _ToolState = "unknown"
_sherpa_onnx_state: _ToolState = "unknown"


def _check() -> None:
    """Probe each dependency once and cache the result."""
    global _mic_state, _deepgram_state, _sherpa_onnx_state

    # Microphone: sounddevice or pyaudio
    if _mic_state == "unknown":
        for mod_name in ("sounddevice", "pyaudio"):
            try:
                importlib.import_module(mod_name)
                _mic_state = "available"
                break
            except ImportError:
                continue
        else:
            _mic_state = "unavailable"

    # Deepgram: needs API key
    if _deepgram_state == "unknown":
        _deepgram_state = (
            "available" if os.environ.get("DEEPGRAM_API_KEY") else "unavailable"
        )

    # Sherpa-ONNX: local offline STT
    if _sherpa_onnx_state == "unknown":
        try:
            importlib.import_module("sherpa_onnx")
            _sherpa_onnx_state = "available"
        except ImportError:
            _sherpa_onnx_state = "unavailable"


@dataclass(slots=True)
class VoiceAvailability:
    """Detailed voice-input availability information."""

    available: bool = False
    mic_available: bool = False
    provider_available: bool = False
    active_provider: str | None = None
    install_hint: str | None = None


def is_voice_input_available() -> bool:
    """Return ``True`` if voice input is possible on this system.

    Requires a working microphone library *and* at least one STT provider.
    """
    _check()
    if _mic_state != "available":
        return False
    return _deepgram_state == "available" or _sherpa_onnx_state == "available"


def get_voice_availability() -> VoiceAvailability:
    """Return detailed voice availability information."""
    _check()

    mic_ok = _mic_state == "available"
    dg_ok = _deepgram_state == "available"
    sp_ok = _sherpa_onnx_state == "available"
    provider_ok = dg_ok or sp_ok

    active: str | None = None
    if dg_ok:
        active = "deepgram"
    elif sp_ok:
        active = "sherpa-onnx"

    hint: str | None = None
    if not mic_ok:
        hint = "Install sounddevice or pyaudio for microphone capture: pip install sounddevice"
    elif not provider_ok:
        hint = "Set DEEPGRAM_API_KEY or install sherpa-onnx for STT: pip install sherpa-onnx"

    return VoiceAvailability(
        available=mic_ok and provider_ok,
        mic_available=mic_ok,
        provider_available=provider_ok,
        active_provider=active,
        install_hint=hint,
    )


def get_voice_install_hint() -> str | None:
    """Return a user-facing install hint, or ``None`` if everything is fine."""
    return get_voice_availability().install_hint


def _reset_cache() -> None:
    """Reset cached states (for testing)."""
    global _mic_state, _deepgram_state, _sherpa_onnx_state
    _mic_state = "unknown"
    _deepgram_state = "unknown"
    _sherpa_onnx_state = "unknown"
