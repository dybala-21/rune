"""Voice module for RUNE — STT, TTS, and unified VoiceService.

Re-exports for public API.
"""

from rune.voice.session import (
    VoiceSession,
    VoiceSessionManager,
    get_voice_session_manager,
    reset_voice_session_manager,
)
from rune.voice.types import (
    STTProvider,
    SynthesisResult,
    TranscriptionResult,
    TTSProvider,
    VoiceConfig,
    VoiceEvent,
    VoiceEventEmitter,
    VoiceState,
)

__all__ = [
    "STTProvider",
    "SynthesisResult",
    "TTSProvider",
    "TranscriptionResult",
    "VoiceConfig",
    "VoiceEvent",
    "VoiceEventEmitter",
    "VoiceSession",
    "VoiceSessionManager",
    "VoiceState",
    "get_voice_session_manager",
    "reset_voice_session_manager",
]
