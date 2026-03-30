"""Unified voice service for RUNE.

Single entry point for all voice operations. Used by CLI, TUI, Web API,
and channel adapters (Telegram/Slack/Discord).

Usage:
    svc = get_voice_service()
    text = await svc.transcribe(audio_bytes)    # any interface
    text = await svc.listen_and_transcribe()     # CLI/TUI (mic input)
    audio = await svc.speak(response_text)       # any interface
"""

from __future__ import annotations

import asyncio
from typing import Any

from rune.utils.logger import get_logger
from rune.voice.player import AudioPlayer
from rune.voice.types import STTProvider, SynthesisResult, TTSProvider, TranscriptionResult

log = get_logger(__name__)


class VoiceService:
    """Unified voice service — transcribe, speak, listen.

    All interfaces (CLI, TUI, Web, Telegram) use the same service instance.
    STT and TTS providers are injected at construction time.
    """

    __slots__ = ("_stt", "_tts", "_player", "_language")

    def __init__(
        self,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        language: str = "en",
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._player = AudioPlayer()
        self._language = language

    @property
    def has_stt(self) -> bool:
        return self._stt is not None

    @property
    def has_tts(self) -> bool:
        return self._tts is not None

    # ── STT (any interface) ──────────────────────────────────────────────

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes to text.

        Used by Web API (audio upload), Telegram (voice message),
        and internally by listen_and_transcribe().
        """
        if self._stt is None:
            raise RuntimeError("No STT provider configured")

        result = await self._stt.transcribe(audio_data)
        log.debug(
            "voice_transcribed",
            text=result.text[:80],
            confidence=round(result.confidence, 2),
        )
        return result.text

    # ── TTS (any interface) ──────────────────────────────────────────────

    async def speak(self, text: str) -> SynthesisResult:
        """Synthesize text to audio.

        Used by CLI (play through speaker), Web API (return audio),
        Telegram (send voice message).
        """
        if self._tts is None:
            return SynthesisResult()  # silent — no TTS configured

        result = await self._tts.synthesize(text)
        log.debug(
            "voice_synthesized",
            text=text[:50],
            duration_ms=round(result.duration_ms),
        )
        return result

    async def speak_and_play(self, text: str) -> None:
        """Synthesize and play through speaker (CLI/TUI)."""
        result = await self.speak(text)
        if result.audio:
            await self._player.play(result.audio, result.sample_rate)

    def stop_playback(self) -> None:
        """Stop audio playback immediately (barge-in)."""
        self._player.stop()

    # ── Microphone input (CLI/TUI only) ──────────────────────────────────

    async def listen_and_transcribe(self) -> str:
        """Listen from microphone and transcribe. Blocks until speech ends.

        Uses VoiceSessionManager for mic + VAD + STT pipeline.
        """
        from rune.voice.session import get_voice_session_manager

        mgr = get_voice_session_manager(self._stt)

        # Wait for a single utterance
        transcript_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def _on_final(text: str) -> None:
            if not transcript_future.done():
                transcript_future.set_result(text)

        mgr.on("final_transcript", _on_final)

        try:
            await mgr.start()
            text = await asyncio.wait_for(transcript_future, timeout=30.0)
            return text.strip()
        except asyncio.TimeoutError:
            log.debug("voice_listen_timeout")
            return ""
        finally:
            mgr.off("final_transcript", _on_final)
            await mgr.stop()


# ── Factory ─────────────────────────────────────────────────────────────

_service: VoiceService | None = None


def get_voice_service(language: str = "en") -> VoiceService:
    """Get or create the singleton VoiceService with auto-detected providers."""
    global _service
    if _service is not None:
        return _service

    stt = _auto_detect_stt()
    tts = _auto_detect_tts(language)

    _service = VoiceService(stt=stt, tts=tts, language=language)
    log.info(
        "voice_service_created",
        has_stt=_service.has_stt,
        has_tts=_service.has_tts,
    )
    return _service


def reset_voice_service() -> None:
    """Reset singleton (for testing)."""
    global _service
    _service = None


def _auto_detect_stt() -> STTProvider | None:
    """Auto-detect best available STT provider.

    Priority: Deepgram > OpenAI Whisper > Groq Whisper > Sherpa-ONNX (local).
    """
    import os

    # 1. Deepgram (best quality, Korean, $0.004/min)
    if os.environ.get("DEEPGRAM_API_KEY"):
        try:
            from rune.voice.providers.deepgram import DeepgramSTTProvider
            return DeepgramSTTProvider(api_key=os.environ["DEEPGRAM_API_KEY"])
        except Exception as exc:
            log.debug("stt_detect_skip", provider="deepgram", error=str(exc)[:80])

    # 2. OpenAI Whisper (most users have this key, 99 languages, $0.006/min)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from rune.voice.providers.openai_stt import OpenAISTTProvider
            return OpenAISTTProvider(api_key=os.environ["OPENAI_API_KEY"])
        except Exception as exc:
            log.debug("stt_detect_skip", provider="openai", error=str(exc)[:80])

    # 3. Groq Whisper (free tier, ultra-fast)
    if os.environ.get("GROQ_API_KEY"):
        try:
            from rune.voice.providers.groq_stt import GroqSTTProvider
            return GroqSTTProvider(api_key=os.environ["GROQ_API_KEY"])
        except Exception as exc:
            log.debug("stt_detect_skip", provider="groq", error=str(exc)[:80])

    # 4. Sherpa-ONNX (fully local, no API key)
    try:
        from rune.voice.providers.sherpa_onnx import SherpaOnnxSTTProvider
        return SherpaOnnxSTTProvider()
    except Exception as exc:
        log.debug("stt_detect_skip", provider="sherpa_onnx", error=str(exc)[:80])

    return None


def _auto_detect_tts(language: str = "en") -> TTSProvider | None:
    """Auto-detect best available TTS provider.

    Priority: Kokoro (local) > Edge TTS (free cloud) > macOS say (offline).
    """
    # 1. Kokoro-ONNX (local, best quality, Korean)
    try:
        from rune.voice.providers.kokoro_tts import KokoroTTSProvider
        return KokoroTTSProvider(language=language)
    except Exception as exc:
        log.debug("tts_detect_skip", provider="kokoro", error=str(exc)[:80])

    # 2. Edge TTS (free, no API key, Korean)
    try:
        from rune.voice.providers.edge_tts_provider import EdgeTTSProvider
        return EdgeTTSProvider(language=language)
    except Exception as exc:
        log.debug("tts_detect_skip", provider="edge_tts", error=str(exc)[:80])

    # 3. macOS say (offline, zero deps)
    import platform
    if platform.system() == "Darwin":
        try:
            from rune.voice.providers.macos_tts import MacOSTTSProvider
            return MacOSTTSProvider(language=language)
        except Exception as exc:
            log.debug("tts_detect_skip", provider="macos_say", error=str(exc)[:80])

    return None
