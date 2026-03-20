"""Voice session for RUNE.

Manages the lifecycle of a voice interaction: start/stop recording,
audio processing, and transcription dispatch.

VoiceSession: Low-level session with external audio injection (feed_audio).
VoiceSessionManager: High-level session that wires MicrophoneService + VAD + STT.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from typing import Any

from rune.utils.logger import get_logger
from rune.voice.types import (
    STTProvider,
    TranscriptionResult,
    VoiceConfig,
    VoiceEventEmitter,
    VoiceState,
)

log = get_logger(__name__)

type TranscriptionCallback = Callable[[TranscriptionResult], Coroutine[Any, Any, None]]
type PartialTranscriptionCallback = Callable[[str], Coroutine[Any, Any, None]]


class VoiceSession(VoiceEventEmitter):
    """Manages a voice recording and transcription session.

    Emits:
        state_change(VoiceState) - when session state transitions
        partial_transcript(str) - streaming intermediate text
        final_transcript(str)   - completed transcription text
        error(Exception)        - on errors
    """

    __slots__ = (
        "_stt",
        "_state",
        "_on_transcription",
        "_on_partial_transcription",
        "_audio_task",
        "_audio_queue",
        "_final_segments",
        "_partial_text",
    )

    def __init__(self, stt_provider: STTProvider) -> None:
        super().__init__()
        self._stt = stt_provider
        self._state: VoiceState = "idle"
        self._on_transcription: TranscriptionCallback | None = None
        self._on_partial_transcription: PartialTranscriptionCallback | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._final_segments: list[str] = []
        self._partial_text: str = ""

    # Properties

    @property
    def state(self) -> VoiceState:
        """Current voice session state."""
        return self._state

    @property
    def on_transcription(self) -> TranscriptionCallback | None:
        return self._on_transcription

    @on_transcription.setter
    def on_transcription(self, callback: TranscriptionCallback | None) -> None:
        self._on_transcription = callback

    @property
    def on_partial_transcription(self) -> PartialTranscriptionCallback | None:
        return self._on_partial_transcription

    @on_partial_transcription.setter
    def on_partial_transcription(self, callback: PartialTranscriptionCallback | None) -> None:
        self._on_partial_transcription = callback

    # Combined text (mirrors TS getCombinedText)

    def get_combined_text(self) -> str:
        """Return all final segments plus any current partial, joined."""
        segments = list(self._final_segments)
        if self._partial_text:
            segments.append(self._partial_text)
        return " ".join(segments).strip()

    # Lifecycle

    async def start(self) -> None:
        """Start the voice session and begin listening."""
        if self._state != "idle":
            log.warning("voice_already_active", state=self._state)
            return

        self._final_segments.clear()
        self._partial_text = ""
        self._set_state("listening")
        self._audio_task = asyncio.create_task(self._audio_loop())
        log.info("voice_session_started")

    async def stop(self) -> str:
        """Stop the voice session and return the final combined text."""
        if self._state not in ("listening", "processing"):
            return ""

        self._set_state("finalizing")

        if self._audio_task is not None:
            self._audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_task
            self._audio_task = None

        # Drain remaining audio
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

        result = self.get_combined_text()
        if result:
            self.emit("final_transcript", result)

        self._final_segments.clear()
        self._partial_text = ""
        self._set_state("idle")

        log.info("voice_session_stopped")
        return result

    # Audio ingestion

    async def feed_audio(self, audio_data: bytes) -> None:
        """Feed audio data into the session for processing.

        Called by the audio capture layer (microphone, websocket, etc.).
        """
        if self._state != "listening":
            return
        await self._audio_queue.put(audio_data)

    # Internal processing

    async def _audio_loop(self) -> None:
        """Background loop that processes queued audio chunks."""
        try:
            while self._state == "listening":
                try:
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.5,
                    )
                except TimeoutError:
                    continue

                await self._process_audio(audio_data)
        except asyncio.CancelledError:
            pass

    async def _process_audio(self, audio_data: bytes) -> None:
        """Transcribe an audio chunk and invoke callbacks."""
        self._set_state("processing")
        try:
            result = await self._stt.transcribe(audio_data)

            if result.is_partial:
                # Partial / intermediate transcript
                self._partial_text = result.text
                combined = self.get_combined_text()
                self.emit("partial_transcript", combined)
                if self._on_partial_transcription is not None:
                    await self._on_partial_transcription(combined)
            elif result.text.strip():
                # Final segment - resolve partial into final
                self._final_segments.append(result.text)
                self._partial_text = ""
                combined = self.get_combined_text()
                # Emit partial with the updated combined text (mirrors TS behaviour)
                self.emit("partial_transcript", combined)

                if self._on_transcription is not None:
                    await self._on_transcription(result)

            log.debug(
                "voice_transcribed",
                text_len=len(result.text),
                confidence=result.confidence,
                is_partial=result.is_partial,
            )

        except Exception as exc:
            log.error("voice_transcription_error", error=str(exc))
            self.emit("error", exc)

        finally:
            if self._state == "processing":
                self._set_state("listening")

    # State management

    def _set_state(self, state: VoiceState) -> None:
        """Transition to *state* and emit a ``state_change`` event."""
        self._state = state
        self.emit("state_change", state)


# VoiceSessionManager - wired mic + VAD + STT pipeline


class VoiceSessionManager(VoiceEventEmitter):
    """High-level voice session that assembles MicrophoneService + VAD + STT.

    Ported from TS ``VoiceSessionManager`` - automatically starts microphone
    capture, routes audio through VAD, and dispatches speech segments to the
    STT provider.

    Emits:
        state_change(VoiceState)   - session state transitions
        partial_transcript(str)    - streaming intermediate text
        final_transcript(str)      - completed transcription
        error(Exception)           - errors
        level(float)               - audio RMS level
        speech_start()             - VAD detected speech
        speech_end()               - VAD detected end of speech

    Usage::

        manager = VoiceSessionManager(stt_provider)
        manager.on("final_transcript", my_handler)
        await manager.start()
        # ... mic captures automatically; transcriptions arrive via events
        await manager.stop()
    """

    __slots__ = (
        "_session",
        "_mic",
        "_vad",
        "_speech_buffer",
        "_in_speech",
        "_sample_rate",
        "_chunk_duration_ms",
        "_config",
    )

    def __init__(
        self,
        stt_provider: STTProvider,
        *,
        config: VoiceConfig | None = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        silence_threshold_ms: int = 1500,
        use_webrtcvad: bool = False,
    ) -> None:
        super().__init__()
        self._session = VoiceSession(stt_provider)
        self._mic: Any = None
        self._vad: Any = None
        self._speech_buffer: bytearray = bytearray()
        self._in_speech: bool = False
        self._sample_rate = sample_rate
        self._chunk_duration_ms = chunk_duration_ms
        self._config = config

        # Forward session events to the manager
        self._session.on("state_change", lambda s: self.emit("state_change", s))
        self._session.on("partial_transcript", lambda t: self.emit("partial_transcript", t))
        self._session.on("final_transcript", lambda t: self.emit("final_transcript", t))
        self._session.on("error", lambda e: self.emit("error", e))

        # Initialize microphone (lazy - may fail if no audio device)
        try:
            from rune.voice.microphone import MicrophoneService
            self._mic = MicrophoneService(
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration_ms,
            )
        except ImportError:
            log.warning("microphone_service_unavailable")

        # Initialize VAD
        vad_silence = (
            config.vad_silence_threshold_ms if config else silence_threshold_ms
        )
        try:
            from rune.voice.vad import VoiceActivityDetector
            self._vad = VoiceActivityDetector(
                silence_threshold_ms=vad_silence,
                use_webrtcvad=use_webrtcvad,
                sample_rate=sample_rate,
            )
        except ImportError:
            log.warning("vad_unavailable")

    @property
    def state(self) -> VoiceState:
        return self._session.state

    @property
    def on_transcription(self) -> TranscriptionCallback | None:
        return self._session.on_transcription

    @on_transcription.setter
    def on_transcription(self, callback: TranscriptionCallback | None) -> None:
        self._session.on_transcription = callback

    @property
    def on_partial_transcription(self) -> PartialTranscriptionCallback | None:
        return self._session.on_partial_transcription

    @on_partial_transcription.setter
    def on_partial_transcription(self, callback: PartialTranscriptionCallback | None) -> None:
        self._session.on_partial_transcription = callback

    def update_config(self, config: VoiceConfig) -> None:
        """Update configuration (e.g. when user changes settings)."""
        self._config = config

    async def toggle(self) -> None:
        """Toggle recording on/off (for keybindings like Ctrl+M)."""
        if self._session.state == "listening":
            await self.stop()
        elif self._session.state == "idle":
            await self.start()
        # Ignore if starting/finalizing (debounce)

    async def start(self) -> None:
        """Start the voice pipeline: mic -> VAD -> STT."""
        # Wire microphone data callback -> VAD processing
        if self._mic is not None:
            self._mic.on_data = self._on_mic_data

        # Wire VAD callbacks
        if self._vad is not None:
            self._vad.on_speech_start = self._on_speech_start
            self._vad.on_speech_end = self._on_speech_end
            self._vad.on_silence = self._on_silence

        # Start the transcription session
        await self._session.start()

        # Start microphone capture
        if self._mic is not None:
            try:
                self._mic.start()
                log.info("voice_manager_mic_started")
            except Exception as exc:
                log.error("voice_manager_mic_start_failed", error=str(exc))
                self.emit("error", exc)

    async def stop(self) -> str:
        """Stop the voice pipeline and return final text."""
        # Stop microphone
        if self._mic is not None:
            with contextlib.suppress(Exception):
                self._mic.stop()

        # Flush any buffered speech
        if self._in_speech and self._speech_buffer:
            await self._session.feed_audio(bytes(self._speech_buffer))
            self._speech_buffer.clear()
            self._in_speech = False

        result = await self._session.stop()
        log.info("voice_manager_stopped")
        return result

    def _on_mic_data(self, pcm_data: bytes) -> None:
        """Handle raw PCM data from the microphone.

        Routes audio through VAD if available, otherwise feeds directly
        to the session.
        """
        if self._vad is not None:
            # Process through VAD - triggers speech_start/speech_end/silence callbacks
            self._vad.process_audio(pcm_data)
            # If in speech, buffer the audio
            if self._in_speech:
                self._speech_buffer.extend(pcm_data)
        else:
            # No VAD - feed raw audio directly to session
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                loop.create_task,
                self._session.feed_audio(pcm_data),
            )

    def _on_speech_start(self) -> None:
        """VAD detected start of speech."""
        self._in_speech = True
        self._speech_buffer.clear()
        self.emit("speech_start")
        log.debug("voice_speech_start")

    def _on_speech_end(self) -> None:
        """VAD detected end of speech; send buffered audio for transcription."""
        if self._speech_buffer:
            audio = bytes(self._speech_buffer)
            self._speech_buffer.clear()
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    loop.create_task,
                    self._session.feed_audio(audio),
                )
            except RuntimeError:
                pass
        self._in_speech = False
        self.emit("speech_end")
        log.debug("voice_speech_end")

    def _on_silence(self) -> None:
        """VAD detected sustained silence; auto-stop if configured."""
        if self._in_speech and self._speech_buffer:
            # Flush any remaining speech
            self._on_speech_end()
        log.debug("voice_silence_detected")


# Singleton factory - mirrors TS ``getVoiceSessionManager``

_instance: VoiceSessionManager | None = None


def get_voice_session_manager(
    stt_provider: STTProvider | None = None,
    *,
    config: VoiceConfig | None = None,
) -> VoiceSessionManager:
    """Return the singleton :class:`VoiceSessionManager`.

    On first call, *stt_provider* is required.  Subsequent calls return the
    existing instance; if *config* is passed it will be updated.

    Raises :class:`RuntimeError` if called without a provider before the
    singleton is initialised.
    """
    global _instance

    if _instance is None:
        if stt_provider is None:
            raise RuntimeError(
                "VoiceSessionManager not initialised — stt_provider is required"
            )
        _instance = VoiceSessionManager(stt_provider, config=config)
    elif config is not None:
        _instance.update_config(config)

    return _instance


def reset_voice_session_manager() -> None:
    """Tear down the singleton (for testing or shutdown)."""
    global _instance
    _instance = None
