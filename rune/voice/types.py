"""Voice type definitions for RUNE.

Shared types for speech-to-text providers and voice sessions.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal

from rune.utils.logger import get_logger

_log = get_logger(__name__)

# ── States ────────────────────────────────────────────────────────────────

VoiceState = Literal["idle", "starting", "listening", "processing", "finalizing", "speaking", "error"]

# ── Event names (mirrors TS VoiceSessionEvents) ──────────────────────────

VoiceEvent = Literal[
    "state_change",
    "partial_transcript",
    "final_transcript",
    "error",
    "level",
    "speech_start",
    "speech_end",
]


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class TranscriptionResult:
    """Result from a speech-to-text transcription."""

    text: str = ""
    confidence: float = 0.0
    language: str = "en"
    duration_ms: float = 0.0
    is_partial: bool = False


@dataclass(slots=True)
class VoiceConfig:
    """Configuration for the voice pipeline.

    Mirrors the TS ``VoiceConfig`` interface.
    """

    enabled: bool = False
    provider: Literal["deepgram", "sherpa_onnx", "auto"] = "auto"
    language: str = "en"
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-2"
    sherpa_onnx_model_path: str = ""
    vad_enabled: bool = True
    vad_silence_threshold_ms: int = 1500


# ── EventEmitter mixin ──────────────────────────────────────────────────

type _AsyncHandler = Callable[..., Coroutine[Any, Any, None]]
type _SyncHandler = Callable[..., None]
type _Handler = _AsyncHandler | _SyncHandler


class VoiceEventEmitter:
    """Lightweight event emitter for voice components.

    Supports both sync and async listeners.  Mirrors the subset of Node.js
    ``EventEmitter`` used by the TS voice module.
    """

    __slots__ = ("_listeners",)

    def __init__(self) -> None:
        self._listeners: defaultdict[str, list[_Handler]] = defaultdict(list)

    # ── Public API ────────────────────────────────────────────────────

    def on(self, event: str, handler: _Handler) -> None:
        """Register a listener for *event*."""
        self._listeners[event].append(handler)

    def off(self, event: str, handler: _Handler) -> None:
        """Remove a specific listener for *event*."""
        with contextlib.suppress(ValueError):
            self._listeners[event].remove(handler)

    def once(self, event: str, handler: _Handler) -> None:
        """Register a listener that fires only once."""

        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            self.off(event, _wrapper)
            return handler(*args, **kwargs)

        self.on(event, _wrapper)

    def remove_all_listeners(self, event: str | None = None) -> None:
        """Remove all listeners, optionally only for a given *event*."""
        if event is None:
            self._listeners.clear()
        else:
            self._listeners.pop(event, None)

    def emit(self, event: str, *args: Any) -> None:
        """Emit *event* synchronously.

        Async handlers are scheduled on the running event loop (fire-and-forget).
        """
        import asyncio
        import inspect

        for handler in list(self._listeners.get(event, [])):
            try:
                result = handler(*args)
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)  # type: ignore[arg-type]
                    except RuntimeError:
                        pass  # no running loop - skip async handler
            except Exception:
                _log.warning("voice_event_handler_error", event_name=event, exc_info=True)

    def listener_count(self, event: str) -> int:
        """Return the number of listeners registered for *event*."""
        return len(self._listeners.get(event, []))


# ── STT Provider ─────────────────────────────────────────────────────────

class STTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe raw audio data to text.

        Parameters
        ----------
        audio_data:
            Raw audio bytes (WAV/PCM expected by most providers).

        Returns
        -------
        TranscriptionResult with the transcribed text and metadata.
        """
        ...
