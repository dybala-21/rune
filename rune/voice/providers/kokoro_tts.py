"""Kokoro TTS provider — local text-to-speech via kokoro-onnx.

Apache 2.0 licensed, 82M parameters, CPU-friendly.
Supports: English, Korean, Japanese, French, Mandarin.
Model auto-downloads on first use (~88MB int8).
"""

from __future__ import annotations

import asyncio
from typing import Any

from rune.utils.logger import get_logger
from rune.voice.types import SynthesisResult, TTSProvider

log = get_logger(__name__)

# Default voices per language
_DEFAULT_VOICES: dict[str, str] = {
    "en": "af_heart",
    "ko": "af_heart",  # Kokoro Korean uses same voice IDs
    "ja": "af_heart",
    "fr": "af_heart",
    "zh": "af_heart",
}


class KokoroTTSProvider(TTSProvider):
    """Local TTS via kokoro-onnx. No torch required."""

    __slots__ = ("_model", "_language", "_default_voice")

    def __init__(self, language: str = "en") -> None:
        self._model: Any = None
        self._language = language
        self._default_voice = _DEFAULT_VOICES.get(language, "af_heart")

    def _ensure_model(self) -> Any:
        """Lazy-load model on first use."""
        if self._model is not None:
            return self._model

        try:
            from kokoro_onnx import Kokoro

            self._model = Kokoro(
                model_path="kokoro-v1.0.onnx",
                voices_path="voices-v1.0.bin",
            )
            log.info("kokoro_tts_loaded")
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. "
                "Install with: uv pip install kokoro-onnx"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load Kokoro TTS: {exc}") from exc

        return self._model

    async def synthesize(self, text: str, voice: str = "") -> SynthesisResult:
        """Synthesize text to audio using Kokoro-82M.

        Runs model inference in a thread pool to avoid blocking the event loop.
        """
        if not text.strip():
            return SynthesisResult()

        voice = voice or self._default_voice
        model = self._ensure_model()

        loop = asyncio.get_running_loop()
        samples, sample_rate = await loop.run_in_executor(
            None,
            lambda: model.create(text, voice=voice, speed=1.0, lang=self._language),
        )

        # Convert float32 numpy array to bytes
        import numpy as np

        audio_bytes = (samples * 32767).astype(np.int16).tobytes()
        duration_ms = len(samples) / sample_rate * 1000

        return SynthesisResult(
            audio=audio_bytes,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )
