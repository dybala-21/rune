"""Deepgram STT provider for RUNE.

Implements the STTProvider interface using the Deepgram speech-to-text API.
"""

from __future__ import annotations

import time
from typing import Any

from rune.utils.logger import get_logger
from rune.voice.types import STTProvider, TranscriptionResult

log = get_logger(__name__)

_DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


class DeepgramProvider(STTProvider):
    """Speech-to-text provider using the Deepgram API."""

    __slots__ = ("_api_key", "_model", "_language")

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-2",
        language: str = "en",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe audio data using the Deepgram API.

        Parameters
        ----------
        audio_data:
            Raw audio bytes. Deepgram accepts WAV, MP3, FLAC, OGG, and raw PCM.

        Returns
        -------
        TranscriptionResult with text, confidence, language, and duration.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for DeepgramProvider: pip install httpx"
            ) from exc

        start_ms = time.monotonic() * 1000

        params: dict[str, Any] = {
            "model": self._model,
            "language": self._language,
            "punctuate": "true",
            "utterances": "false",
        }

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _DEEPGRAM_API_URL,
                    params=params,
                    headers=headers,
                    content=audio_data,
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()

            duration_ms = time.monotonic() * 1000 - start_ms

            # Parse Deepgram response
            results = data.get("results", {})
            channels = results.get("channels", [])

            if not channels:
                return TranscriptionResult(duration_ms=duration_ms)

            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return TranscriptionResult(duration_ms=duration_ms)

            best = alternatives[0]
            text = best.get("transcript", "")
            confidence = best.get("confidence", 0.0)

            # Get detected language if available
            detected_language = (
                results.get("channels", [{}])[0]
                .get("detected_language", self._language)
            )

            # Get audio duration from metadata
            audio_duration = data.get("metadata", {}).get("duration", 0.0)
            audio_duration_ms = audio_duration * 1000

            log.debug(
                "deepgram_transcribed",
                text_len=len(text),
                confidence=confidence,
                latency_ms=round(duration_ms),
            )

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=detected_language,
                duration_ms=audio_duration_ms if audio_duration_ms > 0 else duration_ms,
            )

        except Exception as exc:
            duration_ms = time.monotonic() * 1000 - start_ms
            log.error("deepgram_error", error=str(exc), latency_ms=round(duration_ms))
            raise
