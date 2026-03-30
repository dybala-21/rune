"""Groq Whisper STT provider — ultra-fast cloud transcription.

Uses the Groq API (OpenAI-compatible endpoint). Free tier available.
299x real-time speed with Whisper Large v3.
"""

from __future__ import annotations

import httpx

from rune.utils.logger import get_logger
from rune.voice.types import STTProvider, TranscriptionResult

log = get_logger(__name__)

_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqSTTProvider(STTProvider):
    """Groq Whisper API — free tier, 299x real-time."""

    __slots__ = ("_api_key", "_model", "_language")

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3",
        language: str = "",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        if not audio_data:
            return TranscriptionResult()

        data: dict[str, str] = {"model": self._model}
        if self._language:
            data["language"] = self._language

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data=data,
            )
            response.raise_for_status()

        result = response.json()
        text = result.get("text", "").strip()

        log.debug("groq_stt_result", text=text[:80], model=self._model)
        return TranscriptionResult(text=text, confidence=0.85, language=self._language)
