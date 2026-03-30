"""OpenAI Whisper STT provider — cloud transcription via httpx.

Uses the OpenAI audio/transcriptions API. No additional SDK needed;
httpx (already a RUNE dependency) handles the multipart upload.
"""

from __future__ import annotations

import httpx

from rune.utils.logger import get_logger
from rune.voice.types import STTProvider, TranscriptionResult

log = get_logger(__name__)

_API_URL = "https://api.openai.com/v1/audio/transcriptions"


class OpenAISTTProvider(STTProvider):
    """OpenAI Whisper API — 99 languages, $0.006/min."""

    __slots__ = ("_api_key", "_model", "_language")

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
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

        log.debug("openai_stt_result", text=text[:80], model=self._model)
        return TranscriptionResult(text=text, confidence=0.9, language=self._language)
