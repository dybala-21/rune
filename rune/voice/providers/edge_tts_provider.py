"""Edge TTS provider — free Microsoft Edge text-to-speech.

No API key required. Supports Korean and many other languages.
Uses the edge-tts package (lightweight, ~50KB).
"""

from __future__ import annotations

from rune.utils.logger import get_logger
from rune.voice.types import SynthesisResult, TTSProvider

log = get_logger(__name__)

# Default voices per language (Microsoft Edge neural voices)
_DEFAULT_VOICES: dict[str, str] = {
    "ko": "ko-KR-SunHiNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
}


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS — free, no API key, Korean supported."""

    __slots__ = ("_language", "_voice")

    def __init__(self, language: str = "en", voice: str = "") -> None:
        self._language = language
        self._voice = voice or _DEFAULT_VOICES.get(language, "en-US-AriaNeural")

    async def synthesize(self, text: str, voice: str = "") -> SynthesisResult:
        if not text.strip():
            return SynthesisResult()

        try:
            import edge_tts
        except ImportError:
            raise RuntimeError(
                "edge-tts is not installed. Install with: uv pip install edge-tts"
            )

        voice = voice or self._voice

        communicate = edge_tts.Communicate(text, voice)
        audio_chunks: list[bytes] = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            return SynthesisResult()

        audio_data = b"".join(audio_chunks)

        log.debug(
            "edge_tts_result",
            text=text[:50],
            voice=voice,
            bytes=len(audio_data),
        )

        return SynthesisResult(
            audio=audio_data,
            sample_rate=24000,
            duration_ms=len(audio_data) / 48,  # rough estimate for mp3
        )
