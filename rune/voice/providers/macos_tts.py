"""macOS TTS provider — native say command.

Zero dependencies, zero cost, works offline.
Available on all macOS systems. Korean voice available if installed.
"""

from __future__ import annotations

import asyncio
import platform
import subprocess
import tempfile

from rune.utils.logger import get_logger
from rune.voice.types import SynthesisResult, TTSProvider

log = get_logger(__name__)

_VOICES: dict[str, str] = {
    "ko": "Yuna",       # macOS Korean voice
    "en": "Samantha",
    "ja": "Kyoko",
    "zh": "Ting-Ting",
    "fr": "Thomas",
}


class MacOSTTSProvider(TTSProvider):
    """macOS say command — free, offline, zero dependencies."""

    __slots__ = ("_language", "_voice")

    def __init__(self, language: str = "en", voice: str = "") -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("MacOSTTSProvider is only available on macOS")
        self._language = language
        self._voice = voice or _VOICES.get(language, "Samantha")

    async def synthesize(self, text: str, voice: str = "") -> SynthesisResult:
        if not text.strip():
            return SynthesisResult()

        voice = voice or self._voice
        loop = asyncio.get_running_loop()

        try:
            audio = await loop.run_in_executor(None, self._say_sync, text, voice)
            return SynthesisResult(
                audio=audio,
                sample_rate=22050,
                duration_ms=len(audio) / 44,  # rough estimate
            )
        except Exception as exc:
            log.warning("macos_tts_error", error=str(exc)[:100])
            return SynthesisResult()

    def _say_sync(self, text: str, voice: str) -> bytes:
        """Run macOS say command synchronously."""
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tmp_path = f.name

        try:
            subprocess.run(
                ["say", "-v", voice, "-o", tmp_path, text],
                check=True,
                timeout=30,
                capture_output=True,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            import os
            os.unlink(tmp_path)
