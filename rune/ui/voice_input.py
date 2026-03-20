"""Voice input widget integration for RUNE TUI.

Provides a thin wrapper around system speech-to-text capabilities.
On macOS uses the ``say``/``SFSpeechRecognizer`` bridge (when available);
on Linux checks for ``whisper`` CLI or ``arecord`` + ``vosk``.

The widget can be embedded in a Textual app to toggle recording and
stream transcribed text back to the input area.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class VoiceInputState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ERROR = "error"


@dataclass(slots=True)
class VoiceInputConfig:
    """Configuration for voice input."""

    record_cmd: list[str] = field(default_factory=list)
    transcribe_cmd: list[str] = field(default_factory=list)
    max_duration_sec: float = 30.0
    sample_rate: int = 16000


def detect_voice_backend() -> VoiceInputConfig | None:
    """Auto-detect an available voice input backend.

    Returns a :class:`VoiceInputConfig` when a usable recording +
    transcription pipeline is found, or *None* otherwise.
    """
    # macOS: use sox for recording, whisper for transcription
    if shutil.which("rec") and shutil.which("whisper"):
        return VoiceInputConfig(
            record_cmd=["rec", "-r", "16000", "-c", "1", "-b", "16"],
            transcribe_cmd=["whisper", "--model", "base", "--output_format", "txt"],
        )

    # Linux: arecord + whisper
    if shutil.which("arecord") and shutil.which("whisper"):
        return VoiceInputConfig(
            record_cmd=["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1"],
            transcribe_cmd=["whisper", "--model", "base", "--output_format", "txt"],
        )

    return None


class VoiceInputController:
    """Manage recording and transcription lifecycle.

    Usage::

        ctrl = VoiceInputController(on_transcript=handle_text)
        await ctrl.start_recording()
        # ... user presses stop ...
        text = await ctrl.stop_and_transcribe()
    """

    def __init__(
        self,
        config: VoiceInputConfig | None = None,
        on_transcript: Callable[[str], None] | None = None,
        on_state_change: Callable[[VoiceInputState], None] | None = None,
    ) -> None:
        self._config = config or detect_voice_backend()
        self._on_transcript = on_transcript
        self._on_state_change = on_state_change
        self._state = VoiceInputState.IDLE
        self._proc: asyncio.subprocess.Process | None = None
        self._tmp_path: Path | None = None

    @property
    def state(self) -> VoiceInputState:
        return self._state

    @property
    def available(self) -> bool:
        return self._config is not None

    def _set_state(self, state: VoiceInputState) -> None:
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)

    async def start_recording(self) -> bool:
        """Begin recording audio. Returns *False* if no backend is available."""
        if not self._config:
            self._set_state(VoiceInputState.ERROR)
            return False

        if self._state == VoiceInputState.RECORDING:
            return True

        self._tmp_path = Path(tempfile.mktemp(suffix=".wav"))
        cmd = [*self._config.record_cmd, str(self._tmp_path)]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._set_state(VoiceInputState.RECORDING)
            return True
        except OSError:
            self._set_state(VoiceInputState.ERROR)
            return False

    async def stop_and_transcribe(self) -> str:
        """Stop recording and return the transcribed text."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (TimeoutError, ProcessLookupError):
                self._proc.kill()
            self._proc = None

        if not self._config or not self._tmp_path or not self._tmp_path.exists():
            self._set_state(VoiceInputState.IDLE)
            return ""

        self._set_state(VoiceInputState.TRANSCRIBING)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [*self._config.transcribe_cmd, str(self._tmp_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            text = result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            text = ""
            self._set_state(VoiceInputState.ERROR)

        # Cleanup temp file
        with contextlib.suppress(OSError):
            self._tmp_path.unlink(missing_ok=True)

        self._set_state(VoiceInputState.IDLE)

        if text and self._on_transcript:
            self._on_transcript(text)

        return text

    async def cancel(self) -> None:
        """Cancel any in-progress recording without transcribing."""
        if self._proc is not None:
            try:
                self._proc.kill()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except (TimeoutError, ProcessLookupError):
                pass
            self._proc = None

        if self._tmp_path:
            with contextlib.suppress(OSError):
                self._tmp_path.unlink(missing_ok=True)

        self._set_state(VoiceInputState.IDLE)
