"""Voice Activity Detection for RUNE.

Provides energy-based (RMS) voice activity detection with optional
webrtcvad or silero-vad backends.  Emits ``speech_start``, ``speech_end``,
and ``silence`` callbacks when voice activity state transitions occur.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable

# Try to import webrtcvad as an optional dependency
_webrtcvad: object | None = None
try:
    import webrtcvad as _webrtcvad  # type: ignore[no-redef]
except ImportError:
    _webrtcvad = None


class VoiceActivityDetector:
    """Detects speech / silence transitions in PCM 16-bit audio.

    By default uses RMS energy detection.  If ``webrtcvad`` is installed,
    it can be enabled via *use_webrtcvad*.

    Parameters:
        silence_threshold_ms: Duration of silence before ``speech_end``
            fires (milliseconds).
        speech_energy_threshold: RMS energy above which audio is considered
            speech (0-1 normalised).
        silence_energy_threshold: RMS energy below which audio is considered
            silence.
        use_webrtcvad: Use ``webrtcvad`` instead of energy detection when
            available.
        sample_rate: Audio sample rate (must be 8000, 16000, 32000, or
            48000 for webrtcvad).
    """

    __slots__ = (
        "_silence_threshold_ms",
        "_speech_threshold",
        "_silence_threshold",
        "_is_speaking",
        "_silence_elapsed_ms",
        "_destroyed",
        "_vad",
        "_sample_rate",
        "_on_speech_start",
        "_on_speech_end",
        "_on_silence",
    )

    def __init__(
        self,
        silence_threshold_ms: float = 1500.0,
        speech_energy_threshold: float = 0.015,
        silence_energy_threshold: float = 0.008,
        use_webrtcvad: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self._silence_threshold_ms = silence_threshold_ms
        self._speech_threshold = speech_energy_threshold
        self._silence_threshold = silence_energy_threshold
        self._is_speaking = False
        self._silence_elapsed_ms = 0.0
        self._destroyed = False
        self._sample_rate = sample_rate

        self._on_speech_start: Callable[[], None] | None = None
        self._on_speech_end: Callable[[], None] | None = None
        self._on_silence: Callable[[], None] | None = None

        # Optional webrtcvad backend
        self._vad: object | None = None
        if use_webrtcvad and _webrtcvad is not None:
            vad = _webrtcvad.Vad()  # type: ignore[union-attr]
            vad.set_mode(2)  # moderate aggressiveness
            self._vad = vad

    # Callbacks

    @property
    def on_speech_start(self) -> Callable[[], None] | None:
        return self._on_speech_start

    @on_speech_start.setter
    def on_speech_start(self, cb: Callable[[], None] | None) -> None:
        self._on_speech_start = cb

    @property
    def on_speech_end(self) -> Callable[[], None] | None:
        return self._on_speech_end

    @on_speech_end.setter
    def on_speech_end(self, cb: Callable[[], None] | None) -> None:
        self._on_speech_end = cb

    @property
    def on_silence(self) -> Callable[[], None] | None:
        return self._on_silence

    @on_silence.setter
    def on_silence(self, cb: Callable[[], None] | None) -> None:
        self._on_silence = cb

    # Availability

    @staticmethod
    def webrtcvad_available() -> bool:
        """Return ``True`` if webrtcvad is installed."""
        return _webrtcvad is not None

    # Processing

    def process_audio(self, buffer: bytes) -> None:
        """Process a chunk of PCM 16-bit mono audio.

        Parameters:
            buffer: Raw PCM 16-bit LE audio bytes.
        """
        if self._destroyed:
            return

        has_speech = self._detect_speech(buffer)
        chunk_duration_ms = (len(buffer) / 2) / self._sample_rate * 1000.0

        if has_speech and not self._is_speaking:
            self._is_speaking = True
            self._silence_elapsed_ms = 0.0
            if self._on_speech_start:
                self._on_speech_start()

        elif not has_speech and self._is_speaking:
            self._silence_elapsed_ms += chunk_duration_ms
            if self._silence_elapsed_ms >= self._silence_threshold_ms:
                self._is_speaking = False
                self._silence_elapsed_ms = 0.0
                if self._on_speech_end:
                    self._on_speech_end()
                if self._on_silence:
                    self._on_silence()

        elif has_speech and self._is_speaking:
            self._silence_elapsed_ms = 0.0

    def _detect_speech(self, buffer: bytes) -> bool:
        """Detect speech using webrtcvad or energy fallback."""
        if self._vad is not None:
            try:
                return self._vad.is_speech(buffer, self._sample_rate)  # type: ignore[union-attr]
            except Exception:
                pass  # fall through to energy-based
        return self._calculate_energy(buffer) > self._speech_threshold

    # Energy calculation

    @staticmethod
    def _calculate_energy(buffer: bytes) -> float:
        """Calculate RMS energy of PCM 16-bit LE buffer, normalised to 0-1."""
        n_samples = len(buffer) // 2
        if n_samples == 0:
            return 0.0

        sum_squares = 0.0
        for i in range(0, len(buffer), 2):
            sample = struct.unpack_from("<h", buffer, i)[0]
            sum_squares += sample * sample

        return math.sqrt(sum_squares / n_samples) / 32768.0

    # Lifecycle

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        """Reset detector state."""
        self._is_speaking = False
        self._silence_elapsed_ms = 0.0

    def destroy(self) -> None:
        """Destroy and clean up."""
        self._destroyed = True
        self._on_speech_start = None
        self._on_speech_end = None
        self._on_silence = None
