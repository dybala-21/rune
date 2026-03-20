"""Microphone capture service for RUNE.

Captures audio from the default input device using ``sounddevice`` (preferred)
or ``pyaudio`` (fallback).  Emits PCM 16-bit mono chunks suitable for
feeding into VAD and STT providers.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


class MicrophoneService:
    """Microphone capture service.

    Emits audio data via the ``on_data`` callback as PCM 16-bit LE mono
    bytes, and RMS level (0-1) via ``on_level``.

    Parameters:
        sample_rate: Audio sample rate in Hz (default 16000).
        chunk_duration_ms: Duration of each audio chunk in milliseconds
            (default 100).
    """

    __slots__ = (
        "_sample_rate",
        "_chunk_duration_ms",
        "_is_recording",
        "_stream",
        "_backend",
        "on_data",
        "on_level",
        "on_error",
    )

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration_ms = chunk_duration_ms
        self._is_recording = False
        self._stream: Any = None
        self._backend: str | None = None

        self.on_data: Callable[[bytes], None] | None = None
        self.on_level: Callable[[float], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @staticmethod
    def check_availability() -> bool:
        """Return ``True`` if a microphone backend is importable."""
        for mod in ("sounddevice", "pyaudio"):
            try:
                __import__(mod)
                return True
            except ImportError:
                continue
        return False

    # Start / Stop

    def start(self) -> None:
        """Start recording from the default microphone.

        Raises:
            RuntimeError: If no audio backend is available.
        """
        if self._is_recording:
            return

        frames_per_chunk = int(
            self._sample_rate * self._chunk_duration_ms / 1000
        )

        # Try sounddevice first
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            def _sd_callback(
                indata: Any,
                frames: int,
                time_info: Any,
                status: Any,
            ) -> None:
                if status and self.on_error:
                    self.on_error(RuntimeError(str(status)))

                # indata is numpy float32 array - convert to PCM 16-bit
                try:
                    import numpy as np
                    pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                except Exception:
                    # Manual conversion without numpy
                    pcm16 = b""
                    for i in range(len(indata)):
                        sample = max(-1.0, min(1.0, float(indata[i][0])))
                        pcm16 += struct.pack("<h", int(sample * 32767))

                self._dispatch_audio(pcm16)

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                blocksize=frames_per_chunk,
                callback=_sd_callback,
            )
            self._stream.start()
            self._backend = "sounddevice"
            self._is_recording = True
            log.debug("microphone_started", backend="sounddevice", sample_rate=self._sample_rate)
            return
        except ImportError:
            pass

        # Fallback to pyaudio
        try:
            import pyaudio  # type: ignore[import-untyped]

            pa = pyaudio.PyAudio()

            def _pa_callback(
                in_data: bytes | None,
                frame_count: int,
                time_info: Any,
                status_flags: int,
            ) -> tuple[None, int]:
                if in_data:
                    self._dispatch_audio(in_data)
                return (None, pyaudio.paContinue)

            self._stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=frames_per_chunk,
                stream_callback=_pa_callback,
            )
            self._stream.start_stream()
            self._backend = "pyaudio"
            self._is_recording = True
            log.debug("microphone_started", backend="pyaudio", sample_rate=self._sample_rate)
            return
        except ImportError:
            pass

        raise RuntimeError(
            "No audio backend available. Install sounddevice or pyaudio: "
            "pip install sounddevice"
        )

    def stop(self) -> None:
        """Stop recording."""
        if not self._is_recording or self._stream is None:
            return

        try:
            if self._backend == "sounddevice":
                self._stream.stop()
                self._stream.close()
            elif self._backend == "pyaudio":
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass

        self._stream = None
        self._is_recording = False
        log.debug("microphone_stopped")

    # Internal

    def _dispatch_audio(self, pcm16: bytes) -> None:
        if self.on_data:
            self.on_data(pcm16)
        if self.on_level:
            rms = self._calculate_rms(pcm16)
            self.on_level(rms)

    @staticmethod
    def _calculate_rms(buffer: bytes) -> float:
        """Calculate RMS energy of PCM 16-bit LE buffer, normalised to 0-1."""
        n_samples = len(buffer) // 2
        if n_samples == 0:
            return 0.0
        sum_sq = 0.0
        for i in range(0, len(buffer), 2):
            sample = struct.unpack_from("<h", buffer, i)[0]
            sum_sq += sample * sample
        rms = math.sqrt(sum_sq / n_samples)
        return min(1.0, rms / 32768.0)
