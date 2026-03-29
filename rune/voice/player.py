"""Audio player for RUNE voice output.

Plays PCM audio through the default output device using sounddevice.
Supports barge-in (immediate stop when user starts speaking).
"""

from __future__ import annotations

import asyncio
import threading

from rune.utils.logger import get_logger

log = get_logger(__name__)


class AudioPlayer:
    """Plays raw PCM audio through the default speaker.

    Thread-safe: playback runs in a background thread, ``stop()``
    can be called from any thread for barge-in.
    """

    __slots__ = ("_playing", "_stop_event")

    def __init__(self) -> None:
        self._playing = False
        self._stop_event = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play(self, audio: bytes, sample_rate: int = 24000) -> None:
        """Play audio bytes (PCM 16-bit mono). Non-blocking."""
        if not audio:
            return

        self._stop_event.clear()
        self._playing = True

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._play_sync, audio, sample_rate)
        finally:
            self._playing = False

    def _play_sync(self, audio: bytes, sample_rate: int) -> None:
        """Synchronous playback in thread pool."""
        try:
            import numpy as np
            import sounddevice as sd

            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32767
            block_size = sample_rate // 10  # 100ms blocks

            for i in range(0, len(samples), block_size):
                if self._stop_event.is_set():
                    log.debug("audio_playback_interrupted")
                    break
                block = samples[i : i + block_size]
                sd.play(block, samplerate=sample_rate, blocking=True)

        except ImportError:
            log.warning("sounddevice_not_available")
        except Exception as exc:
            log.warning("audio_playback_error", error=str(exc)[:100])

    def stop(self) -> None:
        """Stop playback immediately (barge-in)."""
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
