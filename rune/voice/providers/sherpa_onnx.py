"""Sherpa-ONNX local streaming STT provider for RUNE.

CPU-only, offline speech recognition using the ``sherpa-onnx`` Python
package.  Requires model files (encoder, decoder, joiner, tokens) at a
configured path.
"""

from __future__ import annotations

import struct
from typing import Any

from rune.utils.logger import get_logger
from rune.voice.types import STTProvider, TranscriptionResult

log = get_logger(__name__)

# Three-state import cache
_sherpa_module: Any | None = None
_sherpa_checked = False


def _load_sherpa_module() -> Any | None:
    global _sherpa_module, _sherpa_checked
    if _sherpa_checked:
        return _sherpa_module
    _sherpa_checked = True
    try:
        import sherpa_onnx  # type: ignore[import-untyped]
        _sherpa_module = sherpa_onnx
        log.debug("sherpa_onnx_loaded")
        return _sherpa_module
    except ImportError:
        _sherpa_module = None
        log.debug("sherpa_onnx_not_available")
        return None


class SherpaOnnxProvider(STTProvider):
    """Local streaming STT provider using Sherpa-ONNX.

    Parameters:
        model_path: Directory containing the ONNX model files
            (``encoder.onnx``, ``decoder.onnx``, ``joiner.onnx``,
            ``tokens.txt``).
        sample_rate: Audio sample rate (default 16000).
    """

    __slots__ = (
        "_model_path",
        "_sample_rate",
        "_recognizer",
        "_stream",
        "_is_connected",
        "_accumulated_text",
        "_last_partial",
        "on_partial",
        "on_ready",
    )

    def __init__(
        self,
        model_path: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._recognizer: Any = None
        self._stream: Any = None
        self._is_connected = False
        self._accumulated_text = ""
        self._last_partial = ""

        # Callbacks
        self.on_partial: Any = None
        self.on_ready: Any = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def check_availability() -> bool:
        """Return ``True`` if sherpa-onnx is importable."""
        mod = _load_sherpa_module()
        return mod is not None

    # Connection

    def connect(self, language: str = "en") -> None:
        """Initialise the Sherpa-ONNX recognizer.

        Parameters:
            language: Language code (currently informational).

        Raises:
            RuntimeError: If sherpa-onnx is not installed or model init fails.
        """
        if self._is_connected:
            return

        mod = _load_sherpa_module()
        if mod is None:
            raise RuntimeError(
                "sherpa-onnx not installed. Run: pip install sherpa-onnx"
            )

        if not self._model_path:
            raise RuntimeError("model_path must be provided for SherpaOnnxProvider")

        try:
            recognizer_config = {
                "feat_config": {"sample_rate": self._sample_rate, "feature_dim": 80},
                "model_config": {
                    "transducer": {
                        "encoder": f"{self._model_path}/encoder.onnx",
                        "decoder": f"{self._model_path}/decoder.onnx",
                        "joiner": f"{self._model_path}/joiner.onnx",
                    },
                    "tokens": f"{self._model_path}/tokens.txt",
                    "num_threads": 2,
                    "debug": False,
                },
                "decoding_method": "greedy_search",
                "max_active_paths": 4,
            }

            # sherpa_onnx API varies by version; try common patterns
            if hasattr(mod, "OnlineRecognizer"):
                self._recognizer = mod.OnlineRecognizer(**recognizer_config)
            else:
                raise RuntimeError("sherpa_onnx.OnlineRecognizer not found")

            self._stream = self._recognizer.create_stream()
            self._is_connected = True

            if self.on_ready:
                self.on_ready()

            log.debug("sherpa_onnx_connected", language=language)

        except Exception as exc:
            self._is_connected = False
            raise RuntimeError(
                f"Sherpa-ONNX init failed: {exc}. "
                "Ensure model files are at the configured model_path."
            ) from exc

    def disconnect(self) -> None:
        """Disconnect and release resources."""
        self._recognizer = None
        self._stream = None
        self._is_connected = False
        self._accumulated_text = ""
        self._last_partial = ""

    # Audio processing

    def send_audio(self, buffer: bytes) -> None:
        """Feed PCM 16-bit LE audio into the recognizer.

        Parameters:
            buffer: Raw PCM 16-bit signed LE audio bytes.
        """
        if not self._is_connected or self._recognizer is None or self._stream is None:
            return

        try:
            # Convert PCM 16-bit to float32 samples
            n_samples = len(buffer) // 2
            samples = [
                struct.unpack_from("<h", buffer, i * 2)[0] / 32768.0
                for i in range(n_samples)
            ]

            self._stream.accept_waveform(self._sample_rate, samples)

            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode(self._stream)

            result = self._recognizer.get_result(self._stream)
            text = (getattr(result, "text", "") or "").strip()

            if text and text != self._last_partial:
                self._last_partial = text
                full = (
                    f"{self._accumulated_text} {text}"
                    if self._accumulated_text
                    else text
                )
                if self.on_partial:
                    self.on_partial(full)

        except Exception as exc:
            log.debug("sherpa_onnx_process_error", error=str(exc))

    def finalize(self) -> str:
        """Flush remaining audio and return the final transcript.

        Returns:
            The complete transcribed text.
        """
        if self._recognizer is None or self._stream is None:
            return self._accumulated_text

        try:
            # Feed 0.5s of silence to flush
            silence = [0.0] * int(self._sample_rate * 0.5)
            self._stream.accept_waveform(self._sample_rate, silence)

            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode(self._stream)

            result = self._recognizer.get_result(self._stream)
            final_text = (getattr(result, "text", "") or "").strip()

            if final_text:
                if self._accumulated_text:
                    self._accumulated_text += f" {final_text}"
                else:
                    self._accumulated_text = final_text

        except Exception as exc:
            log.debug("sherpa_onnx_finalize_error", error=str(exc))

        # Reset stream for next utterance
        if self._recognizer is not None:
            self._stream = self._recognizer.create_stream()
            self._last_partial = ""

        result_text = self._accumulated_text
        self._accumulated_text = ""
        return result_text

    # STTProvider interface

    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe a complete audio chunk.

        Implements the :class:`STTProvider` interface for batch transcription.
        Connects automatically if not already connected.

        Parameters:
            audio_data: Raw PCM 16-bit LE audio bytes.

        Returns:
            A :class:`TranscriptionResult` with the transcription.
        """
        if not self._is_connected:
            self.connect()

        self.send_audio(audio_data)
        text = self.finalize()

        duration_ms = (len(audio_data) / 2) / self._sample_rate * 1000.0

        return TranscriptionResult(
            text=text,
            confidence=0.8 if text else 0.0,
            language="en",
            duration_ms=round(duration_ms, 2),
        )
