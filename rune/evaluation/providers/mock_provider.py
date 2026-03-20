"""Mock LLM provider for RUNE evaluation system.

Supports recording / playback / passthrough modes so that evaluation
runs can be made deterministic without hitting a real LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from rune.utils.fast_serde import json_decode

MockProviderMode = Literal["playback", "recording", "passthrough"]


@dataclass(slots=True)
class RecordedResponse:
    """A cached LLM response keyed by request hash."""

    request_hash: str = ""
    response: str = ""
    timestamp: str = ""
    model: str = "recorded"
    token_usage: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})


@dataclass(slots=True)
class MockProviderOptions:
    """Configuration for :class:`MockProvider`."""

    recording_path: str = ""
    mode: MockProviderMode = "playback"
    fallback_to_real: bool = False


# Type alias for a real LLM call function
RealProvider = Callable[[list[dict[str, str]], str], Awaitable[str]]


class MockProvider:
    """Mock LLM provider with recording / playback support.

    Modes:
    - ``playback`` -- return pre-recorded responses (error if missing).
    - ``recording`` -- call the real LLM and persist each response.
    - ``passthrough`` -- always call the real LLM (live tests).
    """

    __slots__ = (
        "_recordings",
        "_mode",
        "_recording_path",
        "_fallback_to_real",
        "_dirty",
    )

    def __init__(self, options: MockProviderOptions) -> None:
        self._recordings: dict[str, RecordedResponse] = {}
        self._mode: MockProviderMode = options.mode
        self._recording_path: str = options.recording_path
        self._fallback_to_real: bool = options.fallback_to_real
        self._dirty: bool = False
        self._load_recordings()

    # Hashing

    @staticmethod
    def _hash_request(
        messages: list[dict[str, str]], system_prompt: str,
    ) -> str:
        normalized = [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in messages
        ]
        content = json.dumps(
            {"messages": normalized, "systemPrompt": system_prompt},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # Public API

    def create_caller(
        self,
        real_provider: RealProvider | None = None,
    ) -> Callable[[list[dict[str, str]], str], Awaitable[str]]:
        """Return an async callable that routes through mock logic.

        Parameters:
            real_provider: Optional real LLM function used in recording
                or passthrough mode.
        """

        async def _call(
            messages: list[dict[str, str]], system_prompt: str,
        ) -> str:
            return await self.call(messages, system_prompt, real_provider)

        return _call

    async def call(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        real_provider: RealProvider | None = None,
    ) -> str:
        """Route a request through recording/playback/passthrough logic.

        Parameters:
            messages: Conversation messages.
            system_prompt: System prompt string.
            real_provider: Optional real LLM callable.

        Returns:
            The LLM response string.

        Raises:
            RuntimeError: If required provider is missing or no recording found.
        """
        req_hash = self._hash_request(messages, system_prompt)

        # Passthrough
        if self._mode == "passthrough":
            if real_provider is None:
                raise RuntimeError("Real provider required for passthrough mode")
            return await real_provider(messages, system_prompt)

        # Playback
        if self._mode == "playback":
            recorded = self._recordings.get(req_hash)
            if recorded is not None:
                return recorded.response

            if self._fallback_to_real and real_provider is not None:
                response = await real_provider(messages, system_prompt)
                self._save_recording(req_hash, response)
                return response

            raise RuntimeError(
                f"No recording found for request hash: {req_hash}\n"
                "Run with RUNE_EVAL_MODE=record to create recordings."
            )

        # Recording
        if real_provider is None:
            raise RuntimeError("Real provider required for recording mode")
        response = await real_provider(messages, system_prompt)
        self._save_recording(req_hash, response)
        return response

    # Persistence

    def _save_recording(self, req_hash: str, response: str) -> None:
        self._recordings[req_hash] = RecordedResponse(
            request_hash=req_hash,
            response=response,
            timestamp=datetime.now().isoformat(),
        )
        self._dirty = True

    def _load_recordings(self) -> None:
        if not self._recording_path or not os.path.exists(self._recording_path):
            return
        try:
            raw = Path(self._recording_path).read_text(encoding="utf-8")
            data = json_decode(raw)
            items = data if isinstance(data, list) else data.get("recordings", [])
            for item in items:
                rh = item.get("request_hash") or item.get("requestHash", "")
                if rh:
                    self._recordings[rh] = RecordedResponse(
                        request_hash=rh,
                        response=item.get("response", ""),
                        timestamp=item.get("timestamp", ""),
                        model=item.get("model", "recorded"),
                        token_usage=item.get("token_usage") or item.get("tokenUsage", {"input": 0, "output": 0}),
                    )
        except (json.JSONDecodeError, OSError):
            pass

    def flush(self) -> None:
        """Write dirty recordings to disk."""
        if not self._dirty or not self._recording_path:
            return
        path = Path(self._recording_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "request_hash": r.request_hash,
                "response": r.response,
                "timestamp": r.timestamp,
                "model": r.model,
                "token_usage": r.token_usage,
            }
            for r in self._recordings.values()
        ]
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        self._dirty = False

    @property
    def recording_count(self) -> int:
        """Number of cached recordings."""
        return len(self._recordings)
