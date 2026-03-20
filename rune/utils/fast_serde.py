"""High-performance serialization for RUNE hot paths.

Phase 9 optimization - uses ``msgspec`` for daemon IPC and streaming
chunk serialization.  Falls back to stdlib ``json`` when msgspec is not
installed so the rest of the codebase never hard-depends on it.

API boundaries (user-facing models) still use Pydantic; this module is
strictly for internal hot paths where raw throughput matters.
"""

from __future__ import annotations

import json as _json
from typing import Any

# Try to import msgspec; set a flag so callers can check.

try:
    import msgspec
    from msgspec import Struct as _Struct

    HAS_MSGSPEC = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_MSGSPEC = False
    _Struct = object  # type: ignore[assignment,misc]

# Hot-path structs (msgspec.Struct when available, plain dataclass otherwise)

if HAS_MSGSPEC:

    class StreamChunk(_Struct):  # type: ignore[misc]
        """A single streaming token/chunk emitted during generation."""
        chunk_id: str
        content: str
        role: str = "assistant"
        finish_reason: str | None = None
        model: str = ""
        latency_ms: float = 0.0

    class ToolCallEvent(_Struct):  # type: ignore[misc]
        """Lightweight representation of a tool invocation for IPC."""
        call_id: str
        tool_name: str
        arguments: str = ""
        result: str | None = None
        success: bool | None = None
        duration_ms: float = 0.0

    class TokenUsage(_Struct):  # type: ignore[misc]
        """Token accounting record attached to every LLM round-trip."""
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        model: str = ""
        cached: bool = False

else:
    # Fallback: plain classes with the same fields so serialization helpers
    # still work (just slower).
    from dataclasses import dataclass

    @dataclass(slots=True)
    class StreamChunk:  # type: ignore[no-redef]
        chunk_id: str
        content: str
        role: str = "assistant"
        finish_reason: str | None = None
        model: str = ""
        latency_ms: float = 0.0

    @dataclass(slots=True)
    class ToolCallEvent:  # type: ignore[no-redef]
        call_id: str
        tool_name: str
        arguments: str = ""
        result: str | None = None
        success: bool | None = None
        duration_ms: float = 0.0

    @dataclass(slots=True)
    class TokenUsage:  # type: ignore[no-redef]
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        model: str = ""
        cached: bool = False


# Encode / Decode helpers

# Type alias covering all hot-path structs.
_HotType = StreamChunk | ToolCallEvent | TokenUsage

# Pre-built msgspec encoders/decoders (reuse for speed).
if HAS_MSGSPEC:
    _encoder = msgspec.json.Encoder()
    _chunk_decoder = msgspec.json.Decoder(StreamChunk)
    _tool_decoder = msgspec.json.Decoder(ToolCallEvent)
    _usage_decoder = msgspec.json.Decoder(TokenUsage)

    _DECODERS: dict[type, msgspec.json.Decoder[Any]] = {
        StreamChunk: _chunk_decoder,
        ToolCallEvent: _tool_decoder,
        TokenUsage: _usage_decoder,
    }


def encode_chunk(obj: _HotType) -> bytes:
    """Serialize a hot-path struct to JSON bytes.

    Uses msgspec when available; otherwise stdlib json.
    """
    if HAS_MSGSPEC:
        return _encoder.encode(obj)
    # Fallback: dataclass -> dict -> json bytes
    from dataclasses import asdict
    return _json.dumps(asdict(obj)).encode()


def decode_chunk(data: bytes | str, typ: type[_HotType]) -> _HotType:
    """Deserialize JSON bytes into *typ*.

    Parameters
    ----------
    data:
        Raw JSON (bytes or str).
    typ:
        One of ``StreamChunk``, ``ToolCallEvent``, ``TokenUsage``.
    """
    if isinstance(data, str):
        data = data.encode()

    if HAS_MSGSPEC:
        decoder = _DECODERS.get(typ)
        if decoder is None:
            raise ValueError(f"No msgspec decoder registered for {typ!r}")
        return decoder.decode(data)

    # Fallback
    raw = _json.loads(data)
    return typ(**raw)


# Generic JSON helpers (drop-in replacements for stdlib json hot paths)

if HAS_MSGSPEC:
    _generic_encoder = msgspec.json.Encoder()
    _generic_decoder = msgspec.json.Decoder()

    def json_encode(obj: Any) -> str:
        """Serialize *obj* to a JSON string (msgspec fast path)."""
        return _generic_encoder.encode(obj).decode("utf-8")

    def json_encode_bytes(obj: Any) -> bytes:
        """Serialize *obj* to JSON bytes (msgspec fast path)."""
        return _generic_encoder.encode(obj)

    def json_decode(data: str | bytes) -> Any:
        """Deserialize JSON string/bytes to Python objects (msgspec fast path)."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _generic_decoder.decode(data)

else:

    def json_encode(obj: Any) -> str:  # type: ignore[misc]
        """Serialize *obj* to a JSON string (stdlib fallback)."""
        return _json.dumps(obj, ensure_ascii=False)

    def json_encode_bytes(obj: Any) -> bytes:  # type: ignore[misc]
        """Serialize *obj* to JSON bytes (stdlib fallback)."""
        return _json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def json_decode(data: str | bytes) -> Any:  # type: ignore[misc]
        """Deserialize JSON string/bytes to Python objects (stdlib fallback)."""
        return _json.loads(data)


__all__ = [
    "HAS_MSGSPEC",
    "StreamChunk",
    "ToolCallEvent",
    "TokenUsage",
    "encode_chunk",
    "decode_chunk",
    "json_encode",
    "json_encode_bytes",
    "json_decode",
]
