"""Attachment type definitions.

Ported from src/attachments/types.ts - MIME types, resolved attachments,
and channel attachment descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# MIME types

ImageMimeType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
AttachmentMimeType = Literal[
    "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"
]

# Resolved attachment


@dataclass(slots=True)
class ResolvedAttachment:
    """A fully resolved and preprocessed attachment."""

    id: str
    """File path or channel file ID."""

    mime_type: AttachmentMimeType
    """Detected MIME type."""

    data: bytes
    """Preprocessed binary data."""

    original_size: int
    """Original file size in bytes."""

    processed_size: int
    """Size after preprocessing in bytes."""

    estimated_tokens: int
    """Estimated token cost: ceil(w*h/750) for images."""

    preprocessing: list[str] = field(default_factory=list)
    """Applied preprocessing steps."""

    filename: str | None = None
    """Original filename (if provided by channel attachment)."""

    width: int = 0
    """Image width in pixels (images only)."""

    height: int = 0
    """Image height in pixels (images only)."""

    image_type: str | None = None
    """Image classification: screenshot/photo/diagram/text-heavy/unknown."""

    content_hash: str | None = None
    """SHA-256 prefix hash for vision cache (images only)."""


# Channel attachment (for Phase 3 multi-channel support)


@dataclass(slots=True)
class ChannelAttachment:
    """Attachment descriptor from a channel (Slack, Discord, etc.)."""

    file_id: str
    filename: str | None = None
    mime_type: str | None = None
    size: int | None = None
    url: str | None = None
