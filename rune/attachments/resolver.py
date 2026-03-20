"""Attachment resolver.

Ported from src/attachments/resolver.ts - resolves @file references
and channel attachments into ResolvedAttachment objects.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Awaitable, Callable
from pathlib import Path

from rune.attachments.classifier import classify_image_type
from rune.attachments.preprocessor import preprocess_image
from rune.attachments.types import (
    AttachmentMimeType,
    ChannelAttachment,
    ResolvedAttachment,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants

MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB
MAX_PDF_SIZE = 50 * 1024 * 1024     # 50 MB

_MIME_MAP: dict[str, AttachmentMimeType] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}

_SUPPORTED_MIMES: set[str] = set(_MIME_MAP.values())


def _content_hash(data: bytes) -> str:
    """SHA-256 prefix (16 hex chars) of binary data."""
    return hashlib.sha256(data).hexdigest()[:16]


def _estimate_image_tokens(w: int, h: int) -> int:
    """Estimate token cost: ceil(w*h / 750)."""
    if w == 0 or h == 0:
        return 0
    return math.ceil((w * h) / 750)


# Resolver

class AttachmentResolver:
    """Resolve file paths and channel attachments to ``ResolvedAttachment``."""

    async def resolve_file(self, full_path: str) -> ResolvedAttachment | None:
        """Resolve a local file path to a ``ResolvedAttachment``.

        Supports images (png, jpg, jpeg, gif, webp) and PDFs.
        Images are preprocessed (resized, EXIF stripped).
        Returns None for unsupported file types.

        Raises ``ValueError`` if the file exceeds size limits.
        """
        ext = Path(full_path).suffix.lower()
        mime_type = _MIME_MAP.get(ext)
        if mime_type is None:
            return None

        data = Path(full_path).read_bytes()
        max_size = MAX_PDF_SIZE if mime_type == "application/pdf" else MAX_IMAGE_SIZE
        if len(data) > max_size:
            raise ValueError(
                f"File too large: {len(data) / 1024 / 1024:.1f}MB "
                f"(max {max_size / 1024 / 1024:.0f}MB)"
            )

        # Image processing
        if mime_type.startswith("image/"):
            result = await preprocess_image(data, mime_type)  # type: ignore[arg-type]
            classification = classify_image_type(
                result.data,
                result.width,
                result.height,
                result.mime_type,  # type: ignore[arg-type]
            )
            chash = _content_hash(result.data)
            return ResolvedAttachment(
                id=full_path,
                mime_type=result.mime_type,
                data=result.data,
                original_size=len(data),
                processed_size=len(result.data),
                width=result.width,
                height=result.height,
                estimated_tokens=_estimate_image_tokens(result.width, result.height),
                preprocessing=result.steps,
                image_type=classification.type,
                content_hash=chash,
            )

        # PDF: raw buffer, rough token estimate (5KB per page, ~1500 tokens per page)
        return ResolvedAttachment(
            id=full_path,
            mime_type=mime_type,
            data=data,
            original_size=len(data),
            processed_size=len(data),
            estimated_tokens=math.ceil(len(data) / 5000) * 1500,
            preprocessing=[],
        )

    async def resolve_channel_attachment(
        self,
        att: ChannelAttachment,
        download_fn: Callable[[str], Awaitable[bytes]],
    ) -> ResolvedAttachment | None:
        """Resolve a channel attachment by downloading and preprocessing.

        Parameters
        ----------
        att:
            Channel attachment descriptor.
        download_fn:
            Async function that downloads file data given a file ID.

        Returns None for unsupported attachment types.
        Raises ``ValueError`` if the downloaded file exceeds size limits.
        """
        ext = Path(att.filename).suffix.lower() if att.filename else ""
        mime_type: AttachmentMimeType | None = _MIME_MAP.get(ext)
        if mime_type is None and att.mime_type in _SUPPORTED_MIMES:
            mime_type = att.mime_type  # type: ignore[assignment]
        if mime_type is None:
            log.warning(
                "unsupported_attachment_type",
                ext=ext,
                mime_type=att.mime_type,
                file_id=att.file_id,
            )
            return None

        data = await download_fn(att.file_id)
        max_size = MAX_PDF_SIZE if mime_type == "application/pdf" else MAX_IMAGE_SIZE
        if len(data) > max_size:
            raise ValueError(
                f"Channel attachment too large: {len(data) / 1024 / 1024:.1f}MB "
                f"(max {max_size / 1024 / 1024:.0f}MB)"
            )

        # Image processing
        if mime_type.startswith("image/"):
            result = await preprocess_image(data, mime_type)  # type: ignore[arg-type]
            classification = classify_image_type(
                result.data,
                result.width,
                result.height,
                result.mime_type,  # type: ignore[arg-type]
            )
            chash = _content_hash(result.data)
            return ResolvedAttachment(
                id=att.file_id,
                filename=att.filename,
                mime_type=result.mime_type,
                data=result.data,
                original_size=len(data),
                processed_size=len(result.data),
                width=result.width,
                height=result.height,
                estimated_tokens=_estimate_image_tokens(result.width, result.height),
                preprocessing=result.steps,
                image_type=classification.type,
                content_hash=chash,
            )

        # PDF
        return ResolvedAttachment(
            id=att.file_id,
            filename=att.filename,
            mime_type=mime_type,
            data=data,
            original_size=len(data),
            processed_size=len(data),
            estimated_tokens=math.ceil(len(data) / 5000) * 1500,
            preprocessing=[],
        )
