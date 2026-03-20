"""Image type classifier - heuristic-based, no LLM calls, <1ms.

Ported from src/attachments/classifier.ts.

Classification criteria:
  screenshot - 16:9 or 16:10 aspect ratio + large resolution
  photo      - EXIF orientation present (JPEG)
  diagram    - low unique colour count (< 32, sampled)
  text-heavy - very low unique colours (< 16) + high contrast
  unknown    - default

Optimal format recommendations:
  screenshot -> PNG  (text clarity)
  photo      -> JPEG 80%  (size optimisation)
  diagram    -> PNG  (vector-like preservation)
  text-heavy -> PNG  (readability)
  unknown    -> JPEG 85%
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

from rune.attachments.types import ImageMimeType

# Types

ImageType = Literal["screenshot", "photo", "diagram", "text-heavy", "unknown"]


@dataclass(slots=True)
class ImageClassification:
    type: ImageType
    confidence: float  # 0-1
    reason: str


@dataclass(slots=True)
class OptimalFormat:
    mime_type: ImageMimeType
    quality: int | None = None  # JPEG quality (1-100), None for PNG


# Aspect ratio constants

_RATIO_16_9 = 16 / 9    # 1.778
_RATIO_16_10 = 16 / 10  # 1.6
_RATIO_TOLERANCE = 0.05


def _is_near(value: float, target: float, tolerance: float = _RATIO_TOLERANCE) -> bool:
    return abs(value - target) <= tolerance


# EXIF detection (JPEG only)

def _has_exif_marker(buf: bytes) -> bool:
    """Check for EXIF APP1 marker in a JPEG buffer."""
    if len(buf) < 12:
        return False
    # JPEG SOI
    if buf[0] != 0xFF or buf[1] != 0xD8:
        return False

    i = 2
    limit = min(len(buf) - 8, 65536)
    while i < limit:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker == 0xE1:
            # APP1 - check "Exif" signature
            if (
                i + 10 < len(buf)
                and buf[i + 4] == 0x45  # 'E'
                and buf[i + 5] == 0x78  # 'x'
                and buf[i + 6] == 0x69  # 'i'
                and buf[i + 7] == 0x66  # 'f'
            ):
                return True
        # Skip to next marker
        if i + 3 < len(buf):
            seg_len = struct.unpack(">H", buf[i + 2 : i + 4])[0]
            i += 2 + seg_len
        else:
            break
    return False


# Colour diversity sampling

def _sample_colour_diversity(buf: bytes) -> int:
    """Sample byte-triplet diversity from compressed image data.

    Skips header bytes and samples ~256 points across the buffer.
    Returns unique triplet count (0-256).
    """
    sample_count = 256
    step = max(1, len(buf) // sample_count)
    seen: set[int] = set()

    start = min(128, len(buf) // 10)
    i = start
    while i < len(buf) - 2 and len(seen) < 256:
        triplet = (buf[i] << 16) | (buf[i + 1] << 8) | buf[i + 2]
        seen.add(triplet)
        i += step

    return len(seen)


# Public API

def classify_image_type(
    buffer: bytes,
    width: int,
    height: int,
    mime_type: ImageMimeType | None = None,
) -> ImageClassification:
    """Classify an image based on its buffer and dimensions.

    Uses heuristics only (no LLM), completes in <1ms.
    """
    if width == 0 or height == 0:
        return ImageClassification(type="unknown", confidence=0.0, reason="no-dimensions")

    ratio = max(width, height) / min(width, height)
    is_landscape = width >= height
    total_pixels = width * height

    # 1. Screenshot detection: 16:9 or 16:10 + large resolution
    if is_landscape and total_pixels >= 500_000:
        if _is_near(ratio, _RATIO_16_9) or _is_near(ratio, _RATIO_16_10):
            return ImageClassification(
                type="screenshot",
                confidence=0.85,
                reason=f"display-ratio:{ratio:.2f}",
            )
        # Common screen resolutions
        if width in (1920, 2560, 1366, 1440, 3840):
            return ImageClassification(
                type="screenshot",
                confidence=0.9,
                reason=f"exact-resolution:{width}x{height}",
            )

    # 2. Photo detection: EXIF present (JPEG only)
    if mime_type == "image/jpeg" and _has_exif_marker(buffer):
        return ImageClassification(type="photo", confidence=0.8, reason="exif-present")

    # 3. Colour diversity analysis
    colour_diversity = _sample_colour_diversity(buffer)

    if colour_diversity < 16:
        return ImageClassification(
            type="text-heavy",
            confidence=0.75,
            reason=f"low-colors:{colour_diversity}",
        )

    if colour_diversity < 32:
        return ImageClassification(
            type="diagram",
            confidence=0.7,
            reason=f"few-colors:{colour_diversity}",
        )

    # 4. Large JPEG without EXIF -> possible photo (camera EXIF stripped)
    if mime_type == "image/jpeg" and total_pixels >= 2_000_000:
        return ImageClassification(
            type="photo",
            confidence=0.5,
            reason="large-jpeg-no-exif",
        )

    return ImageClassification(type="unknown", confidence=0.0, reason="no-match")


def get_optimal_format(image_type: ImageType) -> OptimalFormat:
    """Return the recommended output format for a given image type."""
    if image_type == "screenshot":
        return OptimalFormat(mime_type="image/png")
    if image_type == "photo":
        return OptimalFormat(mime_type="image/jpeg", quality=80)
    if image_type == "diagram":
        return OptimalFormat(mime_type="image/png")
    if image_type == "text-heavy":
        return OptimalFormat(mime_type="image/png")
    # unknown
    return OptimalFormat(mime_type="image/jpeg", quality=85)
