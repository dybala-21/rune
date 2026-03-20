"""Image preprocessor.

Ported from src/attachments/preprocessor.ts - resizes images, strips EXIF,
and falls back gracefully when Pillow is not installed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from rune.attachments.types import ImageMimeType
from rune.utils.logger import get_logger

log = get_logger(__name__)

MAX_LONG_EDGE = 1568

# Types


@dataclass(slots=True)
class PreprocessResult:
    data: bytes
    mime_type: ImageMimeType
    width: int
    height: int
    steps: list[str] = field(default_factory=list)


# Dimension parsing (no dependencies)

def parse_png_dimensions(buf: bytes) -> tuple[int, int] | None:
    """Parse width/height from a PNG IHDR chunk (offset 16-23)."""
    if len(buf) < 24 or buf[0] != 0x89 or buf[1] != 0x50:
        return None
    w = struct.unpack(">I", buf[16:20])[0]
    h = struct.unpack(">I", buf[20:24])[0]
    return w, h


def parse_jpeg_dimensions(buf: bytes) -> tuple[int, int] | None:
    """Parse width/height from JPEG SOF0/SOF2 markers."""
    i = 2
    while i < len(buf) - 9:
        if buf[i] == 0xFF and buf[i + 1] in (0xC0, 0xC2):
            h = struct.unpack(">H", buf[i + 5 : i + 7])[0]
            w = struct.unpack(">H", buf[i + 7 : i + 9])[0]
            return w, h
        if buf[i] == 0xFF:
            seg_len = struct.unpack(">H", buf[i + 2 : i + 4])[0]
            i += 2 + seg_len
        else:
            i += 1
    return None


def parse_webp_dimensions(buf: bytes) -> tuple[int, int] | None:
    """Parse width/height from a WebP RIFF header."""
    if len(buf) < 30:
        return None
    # VP8 simple
    if buf[12:16] == b"VP8 ":
        w = struct.unpack("<H", buf[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", buf[28:30])[0] & 0x3FFF
        return w, h
    # VP8L
    if buf[12:16] == b"VP8L":
        bits = struct.unpack("<I", buf[21:25])[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return w, h
    return None


def parse_gif_dimensions(buf: bytes) -> tuple[int, int] | None:
    """Parse width/height from GIF header."""
    if len(buf) < 10:
        return None
    w = struct.unpack("<H", buf[6:8])[0]
    h = struct.unpack("<H", buf[8:10])[0]
    return w, h


def parse_dimensions_fallback(
    buf: bytes,
    mime_type: ImageMimeType,
) -> tuple[int, int] | None:
    """Extract dimensions from image headers without any external libraries."""
    parsers = {
        "image/png": parse_png_dimensions,
        "image/jpeg": parse_jpeg_dimensions,
        "image/webp": parse_webp_dimensions,
        "image/gif": parse_gif_dimensions,
    }
    parser = parsers.get(mime_type)
    return parser(buf) if parser else None


# Public API

async def preprocess_image(
    data: bytes,
    mime_type: ImageMimeType,
) -> PreprocessResult:
    """Preprocess an image for LLM consumption.

    - GIF: preserved as-is (animated content).
    - With Pillow: auto-orient, resize to max 1568px long edge.
    - Without Pillow: returns raw buffer with header-parsed dimensions.
    """
    steps: list[str] = []

    # GIF: preserve animated content
    if mime_type == "image/gif":
        dims = parse_dimensions_fallback(data, mime_type)
        return PreprocessResult(
            data=data,
            mime_type=mime_type,
            width=dims[0] if dims else 0,
            height=dims[1] if dims else 0,
            steps=["gif-preserved"],
        )

    # Try Pillow (optional dependency)
    try:
        from io import BytesIO

        from PIL import Image, ImageOps

        img = Image.open(BytesIO(data))

        # Auto-orient from EXIF
        img = ImageOps.exif_transpose(img) or img
        steps.append("exif-stripped")

        orig_w, orig_h = img.size
        long_edge = max(orig_w, orig_h)

        # Resize if long edge exceeds limit
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            new_w = round(orig_w * scale)
            new_h = round(orig_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            steps.append(f"resized:{orig_w}x{orig_h}->{new_w}x{new_h}")

        # Save to buffer in original format
        buf = BytesIO()
        fmt_map = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/webp": "WEBP",
        }
        save_fmt = fmt_map.get(mime_type, "PNG")
        save_kwargs = {}
        if save_fmt == "JPEG":
            save_kwargs["quality"] = 85
            # Ensure RGB for JPEG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

        img.save(buf, format=save_fmt, **save_kwargs)
        result_data = buf.getvalue()
        final_w, final_h = img.size

        return PreprocessResult(
            data=result_data,
            mime_type=mime_type,
            width=final_w,
            height=final_h,
            steps=steps,
        )

    except Exception:
        # Pillow not installed or processing failed - fallback
        log.debug("pillow_not_available_or_failed", fallback="raw_buffer")
        dims = parse_dimensions_fallback(data, mime_type)
        steps.append("no-pillow-fallback")
        return PreprocessResult(
            data=data,
            mime_type=mime_type,
            width=dims[0] if dims else 0,
            height=dims[1] if dims else 0,
            steps=steps,
        )
