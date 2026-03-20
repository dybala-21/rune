"""Tests for rune.attachments.classifier — heuristic image classification."""

from __future__ import annotations

import struct

from rune.attachments.classifier import (
    classify_image_type,
    get_optimal_format,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_with_exif() -> bytes:
    """Minimal JPEG buffer with an EXIF APP1 marker."""
    buf = bytearray(30)
    buf[0] = 0xFF
    buf[1] = 0xD8  # SOI
    buf[2] = 0xFF
    buf[3] = 0xE1  # APP1
    struct.pack_into(">H", buf, 4, 20)  # segment length
    buf[6] = 0x45  # 'E'
    buf[7] = 0x78  # 'x'
    buf[8] = 0x69  # 'i'
    buf[9] = 0x66  # 'f'
    buf[10] = 0x00
    buf[11] = 0x00
    return bytes(buf)


def _jpeg_no_exif() -> bytes:
    """Minimal JPEG buffer without EXIF."""
    buf = bytearray(20)
    buf[0] = 0xFF
    buf[1] = 0xD8  # SOI
    buf[2] = 0xFF
    buf[3] = 0xDB  # DQT (not APP1)
    struct.pack_into(">H", buf, 4, 10)
    return bytes(buf)


def _low_color_buffer() -> bytes:
    """Buffer with low colour diversity (diagram-like)."""
    pattern = bytes([0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF])
    header = bytes(128)
    return header + pattern * 50


# ---------------------------------------------------------------------------
# classify_image_type
# ---------------------------------------------------------------------------


class TestClassifyImageType:
    def test_screenshot_16_9(self):
        result = classify_image_type(bytes(100), 1920, 1080)
        assert result.type == "screenshot"
        assert result.confidence >= 0.85

    def test_screenshot_16_10(self):
        result = classify_image_type(bytes(100), 1440, 900)
        assert result.type == "screenshot"
        assert result.confidence >= 0.85

    def test_screenshot_exact_resolution(self):
        result = classify_image_type(bytes(100), 2560, 1440)
        assert result.type == "screenshot"

    def test_small_image_not_screenshot(self):
        result = classify_image_type(bytes(100), 320, 180)
        assert result.type != "screenshot"

    def test_photo_by_exif(self):
        buf = _jpeg_with_exif()
        result = classify_image_type(buf, 4032, 3024, "image/jpeg")
        assert result.type == "photo"
        assert result.reason == "exif-present"

    def test_no_photo_without_exif(self):
        buf = _jpeg_no_exif()
        result = classify_image_type(buf, 400, 300, "image/jpeg")
        assert result.type != "photo"

    def test_low_color_buffer_diagram_or_text_heavy(self):
        buf = _low_color_buffer()
        result = classify_image_type(buf, 800, 600, "image/png")
        assert result.type in ("diagram", "text-heavy")

    def test_large_jpeg_no_exif_possible_photo(self):
        buf = bytearray(2000)
        buf[0] = 0xFF
        buf[1] = 0xD8
        buf[2] = 0xFF
        buf[3] = 0xDB
        struct.pack_into(">H", buf, 4, 10)
        for i in range(20, len(buf)):
            buf[i] = (i * 37 + 127) % 256
        result = classify_image_type(bytes(buf), 3000, 2000, "image/jpeg")
        assert result.type == "photo"
        assert result.confidence == 0.5
        assert result.reason == "large-jpeg-no-exif"

    def test_zero_dimensions_returns_unknown(self):
        result = classify_image_type(bytes(10), 0, 0)
        assert result.type == "unknown"
        assert result.confidence == 0

    def test_unrecognized_image_returns_unknown(self):
        buf = bytes(range(256)) * 2  # 512 bytes varied data
        result = classify_image_type(buf, 500, 500, "image/png")
        assert result.type == "unknown"


# ---------------------------------------------------------------------------
# get_optimal_format
# ---------------------------------------------------------------------------


class TestGetOptimalFormat:
    def test_screenshot_png(self):
        fmt = get_optimal_format("screenshot")
        assert fmt.mime_type == "image/png"
        assert fmt.quality is None

    def test_photo_jpeg_80(self):
        fmt = get_optimal_format("photo")
        assert fmt.mime_type == "image/jpeg"
        assert fmt.quality == 80

    def test_diagram_png(self):
        fmt = get_optimal_format("diagram")
        assert fmt.mime_type == "image/png"

    def test_text_heavy_png(self):
        fmt = get_optimal_format("text-heavy")
        assert fmt.mime_type == "image/png"

    def test_unknown_jpeg_85(self):
        fmt = get_optimal_format("unknown")
        assert fmt.mime_type == "image/jpeg"
        assert fmt.quality == 85
