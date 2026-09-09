"""Find a local font covering the PDF's characters."""

from __future__ import annotations

import os
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)


def pdf_font(text: str) -> Path | None:
    """Return a covering font, or None for the built-in Latin font.

    RUNE_DOCUMENT_FONT overrides system font discovery. Raise if no candidate
    covers the text. Glyph coverage does not guarantee complex script shaping.
    """
    from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

    required = {ord(c) for c in text if c not in "\n\r\t"}
    configured = os.environ.get("RUNE_DOCUMENT_FONT")
    if not configured and all(c < 256 for c in required):
        return None
    candidates = [Path(configured).expanduser()] if configured else [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    if not configured:
        for directory in (Path.home() / ".local/share/fonts", Path("/usr/share/fonts/truetype")):
            if directory.is_dir():
                candidates.extend(sorted(directory.rglob("*.ttf")))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with TTFont(path, fontNumber=0, lazy=True) as font:
                if required.issubset((font.getBestCmap() or {}).keys()):
                    return path
        except Exception as exc:
            log.debug("document_font_unusable", path=str(path), error=str(exc))
    raise ValueError("No local font covers the document text. Set RUNE_DOCUMENT_FONT to a covering TTF/OTF/TTC font.")
