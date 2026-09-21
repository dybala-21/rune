"""Keep tool images intact until the provider serializes the request."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from rune.types import CapabilityResult

MAX_IMAGE_BYTES = 5 * 1024 * 1024
_IMAGE_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}


class CachedToolResult(str):
    """A read served from the run's cache, with its state-aware identity."""

    cache_key: str

    def __new__(cls, text: str, cache_key: str) -> CachedToolResult:
        result = super().__new__(cls, text)
        result.cache_key = cache_key
        return result


@dataclass(frozen=True, slots=True)
class ToolImage:
    data: str = field(repr=False)
    mime_type: str
    width: int
    height: int

    @classmethod
    def from_bytes(cls, data: bytes) -> ToolImage:
        from PIL import Image

        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image is empty or exceeds the 5 MiB delivery limit")
        try:
            with Image.open(BytesIO(data)) as image:
                mime_type = _IMAGE_TYPES.get(image.format or "")
                if mime_type is None or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Only static PNG, JPEG, GIF and WebP images are supported")
                width, height = image.size
                if width * height > 36_000_000:
                    raise ValueError("Image exceeds the pixel limit; capture a smaller viewport")
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Image data could not be decoded") from exc
        return cls(base64.b64encode(data).decode("ascii"), mime_type, width, height)

    def content(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": f"data:{self.mime_type};base64,{self.data}"}}


@dataclass(frozen=True, slots=True)
class ToolOutput:
    text: str
    images: tuple[ToolImage, ...] = ()

    def __str__(self) -> str:
        return self.text

    def with_text(self, text: str) -> ToolOutput:
        return replace(self, text=text)

    def content(self) -> list[dict[str, Any]]:
        return [{"type": "text", "text": self.text}, *(image.content() for image in self.images)]


def tool_content(result: str | ToolOutput) -> str | list[dict[str, Any]]:
    return result.content() if isinstance(result, ToolOutput) else result


def output_for_model(text: str, name: str, result: CapabilityResult) -> str | ToolOutput:
    if not result.success:
        return text
    metadata = result.metadata or {}
    paths: list[str] = []
    if name == "browser_screenshot":
        path = metadata.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("Screenshot returned no image path")
        paths = [path]
    elif name in {"browser_batch", "browser_workflow"}:
        paths = metadata.get("image_paths", [])
        if not isinstance(paths, list) or len(paths) > 2 or any(not isinstance(path, str) for path in paths):
            raise ValueError("Invalid browser screenshot paths")
    if paths:
        # Only browser screenshot producers authorize file reads. Page text,
        # shell output and remote metadata cannot ask us to open host files.
        images = []
        for path in paths:
            with Path(path).open("rb") as source:
                images.append(ToolImage.from_bytes(source.read(MAX_IMAGE_BYTES + 1)))
        return ToolOutput(text, tuple(images))
    encoded = metadata.get("image_base64") or metadata.get("imageBase64")
    if encoded is None:
        return text
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError("Invalid or oversized image data")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc
    return ToolOutput(text, (ToolImage.from_bytes(data),))
