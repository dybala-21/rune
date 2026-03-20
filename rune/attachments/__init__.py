"""RUNE attachments - classify, preprocess, resolve, and cache attachments."""

from rune.attachments.classifier import (
    ImageClassification,
    ImageType,
    OptimalFormat,
    classify_image_type,
    get_optimal_format,
)
from rune.attachments.preprocessor import preprocess_image
from rune.attachments.resolver import AttachmentResolver
from rune.attachments.types import (
    AttachmentMimeType,
    ChannelAttachment,
    ImageMimeType,
    ResolvedAttachment,
)
from rune.attachments.vision_cache import (
    VisionCache,
    VisionCacheEntry,
    VisionCacheStats,
    extract_image_summary,
)

__all__ = [
    # types
    "AttachmentMimeType",
    "ChannelAttachment",
    "ImageMimeType",
    "ResolvedAttachment",
    # classifier
    "ImageClassification",
    "ImageType",
    "OptimalFormat",
    "classify_image_type",
    "get_optimal_format",
    # preprocessor
    "preprocess_image",
    # resolver
    "AttachmentResolver",
    # vision_cache
    "VisionCache",
    "VisionCacheEntry",
    "VisionCacheStats",
    "extract_image_summary",
]
