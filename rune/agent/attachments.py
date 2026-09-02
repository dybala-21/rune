"""Turn user attachments into a message the model can actually read.

Attachments arrive from the UI as ``{name, mimeType, data}`` with *data* being
raw base64. Images are downscaled by ``rune.attachments.preprocess_image`` and
become OpenAI-style ``image_url`` parts (litellm translates these for every
provider); anything else is named in the text so the model knows it exists but
is never sent as content a provider would reject.

The built message is seeded once, before the loop starts. It must not be
rebuilt per step: the conversation is resent on every step, so re-encoding
there would add the image cost to each one.
"""

from __future__ import annotations

import base64
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Providers reject images well below the UI's 20MB file cap — Anthropic and
# OpenAI both stop around 5MB of base64. Checked after preprocessing, since
# that is what actually goes on the wire.
MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024


def _is_image(mime_type: str) -> bool:
    return mime_type.lower().startswith("image/")


async def _shrink(data_b64: str, mime: str) -> tuple[str, str]:
    """Downscale an image and return ``(base64, mime)``.

    A phone photo is several MB and far larger than any model's vision input
    resolution, so sending it whole just buys latency and tokens. Falls back to
    the original bytes if decoding or preprocessing fails.
    """
    try:
        from rune.attachments import preprocess_image

        raw = base64.b64decode(data_b64, validate=True)
        result = await preprocess_image(raw, mime)  # type: ignore[arg-type]
        if len(result.data) < len(raw):
            log.info(
                "attachment_preprocessed",
                before=len(raw), after=len(result.data), steps=",".join(result.steps),
            )
            return base64.b64encode(result.data).decode(), result.mime_type
    except Exception as exc:
        log.debug("attachment_preprocess_skipped", error=str(exc)[:100])
    return data_b64, mime


async def build_user_content(
    goal: str,
    attachments: list[dict[str, Any]],
    *,
    vision: bool,
) -> tuple[str | list[dict[str, Any]], list[str]]:
    """Build the user message content for *goal* plus *attachments*.

    Returns ``(content, notes)``. *content* is a plain string when there is
    nothing to attach, otherwise a list of content parts. *notes* holds
    human-readable reasons an attachment was not sent, for the caller to
    surface — an unread image must never pass silently.
    """
    if not attachments:
        return goal, []

    parts: list[dict[str, Any]] = [{"type": "text", "text": goal}]
    notes: list[str] = []
    sent_images = 0

    for att in attachments:
        name = str(att.get("name") or "attachment")
        mime = str(att.get("mimeType") or "")
        data = att.get("data") or ""

        if not data:
            notes.append(f"{name}: empty file, not sent")
            continue

        if not _is_image(mime):
            notes.append(f"{name} ({mime or 'unknown type'}): not an image, not sent to the model")
            continue

        if not vision:
            notes.append(f"{name}: this model can't read images")
            continue

        try:
            base64.b64decode(data, validate=True)
        except Exception:
            notes.append(f"{name}: file data was corrupted in transit, not sent")
            continue

        data, mime = await _shrink(data, mime)

        if len(data) > MAX_IMAGE_BASE64_BYTES:
            mb = len(data) / (1024 * 1024)
            limit = MAX_IMAGE_BASE64_BYTES // (1024 * 1024)
            notes.append(f"{name}: {mb:.1f}MB is over the {limit}MB image limit")
            continue

        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        })
        sent_images += 1

    log.info("attachments_prepared", sent=sent_images, skipped=len(notes))

    # Nothing survived — a plain string, so providers that dislike single-part
    # arrays are unaffected. The caller still seeds it, so the model learns a
    # file was attached and why it went unread.
    if sent_images == 0:
        text = goal
        if notes:
            text += "\n\n[Attachments not sent: " + "; ".join(notes) + "]"
        return text, notes

    # The note goes in its own part. Keeping parts[0] equal to the bare goal is
    # what lets the adapter recognise the goal is already in history; otherwise
    # it appends a second, text-only copy on every step.
    if notes:
        parts.append({
            "type": "text",
            "text": "[Attachments not sent: " + "; ".join(notes) + "]",
        })
    return parts, notes


def content_text(content: Any) -> str:
    """The text of a message whose content may be a multimodal part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return str(part.get("text") or "")
    return ""
