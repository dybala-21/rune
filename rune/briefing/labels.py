"""The briefing's fixed wording, in the owner's language.

Every fact in a briefing is written by code from the accounts' own fields. The words around the
facts ("Today", "Unread mail") are these labels. English is built in. For another language the
model translates the list once: it is given the list and a language tag and nothing else, so
nothing a mail says reaches it. Its answer is checked label by label (same keys, placeholders kept,
no link and no number of its own) and kept until the list changes. If any label fails, the
briefing goes out in English.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from string import Formatter
from typing import Any

from rune.briefing import kept
from rune.briefing.sanitize import LINK, NUMBER, clean
from rune.utils.logger import get_logger

log = get_logger(__name__)

ENGLISH: dict[str, str] = {
    "title": "Morning briefing",
    "greeting": "Good morning.",
    "today": "Today",
    "all_day": "All day",
    "overlaps": "(overlaps another event)",
    "no_title": "(no title)",
    "no_events": "Nothing on the calendar.",
    "calendar_missing": "Calendar isn't connected.",
    "calendar_failed": "The calendar could not be read.",
    "mail": "Unread mail",
    "no_subject": "(no subject)",
    "unknown_sender": "unknown sender",
    "more_mail": "…and {count} more in Gmail.",
    "no_mail": "No new unread mail.",
    "mail_missing": "Gmail isn't connected.",
    "mail_failed": "Mail could not be read.",
    "connect_below": "Connect it with the button below.",
    "connect_calendar": "Connect Calendar",
    "connect_mail": "Connect Gmail",
    "headlines": "Headlines",
    "accounts_failed": "Connected accounts could not be reached, so the briefing wasn't made.",
    "nothing": "There was nothing to report, so no briefing was sent.",
}

# A BCP-47 tag's shape (a structured format): a language, then up to three subtags.
_TAG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")
_TIMEOUT = 45.0

_PROMPT = """You translate the fixed wording of a personal assistant's daily briefing.

You get a language tag (BCP-47) and a JSON object of English labels. Return the same keys, each
label written naturally in that language, in the polite register an assistant uses with the person
it works for, and about as short as the English. Keep every {placeholder} exactly as written, keep
product names (Gmail, Calendar) as they are, and add nothing: no numbers, links or notes."""


def tag(value: object) -> str:
    """A language tag in its kept form, or "" when the value isn't one."""
    return value.strip().lower() if isinstance(value, str) and _TAG.fullmatch(value.strip()) else ""


def language() -> str:
    """The language the owner asked for the briefing in; else the configured one; "" when neither says."""
    chosen = tag(kept.read("language").get("language"))
    if chosen:
        return chosen
    from rune.config.loader import get_config

    # The config's default is "en" whether or not anyone chose it, so only another value counts.
    configured = tag(get_config().general.locale)
    return configured if configured.split("-", 1)[0] != "en" else ""


def set_language(value: str) -> str:
    chosen = tag(value)
    if not chosen:
        raise ValueError("Use a language tag such as ko, en or ja")
    if chosen != tag(kept.read("language").get("language")):
        kept.write("language", {"language": chosen})
    return chosen


def _source() -> str:
    return hashlib.sha256(json.dumps(ENGLISH, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(text) if name is not None}


def checked(data: object) -> dict[str, str]:
    """The translated labels if every one can stand in for its English label; raises ValueError otherwise."""
    if not isinstance(data, dict) or set(data) != set(ENGLISH):
        raise ValueError("keys")
    labels: dict[str, str] = {}
    for key, english in ENGLISH.items():
        label = clean(data[key], 400)
        try:
            fields = _fields(label)
        except ValueError as exc:
            raise ValueError(key) from exc
        # A label carries no fact: a number, a link or another placeholder in it is the model's own.
        if (not label or len(label) > max(40, len(english) * 4) or LINK in label or NUMBER in label
                or fields != _fields(english) or any(c.isdigit() for c in label)):
            raise ValueError(key)
        labels[key] = label
    return labels


async def _translate(language_tag: str) -> dict[str, str]:
    from rune.agent.classification_response import decode_object
    from rune.llm.client import get_llm_client
    from rune.types import ModelTier

    schema: dict[str, Any] = {"type": "object", "additionalProperties": False, "required": list(ENGLISH),
                              "properties": {key: {"type": "string"} for key in ENGLISH}}
    async with asyncio.timeout(_TIMEOUT):
        response = await get_llm_client().completion(
            messages=[{"role": "system", "content": _PROMPT},
                      {"role": "user", "content": json.dumps({"language": language_tag, "labels": ENGLISH},
                                                             ensure_ascii=False)}],
            tier=ModelTier.FAST, max_tokens=2048, timeout=_TIMEOUT, max_retries=1,
            response_format={"type": "json_schema", "json_schema": {"name": "briefing_labels", "strict": True,
                                                                    "schema": schema}})
    return checked(decode_object(response))


async def load(language_tag: str) -> dict[str, str]:
    """The labels in ``language_tag``; English when it is English, unset, or can't be had."""
    language_tag = tag(language_tag)
    if not language_tag or language_tag.split("-", 1)[0] == "en":
        return ENGLISH
    name, source = f"labels-{language_tag}", _source()
    saved = kept.read(name)
    if saved.get("source") == source:
        try:
            return checked(saved.get("labels"))
        except ValueError as exc:
            log.warning("briefing_labels_unusable", language=language_tag, label=str(exc))
    try:
        labels = await _translate(language_tag)
    except Exception as exc:
        # Not kept, so the next briefing asks again.
        log.warning("briefing_labels_failed", language=language_tag, error=type(exc).__name__, label=str(exc)[:40])
        return ENGLISH
    try:
        kept.write(name, {"source": source, "labels": labels})
    except OSError as exc:
        log.warning("briefing_labels_not_kept", language=language_tag, error=type(exc).__name__)
    return labels
