"""Headlines the host fetched for this owner's briefing.

Reading mail closes this machine's open web until the owner opens it again, and that lasts across
runs, so the machine can't fetch news itself on the morning after. The host fetches its feeds and
hands the headlines over (`briefing.news`) shortly before the briefing; they are kept here.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rune.briefing.sanitize import clean
from rune.utils.logger import get_logger

log = get_logger(__name__)

MAX_ITEMS = 30
# Older than this, they are yesterday's news: the briefing goes without.
MAX_AGE_SECONDS = 36 * 3600


@dataclass(frozen=True, slots=True)
class Headline:
    title: str
    source: str
    url: str
    published: str = ""


def _path() -> Path:
    from rune.utils.paths import rune_data

    return rune_data() / "briefing" / "news.json"


def _headline(item: object) -> Headline | None:
    if not isinstance(item, dict):
        return None
    title, url = clean(item.get("title"), 200), item.get("url")
    if not title or not isinstance(url, str) or not url.startswith("https://") or len(url) > 2048 or any(c.isspace() for c in url):
        return None
    published = item.get("published")
    return Headline(title=title, source=clean(item.get("source"), 60), url=url,
                    published=published[:40] if isinstance(published, str) else "")


def store(items: object) -> int:
    """Keep the headlines handed over; returns how many were usable. Replaces the last set whole."""
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    kept = [h for h in map(_headline, items[: MAX_ITEMS * 2]) if h is not None][:MAX_ITEMS]
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"received_at": time.time(), "items": [asdict(h) for h in kept]}, stream, ensure_ascii=False)
    os.replace(partial, path)
    log.info("briefing_news_stored", items=len(kept), offered=len(items))
    return len(kept)


def load(now: float | None = None) -> list[Headline]:
    """The last headlines handed over, or none when they are missing or too old."""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.warning("briefing_news_unreadable", error=type(exc).__name__)
        return []
    received = data.get("received_at") if isinstance(data, dict) else None
    if not isinstance(received, int | float) or (now or time.time()) - received > MAX_AGE_SECONDS:
        return []
    return [h for h in map(_headline, data.get("items") or []) if h is not None]
