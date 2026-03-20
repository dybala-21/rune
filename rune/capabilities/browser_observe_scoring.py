"""Browser observation scoring for RUNE.

Ported from src/capabilities/browser-observe-scoring.ts - scores
elements returned by browser.observe to prioritise the most relevant
interactive elements for the current task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Types

@dataclass(slots=True)
class ObserveElement:
    """Minimal element descriptor for scoring."""
    role: str
    name: str
    tag_name: str
    input_type: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


# Pattern constants

_CALENDAR_DOMAIN = re.compile(
    r"캘린더|calendar|일정|schedule|meeting|미팅|회의", re.IGNORECASE,
)
_CALENDAR_WRITE_ACTION = re.compile(
    r"등록|생성|추가|수정|변경|삭제|예약|만들|작성|save|create|add|update|delete|book|schedule",
    re.IGNORECASE,
)
_CALENDAR_NAV = re.compile(
    r"다음\s*(주|달)|이전\s*(주|달)|next\s*(week|month)|prev(?:ious)?\s*(week|month)",
    re.IGNORECASE,
)
_DAY_HEADER = re.compile(r"^(일|월|화|수|목|금|토)\s*\d{1,2}$", re.IGNORECASE)
_TIME_SLOT = re.compile(
    r"(?:오전|오후|\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b\d{1,2}\s*시\b|\bam\b|\bpm\b)",
    re.IGNORECASE,
)
_CREATE_ENTRY = re.compile(
    r"만들기|create|new event|새 일정|추가|작성|event", re.IGNORECASE,
)
_APPOINTMENT_PAGE = re.compile(
    r"예약 가능한|appointment|booking page|약속 일정", re.IGNORECASE,
)

# Synonym map for task-hint keyword expansion
_HINT_SYNONYMS: dict[str, list[str]] = {
    "리뷰": ["리뷰", "review", "후기", "평점", "rating"],
    "로그인": ["로그인", "login", "sign in", "로그", "계정"],
    "검색": ["검색", "search", "찾기", "find"],
    "예약": ["예약", "book", "reserve", "결제", "payment"],
    "가격": ["가격", "price", "요금", "rate", "비용"],
    "필터": ["필터", "filter", "정렬", "sort", "조건"],
    "장바구니": ["장바구니", "cart", "basket", "담기"],
    "메일": ["메일", "mail", "email", "받은편지함", "inbox", "보낸편지함", "sent"],
    "대화": ["대화", "conversation", "thread", "message", "메시지"],
    "일정": ["일정", "캘린더", "calendar", "meeting", "미팅", "회의", "event", "schedule"],
}


# Helpers

def is_calendar_event_write_task(task_hint: str | None) -> bool:
    """Return ``True`` when *task_hint* describes a calendar write action."""
    if not task_hint:
        return False
    lower = task_hint.lower()
    return bool(_CALENDAR_DOMAIN.search(lower) and _CALENDAR_WRITE_ACTION.search(lower))


def build_task_keywords(task_hint: str | None) -> list[str]:
    """Expand *task_hint* into a deduplicated list of search keywords."""
    if not task_hint:
        return []

    lower = task_hint.lower()
    words = [w for w in re.split(r"[\s,./]+", lower) if len(w) >= 2]
    keywords: set[str] = set(words)

    for key, synonyms in _HINT_SYNONYMS.items():
        if key in lower or any(s in lower for s in synonyms):
            keywords.update(synonyms)

    return list(keywords)


# Scoring

def score_observe_element(
    el: ObserveElement,
    task_keywords: list[str],
    task_hint: str | None = None,
) -> int:
    """Score an element for relevance to the current task.

    Higher scores mean the element is more likely to be useful.
    """
    score = 0

    # Base role scores
    if el.role in ("searchbox", "textbox", "combobox"):
        score += 100
    if el.tag_name in ("input", "textarea", "select"):
        score += 80
    if el.input_type == "search":
        score += 50
    if el.role == "button":
        score += 40
    if el.role in ("checkbox", "radio", "switch"):
        score += 30
    if el.role in ("option", "menuitem"):
        score += 25
    if el.role == "tab":
        score += 20
    if el.role == "link":
        score += 30
    if el.role in ("row", "gridcell", "listitem"):
        score += 35

    # Text matching
    el_text = " ".join([
        el.name,
        el.attributes.get("placeholder", ""),
        el.attributes.get("href", ""),
        el.attributes.get("aria-label", ""),
    ]).lower()

    if task_keywords:
        for kw in task_keywords:
            if kw in el_text:
                score += 200
                break

    # Calendar-specific scoring
    if is_calendar_event_write_task(task_hint):
        if el.role in ("row", "gridcell", "listitem"):
            score += 140
        if _TIME_SLOT.search(el_text):
            score += 120
        if _CREATE_ENTRY.search(el_text):
            score += 220

        if _CALENDAR_NAV.search(el_text):
            score -= 220
        if _DAY_HEADER.search(el.name.strip()):
            score -= 200
        if _APPOINTMENT_PAGE.search(el_text) and not re.search(
            r"예약 가능한|appointment", task_hint or "", re.IGNORECASE,
        ):
            score -= 260
        href = el.attributes.get("href", "")
        if el.role == "link" and re.search(r"/(week|month|day)/", href, re.IGNORECASE):
            score -= 180

    return score


def get_observe_role_order(
    has_dropdown_or_dialog: bool,
    task_hint: str | None = None,
) -> list[str]:
    """Return the preferred element role order for observation."""
    if has_dropdown_or_dialog:
        return [
            "option", "menuitem", "listbox", "searchbox", "textbox",
            "combobox", "button", "link", "checkbox", "radio", "tab", "generic",
        ]

    if is_calendar_event_write_task(task_hint):
        return [
            "gridcell", "row", "listitem", "button", "textbox", "combobox",
            "searchbox", "menuitem", "option", "link", "tab", "checkbox",
            "radio", "generic",
        ]

    return [
        "searchbox", "textbox", "combobox", "button", "link", "checkbox",
        "radio", "tab", "menuitem", "option", "generic",
    ]
