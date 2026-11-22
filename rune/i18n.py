"""The core's own fixed wording, in the owner's language: one catalog per language in rune/locales.

English (`en.json`) is the source: every other catalog has its keys. A label may use any of the
{values} the code passes for it (a language may write a month as its number, not its name); one
that names anything else, or that doesn't parse, gets the English. Tests check every catalog. A language without a catalog, or a key a catalog lacks, gets the
English. Wording the model writes is never here; this is only what code says when the model can't
(the month's model budget used up) or shouldn't.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from string import Formatter

SOURCE = "en"


@cache
def catalog(language: str) -> dict[str, str]:
    """The catalog for a language's primary subtag ("ko-KR" reads ko.json); {} when there is none."""
    name = (language or "").split("-", 1)[0].strip().lower()
    if not name.isalpha():
        return {}
    try:
        text = resources.files("rune.locales").joinpath(f"{name}.json").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}  # a broken catalog reads English rather than breaking the answer
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(data, dict) else {}


def placeholders(text: str) -> set[str]:
    return {field for _, field, _, _ in Formatter().parse(text) if field}


def text(key: str, language: str = SOURCE, **values: object) -> str:
    """`key` in `language`, else in English, with `values` filled in."""
    found = catalog(language).get(key)
    if found is not None:
        try:
            if placeholders(found) <= set(values):
                return found.format(**values)
        except (ValueError, IndexError):
            pass
    return catalog(SOURCE)[key].format(**values)
