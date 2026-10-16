"""Numeric values for comparing search excerpts with source pages."""

import re
from decimal import Decimal, InvalidOperation

_URL = re.compile(r"https?://[^\s<>\]\)\"']+")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\d.])")


def numeric_literals(text: str) -> set[Decimal]:
    """Parse numbers, excluding URLs and list markers."""
    text = _LIST_MARKER.sub("", _URL.sub("", text))
    values = set()
    for match in _NUMBER.finditer(text):
        if len(match.group()) > 128:
            continue
        try:
            value = Decimal(match.group().replace(",", ""))
        except InvalidOperation:
            continue
        if abs(value.as_tuple().exponent) <= 1000:
            values.add(value)
    return values
