"""Compare explicit numeric and time values without interpreting prose."""

import re
from datetime import time
from decimal import Decimal
from typing import Literal

ValueType = Literal["text", "number", "time"]
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_TIME = re.compile(r"[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,6})?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?")


def normalized(value: str, value_type: ValueType):
    if value_type == "text":
        return value
    if value_type == "number" and _NUMBER.fullmatch(value):
        return Decimal(value)
    if value_type == "time" and _TIME.fullmatch(value):
        try:
            return time.fromisoformat(value).isoformat()
        except ValueError:
            return None
    return None


def equivalent(actual: str | None, expected: str, value_type: ValueType = "text") -> bool:
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    left, right = normalized(actual, value_type), normalized(expected, value_type)
    return left is not None and right is not None and left == right
