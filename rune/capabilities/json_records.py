"""Select or aggregate JSON records without model arithmetic."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, DecimalException, Inexact, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecordQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default="", description="JSON Pointer to the array; empty for root.")
    where: dict[str, str | int | float | bool | None] = Field(default_factory=dict, description="Exact field matches.")
    operation: Literal["latest", "sum", "mean", "min", "max", "count"]
    field: str = Field(default="", description="Order/aggregate field; omit for count.")


def record_paths(data) -> str:
    """Describe a few array paths without copying their values."""
    from itertools import islice

    pending, found, visited = [(data, "", 0)], [], 0
    while pending and visited < 64 and len(found) < 4:
        node, path, depth = pending.pop()
        visited += 1
        if len(path) > 200 or depth > 8:
            continue
        if isinstance(node, list) and node and isinstance(node[0], dict):
            fields = [key for key in islice(node[0], 8) if len(key) <= 80]
            found.append({"path": path, "records": len(node), "sampleFields": fields})
        elif isinstance(node, dict):
            for key, value in islice(node.items(), 16):
                if isinstance(value, (dict, list)) and len(key) <= 80:
                    pending.append((value, path + "/" + key.replace("~", "~0").replace("/", "~1"), depth + 1))
    return " Record arrays: " + json.dumps(found, ensure_ascii=False, separators=(",", ":")) if found else ""


def dump_json(value) -> str:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite number")
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + dump_json(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(dump_json(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _number(text: str) -> Decimal:
    value = Decimal(text)
    if not value.is_finite() or len(value.as_tuple().digits) > 128 or abs(value.as_tuple().exponent) > 128:
        raise ValueError("Number exceeds calculation limits")
    return value


def _timestamp(value) -> tuple[str, datetime]:
    if not isinstance(value, str):
        raise ValueError("Latest requires ISO date or timestamp strings")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if len(value) == 10:
        return "date", stamp.replace(tzinfo=UTC)
    if stamp.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return "timestamp", stamp.astimezone(UTC)


def query_records(text: str, query: RecordQuery, *, numeric_strings: bool = False) -> tuple[str, dict]:
    if len(text) > 1_000_000 or len(query.where) > 16:
        raise ValueError("Record query exceeds its size limit")
    try:
        node = json.loads(text, parse_int=_number, parse_float=_number, parse_constant=_number)
        scope = {}
        if query.path and not query.path.startswith("/"):
            raise ValueError("Record path must be a JSON Pointer")
        parts = query.path.split("/")[1:] if query.path else []
        if len(parts) > 16:
            raise ValueError("Record path is too deep")
        for part in parts:
            if any(char == "~" and (i + 1 == len(part) or part[i + 1] not in "01") for i, char in enumerate(part)):
                raise ValueError("Invalid JSON Pointer escape")
            if isinstance(node, dict):
                scope.update({k: v for k, v in node.items() if not isinstance(v, (dict, list))})
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list):
                if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                    raise ValueError("Invalid JSON Pointer array index")
                node = node[int(key)]
            else:
                node = node[key]
        if not isinstance(node, list) or len(node) > 10_000 or any(not isinstance(row, dict) for row in node):
            raise ValueError("Record path must identify an array of at most 10000 objects")
        rows = [{**scope, **row} for row in node]
        for row in rows:
            if any(key not in row for key in query.where):
                raise ValueError("A filter field is missing from a record")
        def matches(row):
            for key, expected in query.where.items():
                actual = row[key]
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    if not isinstance(actual, Decimal) or actual != _number(str(expected)):
                        return False
                elif type(actual) is not type(expected) or actual != expected:
                    return False
            return True
        rows = [row for row in rows if matches(row)]
        if query.operation == "count":
            result = {"operation": "count", "value": len(rows)}
        elif not rows or not query.field or any(query.field not in row for row in rows):
            raise ValueError("No matching records or a required field is missing")
        elif query.operation == "latest":
            stamps = [_timestamp(row[query.field]) for row in rows]
            if len({kind for kind, _ in stamps}) != 1:
                raise ValueError("Date-only records cannot be compared with timestamped records")
            latest = max(stamp for _, stamp in stamps)
            selected = [row for row, (_, stamp) in zip(rows, stamps, strict=True) if stamp == latest]
            if any(row != selected[0] for row in selected[1:]):
                raise ValueError("Conflicting records share the latest timestamp")
            result = selected[0]
        else:
            values = [row[query.field] for row in rows]
            if numeric_strings:
                values = [_number(value) if isinstance(value, str) and value.strip() else value for value in values]
            if any(not isinstance(v, Decimal) for v in values):
                raise ValueError("Aggregation requires numeric fields, not strings or booleans")
            with localcontext() as context:
                context.prec = 512
                context.traps[Inexact] = True
                total = sum(values, Decimal(0))
                value = {"sum": lambda: total, "mean": lambda: total / len(values),
                         "min": lambda: min(values), "max": lambda: max(values)}[query.operation]()
                if query.operation == "mean" and value * len(values) != total:
                    raise ValueError("The mean has a repeating decimal; use sum and count with explicit rounding")
            result = {"operation": query.operation, "field": query.field, "count": len(rows), "value": value}
        return dump_json(result), {"query": query.model_dump(), "records": len(node), "matched": len(rows),
                                   "scope": "retrieved_response"}
    except (KeyError, IndexError, TypeError, DecimalException, RecursionError) as exc:
        raise ValueError("Record query does not match the JSON structure") from exc
