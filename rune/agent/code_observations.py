"""Bound review context while retaining changes across files."""

import json
import os

from rune.utils.logger import get_logger

log = get_logger(__name__)
MAX_RECORDS = 12
MAX_RECORD_CHARS = 1_800
MAX_REVIEW_CHARS = 10_800
_OMITTED = "\n[excerpt omitted]\n"


def excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = (limit - len(_OMITTED)) * 2 // 3
    return text[:head] + _OMITTED + text[-(limit - len(_OMITTED) - head):]


def observation(identity: int, name: str, arguments: dict | str, output: str,
                success: bool | None = None) -> dict:
    if isinstance(arguments, dict):
        params = arguments
        arguments = json.dumps(arguments, ensure_ascii=False, default=str)
    else:
        try:
            params = json.loads(arguments)
        except (ValueError, TypeError) as exc:
            log.debug("claim_observation_arguments_unavailable", error=type(exc).__name__)
            params = {}
    path = params.get("path", params.get("file_path", "")) if isinstance(params, dict) else ""
    path = os.path.normpath(path) if isinstance(path, str) and path else ""
    header = f"{name}({excerpt(arguments, 1_000)})\nObserved result:\n"
    if success is not None:
        header = f"success={success}\n" + header
    text = header + excerpt(output, MAX_RECORD_CHARS - len(header))
    return {"id": identity, "text": text, "name": name, "path": path, "success": success}


def retain_observations(records: list[dict]) -> list[dict]:
    if len(records) <= MAX_RECORDS:
        return records
    first_reads, writes, latest_reads = {}, {}, {}
    for record in records:
        path, name = record.get("path"), record.get("name")
        if not path:
            continue
        if name == "file_read" and record.get("success") is not False:
            first_reads.setdefault(path, record)
            latest_reads[path] = record
        elif name in {"file_edit", "file_write"} and record.get("success") is not False:
            writes[path] = record
    priorities = [records[-1], *reversed(list(writes.values())),
                  *first_reads.values(), *reversed(list(latest_reads.values())), records[0], *reversed(records)]
    kept = {}
    for record in priorities:
        kept.setdefault(record["id"], record)
        if len(kept) == MAX_RECORDS:
            break
    return sorted(kept.values(), key=lambda record: record["id"])


def review_observations(records: list[dict]) -> list[dict]:
    records = retain_observations(records)
    low, high = 0, MAX_RECORD_CHARS
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(len(record["text"]), middle) for record in records) <= MAX_REVIEW_CHARS:
            low = middle
        else:
            high = middle - 1
    return [{"id": record["id"], "text": excerpt(record["text"], low)} for record in records]
