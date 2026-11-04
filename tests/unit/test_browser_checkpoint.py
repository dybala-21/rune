import json
from copy import deepcopy

import pytest

from rune.agent.browser_checkpoint import checkpoint_properties
from rune.agent.browser_context import compact_browser_history

URL = "https://nol.yanolja.com/stay/overseas/TDP-1/list?checkInDate=2026-11-15"
OTHER = "https://nol.yanolja.com/stay/overseas/TDP-2/list?checkInDate=2026-11-15"
QUERY = {"city": "Sapporo", "count": 3}
PROGRESS = {"query": QUERY, "required": 3, "complete_properties": 1,
            "sources": [{"id": "source-a", "url": URL, "property_id": "one"},
                        {"id": "source-b", "url": OTHER, "property_id": "two"}],
            "candidates": [{"id": "rate-a", "property_id": "one", "missing": []},
                           {"id": "rate-b", "property_id": "two", "missing": ["taxes_quote"]}]}


def exchange(identity, name, args, result):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": identity,
         "content": result if isinstance(result, str) else json.dumps(result)}]


def trace():
    messages = [{"role": "system", "content": "No booking."}, {"role": "user", "content": "Compare three hotels."}]
    messages += exchange("old-run", "browser_stay_evidence", {"action": "read"}, {"source": "source-a", "lines": "Old task"})
    messages += exchange("begin", "browser_stay_evidence", {"action": "begin", "query": QUERY}, {"progress": {"query": QUERY}})
    messages += exchange("search", "browser_navigate", {"url": "https://nol.yanolja.com/search"},
                         "URL: https://nol.yanolja.com/search\nUnvisited hotel links")
    messages += exchange("nav", "browser_navigate", {"url": URL}, "URL: " + URL + "\nPage body " * 500)
    messages += exchange("capture", "browser_stay_evidence", {"action": "capture"},
                         {"source": "source-a", "url": URL, "lines": "Saved policy " * 1000})
    messages += exchange("pending", "browser_stay_evidence", {"action": "read", "source": "source-b"},
                         {"source": "source-b", "lines": "Not yet recorded"})
    messages += exchange("error", "browser_stay_evidence", {"action": "record"}, "[ERROR] Incorrect room quote")
    messages += exchange("current", "browser_observe", {}, "URL: " + URL + '\n--- Interactive Elements (1/1) ---\n[e0] button "Next"')
    messages += exchange("record", "browser_stay_evidence", {"action": "record"},
                         {"recorded": "rate-a", "missing": [], "progress": PROGRESS})
    return messages


def ids(messages):
    return {m["tool_call_id"] for m in messages if m.get("role") == "tool"}


def assert_paired(messages):
    calls = {call["id"] for m in messages for call in m.get("tool_calls", [])}
    assert calls == ids(messages)


def test_completed_property_checkpoint_keeps_discovery_errors_and_current_controls():
    original = trace()
    before = deepcopy(original)
    result = checkpoint_properties(original, PROGRESS)
    assert ids(result) == {"old-run", "begin", "search", "pending", "error", "current", "record"}
    assert result[:2] == original[:2] and original == before
    assert "Not yet recorded" in json.dumps(result)
    assert '[e0] button' in json.dumps(result)
    state = json.loads(result[-1]["content"])
    assert state["progress"] == PROGRESS and state["checkpoint"]
    assert len(json.dumps(result)) < len(json.dumps(original)) / 2
    assert checkpoint_properties(result, PROGRESS) == result
    assert_paired(result)


def test_only_trusted_browser_read_state_enables_checkpoint():
    messages = trace()
    ordinary = compact_browser_history(messages)
    assert "capture" in ids(ordinary)
    compacted = compact_browser_history(messages, retrieval_progress=PROGRESS)
    assert "capture" not in ids(compacted)
    assert "pending" in ids(compacted)
    assert_paired(compacted)


def test_checkpoint_survives_prior_inventory_compaction():
    messages = trace()
    prefix = compact_browser_history(messages[:-2])
    result = checkpoint_properties(prefix + messages[-2:], PROGRESS)
    assert "capture" not in ids(result)
    assert json.loads(result[-1]["content"])["progress"]["sources"] == PROGRESS["sources"]


@pytest.mark.parametrize("change", ["no_begin", "failed_begin", "incomplete", "failed_record", "unowned_record"])
def test_incomplete_or_unowned_progress_never_authorizes_checkpoint(change):
    messages, progress = trace(), deepcopy(PROGRESS)
    if change == "no_begin":
        messages = [m for m in messages if m.get("tool_call_id") != "begin"
                    and not any(c["id"] == "begin" for c in m.get("tool_calls", []))]
    elif change == "failed_begin":
        next(m for m in messages if m.get("tool_call_id") == "begin")["content"] = "[ERROR] Contract mismatch"
    elif change == "incomplete":
        progress["candidates"][0]["missing"] = ["taxes_quote"]
    elif change == "failed_record":
        messages[-1]["content"] = "[ERROR] Rate does not match"
    else:
        messages[-1]["content"] = '{"recorded":"invented","missing":[]}'
    assert checkpoint_properties(messages, progress) == messages


def test_mixed_tool_batch_is_retained_as_a_whole():
    messages = trace()
    index = next(i for i, m in enumerate(messages) if any(c["id"] == "capture" for c in m.get("tool_calls", [])))
    file = exchange("file", "file_read", {"path": "important.txt"}, "Keep this separate task evidence")
    messages[index]["tool_calls"] += file[0]["tool_calls"]
    messages.insert(index + 2, file[1])
    result = checkpoint_properties(messages, PROGRESS)
    assert {"capture", "file"} <= ids(result)
    assert_paired(result)
