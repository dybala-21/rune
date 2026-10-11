"""Changing views must not erase the data needed to complete a browser task."""

from copy import deepcopy

from rune.agent.browser_context import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_SNAPSHOTS,
    MAX_SNAPSHOT_CHARS,
    compact_browser_history,
)

AVAILABILITY = ('URL: http://localhost/booking\n- document:\n'
                '  - table:\n    - row "Room Capacity Projector Time"\n'
                '    - row "Cedar 8 Yes 10:00"\n')
FORM = 'URL: http://localhost/booking\n- textbox "Meeting title": "Release review"\n'
REFS = '--- Interactive Elements (1/1) ---\n[e' + 'a' * 32 + '_0] button "Save"'


def observation(identity, text, name="browser_observe"):
    return [{"role": "assistant", "tool_calls": [
        {"id": identity, "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": identity, "content": text}]


def test_previous_view_keeps_facts_but_not_refs_and_does_not_mutate_saved_messages():
    messages = observation("a", AVAILABILITY + REFS) + observation("b", FORM + REFS, "browser_act")
    original = deepcopy(messages)
    compacted = compact_browser_history(messages)
    assert 'Cedar 8 Yes 10:00' in compacted[1]["content"]
    assert "Earlier browser observation" in compacted[1]["content"]
    assert "a" * 32 not in compacted[1]["content"]
    assert compacted[-1] == messages[-1]
    assert messages == original
    assert compact_browser_history(compacted) == compacted


def test_batch_extraction_after_element_list_is_retained():
    messages = observation("a", "Step 1 (act): " + FORM + REFS +
                           '\nStep 2 (extract): Cedar\t8\tYes\t10:00', "browser_batch")
    messages += observation("b", FORM + REFS)
    assert "Cedar\t8\tYes\t10:00" in compact_browser_history(messages)[1]["content"]


def test_nonbrowser_and_multimodal_outputs_are_untouched():
    messages = observation("a", AVAILABILITY + REFS, "file_read")
    messages += observation("b", [{"type": "text", "text": FORM + REFS}])
    messages += observation("c", FORM + REFS)
    assert compact_browser_history(messages) == messages


def test_page_without_controls_expires_prior_refs():
    messages = observation("a", FORM + REFS) + observation("b", "URL: http://localhost/receipt\nSaved.")
    compacted = compact_browser_history(messages)
    assert "a" * 32 not in compacted[1]["content"]
    assert compacted[-1] == messages[-1]


def test_action_refs_expire_without_changing_page_values_that_resemble_refs():
    messages = observation("a", 'Action dispatched: click on e0\nURL: http://localhost/\n'
                           '- row "Product e123"\n' + REFS, "browser_act")
    messages += observation("b", FORM + REFS)
    historical = compact_browser_history(messages)[1]["content"]
    assert 'Product e123' in historical
    assert 'click on e0' not in historical


def test_history_stays_bounded_across_many_rounds_and_discloses_omissions():
    messages = observation("a", AVAILABILITY + REFS)
    for index in range(30):
        messages += observation(str(index), f'URL: http://localhost/{index}\n' + "x" * 9_000 + "\n" + REFS)
        messages = compact_browser_history(messages)
    retained = [m["content"] for m in messages[:-1]
                if m.get("role") == "tool" and m["content"].startswith("[Earlier browser observation;")]
    assert len(retained) <= MAX_HISTORY_SNAPSHOTS
    assert sum(map(len, retained)) <= MAX_HISTORY_CHARS
    assert all(len(text) <= MAX_SNAPSHOT_CHARS for text in retained)
    assert all("truncated" in text for text in retained)
    assert any("omitted from context" in m.get("content", "") for m in messages)
    assert messages[-1]["content"].endswith(REFS)


def test_duplicate_snapshots_do_not_displace_distinct_views():
    messages = observation("a", AVAILABILITY + REFS)
    for index in range(15):
        messages += observation(str(index), FORM + REFS)
        messages = compact_browser_history(messages)
    assert 'Cedar 8 Yes 10:00' in messages[1]["content"]
