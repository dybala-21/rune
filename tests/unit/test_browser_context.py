"""Changing views must not erase the data needed to complete a browser task."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.browser_context import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_SNAPSHOTS,
    MAX_SNAPSHOT_CHARS,
    compact_browser_history,
    observe_followup,
)

AVAILABILITY = ('URL: http://localhost/booking\n- document:\n'
                '  - table:\n    - row "Room Capacity Projector Time"\n'
                '    - row "Cedar 8 Yes 10:00"\n')
FORM = 'URL: http://localhost/booking\n- textbox "Meeting title": "Release review"\n'
REFS = '--- Interactive Elements (1/1) ---\n[e' + 'a' * 32 + '_0] button "Save"'


@pytest.mark.parametrize("kind,continuation,domain_change,status,available,observe", [
    ("browser", True, False, "open", True, True),
    ("browser", False, False, "open", True, False),
    ("browser", True, True, "open", True, False),
    ("browser", True, False, "closed", True, False),
    ("browser", True, False, "unknown", True, False),
    ("browser", True, False, "open", False, False),
    ("chat", True, False, "open", True, False),
    ("research", True, False, "open", True, False),
    ("full", True, False, "open", True, False),
])
async def test_only_related_browser_interactions_receive_a_fresh_view(kind, continuation, domain_change, status, available, observe):
    from rune.agent.goal_classifier import ClassificationResult

    classification = ClassificationResult(kind, .99, 2, is_continuation=continuation,
                                          is_domain_change=domain_change, available=available)
    tool = SimpleNamespace(function=AsyncMock(return_value=FORM + REFS))
    history = [{"role": "user", "content": "Earlier task"}]
    original = deepcopy(history)
    seeded = await observe_followup(history, "Change the quantity", classification, {"status": status}, tool)
    assert history == original
    assert tool.function.await_count == int(observe)
    if observe:
        assert seeded[-3] == {"role": "user", "content": "Change the quantity"}
        call = seeded[-2]["tool_calls"][0]
        assert call["function"] == {"name": "browser_observe", "arguments": "{}"}
        assert seeded[-1] == {"role": "tool", "tool_call_id": call["id"], "content": FORM + REFS}
    else:
        assert seeded is history


async def test_followup_keeps_attachments_and_explicit_observation_failure():
    from rune.agent.goal_classifier import ClassificationResult

    goal = "Change the quantity"
    content = [{"type": "text", "text": goal}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    history = [{"role": "user", "content": content}]
    classification = ClassificationResult("browser", .99, 2, is_continuation=True)
    tool = SimpleNamespace(function=AsyncMock(return_value="[BLOCKED] Browser access denied"))
    seeded = await observe_followup(history, goal, classification, {"status": "open"}, tool)
    assert len(seeded) == 3 and seeded[0]["content"] == content
    assert seeded[-1]["content"] == "[BLOCKED] Browser access denied"
    tool.function.assert_awaited_once_with()


async def test_followup_observation_respects_registry_denial_and_reports_it(monkeypatch, tmp_path):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    reg = CapabilityRegistry()
    register_browser_capabilities(reg)
    execute = AsyncMock()
    reg.get("browser_observe").execute = execute
    started, ended = AsyncMock(), AsyncMock()
    wrappers = build_tool_set(ToolAdapterOptions(allowed_tools=["browser_observe"],
                              on_tool_start=started, on_tool_end=ended), registry=reg)
    reg.set_denied_patterns(["browser_observe"])
    classification = ClassificationResult("browser", .99, 2, is_continuation=True)
    seeded = await observe_followup([], "Change quantity", classification, {"status": "open"}, wrappers["browser_observe"])
    execute.assert_not_called()
    started.assert_awaited_once()
    ended.assert_awaited_once()
    assert not ended.call_args.args[1].success
    assert "denied" in seeded[-1]["content"].lower()


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
