"""Cap→upshift handoff: facts cross the retry boundary, trajectories don't."""

from rune.agent.loop import _cap_handoff_facts


def _call(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_handoff_pairs_calls_with_results_in_order():
    r = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "tool_calls": [
            _call("web_search", '{"query": "megabox songdo"}'),
            _call("web_fetch", '{"url": "https://megabox.co.kr"}'),
        ]},
        {"role": "tool", "content": "Top results: ..."},
        {"role": "tool", "content": "Error: dynamic page, empty shell"},
    ]
    out = _cap_handoff_facts(r)
    lines = out.split("\n")
    assert lines[0].startswith('- web_search({"query": "megabox songdo"}) → ok: Top results')
    assert lines[1].startswith('- web_fetch({"url": "https://megabox.co.kr"}) → FAILED: Error:')


def test_handoff_marks_calls_cut_off_without_results():
    r = [
        {"role": "assistant", "tool_calls": [_call("browser_navigate", "{}")]},
    ]
    out = _cap_handoff_facts(r)
    assert out == "- browser_navigate({}) → (cut off before result)"


def test_handoff_bounds_size():
    msgs = []
    for i in range(40):
        msgs.append({"role": "assistant", "tool_calls": [_call("web_search", f'{{"q": "{i}"}}' + "x" * 300)]})
        msgs.append({"role": "tool", "content": "r" * 500})
    out = _cap_handoff_facts(msgs)
    assert len(out) <= 2000
    assert len(out.split("\n")) <= 15


def test_handoff_never_includes_raw_trajectory_text():
    # Assistant narration (the trajectory) must not cross the boundary.
    r = [
        {"role": "assistant", "content": "I will now try the official site...",
         "tool_calls": [_call("web_fetch", "{}")]},
        {"role": "tool", "content": "shell html"},
        {"role": "assistant", "content": "Apologies, I could not retrieve it."},
    ]
    out = _cap_handoff_facts(r)
    assert "I will now" not in out
    assert "Apologies" not in out


def test_handoff_empty_transcript():
    assert _cap_handoff_facts([]) == ""
    assert _cap_handoff_facts([None, "junk", 42]) == ""
