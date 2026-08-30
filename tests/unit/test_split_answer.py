"""split_answer must show the last narrating step, never a duplicated transcript."""
from rune.api.server import split_answer


def test_single_pass_returns_all_text():
    full, answer = split_answer(["Hello ", "world"], [0])
    assert full == "Hello world"
    assert answer == "Hello world"


def test_two_narrating_steps_returns_last_only():
    # step A then step B; boundary at index 1.
    full, answer = split_answer(["A", "B"], [0, 1])
    assert answer == "B"
    assert full == "AB"


def test_trailing_textless_step_does_not_dump_full_transcript():
    # The real bug: summary A (+tool), summary B, then a tool-only / re-observe
    # step that adds no text. The old `or full` fallback returned everything and
    # the chat showed the summary twice.
    collected = ["Summary A ", "with table. ", "Summary B ", "final table."]
    step_starts = [0, 2, 4]  # A at 0, B at 2, empty trailing step at 4
    full, answer = split_answer(collected, step_starts)
    assert answer == "Summary B final table."
    assert "Summary A" not in answer
    assert full == "Summary A with table. Summary B final table."


def test_all_empty_returns_empty():
    _, answer = split_answer(["", ""], [0, 1])
    assert answer == ""
