"""split_answer must show the last narrating step, never a duplicated transcript."""
from rune.api.server import join_steps, split_answer


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


def test_join_steps_separates_each_step_with_a_blank_line():
    # Deltas arrive token by token, so a step's own text concatenates directly,
    # but consecutive steps are separate remarks and must not run together.
    collected = ["Opening ", "the page. ", "Now ", "searching."]
    assert join_steps(collected, [0, 2]) == "Opening the page.\n\nNow searching."


def test_join_steps_drops_steps_that_produced_no_text():
    collected = ["Only step. ", "", ""]
    assert join_steps(collected, [0, 1, 2]) == "Only step."


def test_join_steps_single_step_has_no_separator():
    assert join_steps(["a", "b", "c"], [0]) == "abc"


def test_answer_keeps_paragraphs_when_the_last_step_spans_several():
    # Walking back past textless steps must not glue the recovered steps together.
    collected = ["First. ", "Second."]
    _, answer = split_answer(collected, [0, 1, 2])
    assert answer == "Second."
