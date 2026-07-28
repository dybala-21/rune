"""Fuzzy edit-application ladder (edit_matching) + file_edit wiring.

Near-miss search strings are the most common weak-model edit failure;
exact-match-only file_edit bounces them into retry spirals. The ladder
recovers unique trimmed/whitespace-normalized matches, hints the closest
real section otherwise, and escalates after repeated failures on one file.
"""

from __future__ import annotations

import pytest

from rune.capabilities.edit_matching import (
    apply_block,
    closest_section_hint,
    escalation_hint,
    find_block,
    record_edit_failure,
    record_edit_success,
)

CONTENT = (
    "class Printer:\n"
    "    def render(self, x):\n"
    "        value = compute(x)\n"
    "        return str(value)\n"
    "\n"
    "def helper():\n"
    "    return 1\n"
)


def test_trimmed_match_recovers_indent_drift():
    # Model copied the block with wrong leading indentation.
    search = "def render(self, x):\n    value = compute(x)\n    return str(value)"
    m = find_block(CONTENT, search)
    assert m is not None and m.strategy == "trimmed"
    out = apply_block(CONTENT, m, "def render(self, x):\n    return repr(x)")
    # Re-indented to the file's 4-space class-body level.
    assert "    def render(self, x):\n        return repr(x)\n" in out
    assert "compute" not in out


def test_ws_normalized_match():
    # A single line with extra internal spaces still matches uniquely.
    m = find_block(CONTENT, "        value =    compute(x)")
    assert m is not None
    assert m.strategy in ("trimmed", "ws-normalized")


def test_ambiguous_match_refuses(tmp_path):
    content = "a = 1\nb = 2\na = 1\n"
    assert find_block(content, "a = 1") is None  # two candidates → refuse


def test_no_match_gives_closest_hint():
    hint = closest_section_hint(CONTENT, "value = compute_v2(x)\nreturn str(value)")
    assert "Closest matching section" in hint
    assert "compute(x)" in hint


def test_failure_counter_and_escalation():
    record_edit_success("/tmp/x.py")  # reset
    assert record_edit_failure("/tmp/x.py") == 1
    assert escalation_hint("/tmp/x.py", 1) == ""
    assert record_edit_failure("/tmp/x.py") == 2
    msg = escalation_hint("/tmp/x.py", 2)
    assert "file_write" in msg and "re-read" in msg
    record_edit_success("/tmp/x.py")
    assert record_edit_failure("/tmp/x.py") == 1  # reset worked


def _allow_guardian(monkeypatch):
    """pytest tmp dirs live under /var, which Guardian protects — stub it."""
    from types import SimpleNamespace

    import rune.capabilities.file as file_cap

    monkeypatch.setattr(
        file_cap, "get_guardian",
        lambda: SimpleNamespace(
            validate_file_path=lambda p: SimpleNamespace(
                allowed=True, reason=""
            )
        ),
    )


@pytest.mark.asyncio
async def test_file_edit_fuzzy_end_to_end(tmp_path, monkeypatch):
    from rune.capabilities.file import FileEditParams, file_edit

    _allow_guardian(monkeypatch)
    target = tmp_path / "proj" / "mod.py"
    target.parent.mkdir()
    target.write_text(CONTENT)

    # Indent-drifted search block (would fail exact match).
    res = await file_edit(FileEditParams(
        path=str(target),
        search="def render(self, x):\n    value = compute(x)\n    return str(value)",
        replace="def render(self, x):\n    return repr(x)",
    ))
    assert res.success, res.error
    assert "fuzzy match" in res.output
    assert "    def render(self, x):\n        return repr(x)" in target.read_text()


@pytest.mark.asyncio
async def test_file_edit_no_match_hints_and_escalates(tmp_path, monkeypatch):
    from rune.capabilities.edit_matching import record_edit_success
    from rune.capabilities.file import FileEditParams, file_edit

    _allow_guardian(monkeypatch)
    target = tmp_path / "proj" / "mod.py"
    target.parent.mkdir()
    target.write_text(CONTENT)
    record_edit_success(str(target.resolve()))

    p = FileEditParams(path=str(target), search="totally unrelated text",
                       replace="x")
    r1 = await file_edit(p)
    assert not r1.success
    r2 = await file_edit(p)
    assert not r2.success
    assert "consecutive failure #2" in (r2.error or "")
    assert "file_write" in (r2.error or "")
