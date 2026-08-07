"""Navigate hands back the refs it already extracted, so observe isn't needed."""

from rune.capabilities.browser.helpers import (
    MAX_LISTED_ELEMENTS,
    ElementMeta,
    format_interactive_elements,
)


def _el(ref: str, role: str = "button", name: str = "", breadcrumb: str = "") -> ElementMeta:
    return ElementMeta(ref=ref, role=role, name=name, breadcrumb=breadcrumb)


def test_empty_list_renders_nothing():
    assert format_interactive_elements([]) == ""


def test_renders_ref_role_name_and_breadcrumb():
    out = format_interactive_elements([
        _el("e1", "button", "코엑스", "main > list"),
        _el("e2", "link"),
    ])
    lines = out.strip().split("\n")
    assert lines[0] == "--- Interactive Elements (2/2) ---"
    assert lines[1] == '[e1] button "코엑스" in(main > list)'
    assert lines[2] == "[e2] link", "no empty name or breadcrumb fragments"


def test_caps_the_list_but_reports_the_true_total():
    out = format_interactive_elements([_el(f"e{i}") for i in range(120)])
    lines = out.strip().split("\n")
    assert lines[0] == f"--- Interactive Elements ({MAX_LISTED_ELEMENTS}/120) ---"
    assert len(lines) == MAX_LISTED_ELEMENTS + 1


def test_marker_matches_the_adapter_compression_probe():
    """Stale snapshots are collapsed by matching this exact marker text."""
    out = format_interactive_elements([_el("e1")])
    assert "Interactive Elements" in out


def test_navigate_and_open_return_the_snapshot():
    """Both entry points must inline it — a discarded extraction costs a round."""
    import inspect

    from rune.capabilities.browser import core

    for fn in (core.browser_navigate, core.browser_open):
        body = inspect.getsource(fn)
        assert "elements = await extract_interactive_elements(page)" in body, fn.__name__
        assert "format_interactive_elements(elements)" in body, fn.__name__
