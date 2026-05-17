"""Tests for rune.agent.obs_cap.mask_stale_tool_messages.

Deterministic; no LLM or agent stack.
"""

from __future__ import annotations

from rune.agent.obs_cap import mask_stale_tool_messages


def tool(tid: str, content, n: int = 0) -> dict:
    return {"role": "tool", "tool_call_id": tid, "content": content or ("x" * n)}


def asst(n: int) -> dict:
    return {"role": "assistant", "content": "a" * n, "tool_calls": [{"id": "c"}]}


def test_below_threshold_returns_input_unchanged() -> None:
    msgs = [{"role": "system", "content": "s"}, tool("1", None, 100)]
    out = mask_stale_tool_messages(msgs, activate_over=80_000)
    assert out is msgs  # identity: zero copy / zero behavior change


def test_few_tool_messages_unchanged_even_when_large() -> None:
    msgs = [tool("1", None, 90_000), tool("2", None, 90_000)]
    out = mask_stale_tool_messages(msgs, keep_last=2, activate_over=80_000)
    assert out is msgs  # <= keep_last tool messages -> nothing to truncate


def test_old_tool_truncated_recent_kept_full() -> None:
    big = 50_000
    msgs = [
        {"role": "system", "content": "sys"},
        asst(10),
        tool("old", None, big),
        asst(10),
        tool("mid", None, big),
        asst(10),
        tool("recent", None, big),
    ]
    out = mask_stale_tool_messages(
        msgs, keep_last=2, trunc=3072, activate_over=80_000
    )

    assert out is not msgs
    assert len(out) == len(msgs)  # nothing dropped
    by_id = {m.get("tool_call_id"): m for m in out if m.get("role") == "tool"}
    # oldest tool truncated, elision marker present, bounded
    assert len(by_id["old"]["content"]) < big
    assert "elided" in by_id["old"]["content"]
    assert "narrower command" in by_id["old"]["content"]  # context note kept
    assert len(by_id["old"]["content"]) <= 3072 + 200
    # last keep_last=2 tool results kept full
    assert len(by_id["mid"]["content"]) == big
    assert len(by_id["recent"]["content"]) == big
    # non-tool + ids + order preserved
    assert [m.get("role") for m in out] == [m.get("role") for m in msgs]
    assert out[0]["content"] == "sys"
    assert all(m["content"] == "a" * 10 for m in out if m["role"] == "assistant")
    assert msgs[2]["content"] == "x" * big  # original list untouched


def test_small_old_tool_not_truncated() -> None:
    msgs = [tool("a", None, 90_000), tool("b", "tiny"), tool("c", None, 90_000)]
    out = mask_stale_tool_messages(
        msgs, keep_last=1, trunc=3072, activate_over=80_000
    )
    b = next(m for m in out if m["tool_call_id"] == "b")
    assert b["content"] == "tiny"  # below trunc -> left as-is


def test_non_str_tool_content_is_safe() -> None:
    msgs = [
        tool("a", None, 90_000),
        {"role": "tool", "tool_call_id": "b", "content": None},
        tool("c", None, 90_000),
    ]
    out = mask_stale_tool_messages(msgs, keep_last=1, activate_over=80_000)
    assert out[1]["content"] is None  # no crash, untouched


def test_keep_last_zero_truncates_all_old() -> None:
    msgs = [tool("a", None, 90_000), tool("b", None, 90_000)]
    out = mask_stale_tool_messages(
        msgs, keep_last=0, trunc=1000, activate_over=80_000
    )
    assert all(len(m["content"]) < 90_000 for m in out)
