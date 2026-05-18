"""Tests for rune.agent.goal_review - adversarial review + self-critique.

Stub LLM (litellm-shaped dict); no provider is touched.
"""

from __future__ import annotations

import json
from typing import Any

from rune.agent.goal_loop import GoalSpec, ReviewContext
from rune.agent.goal_review import make_adversarial_review_fn, make_ssc_critique_fn


class StubLLM:
    def __init__(self, content: str = "", *, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise = raise_exc
        self.last_messages: list[dict[str, Any]] = []

    async def completion(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.last_messages = messages
        if self._raise is not None:
            raise self._raise
        return {"choices": [{"message": {"content": self._content}}]}


def spec() -> GoalSpec:
    return GoalSpec(goal="do X", acceptance_criteria=["ac1"], constraints=["c1"])


def rc(*, artifact: str = "src", claim: str = "x") -> ReviewContext:
    return ReviewContext(
        spec=spec(),
        claim=claim,
        validation_output="$ go test ./...\n[exit 0]\nok",
        artifact=artifact,
    )


# -- adversarial review ------------------------------------------------------


async def test_review_allow() -> None:
    fn = make_adversarial_review_fn(
        llm=StubLLM(json.dumps({"allow": True, "reason": "genuinely done"}))
    )
    allow, reason = await fn(rc())
    assert allow is True
    assert reason == "genuinely done"


async def test_review_block() -> None:
    # a block must cite verbatim evidence the reviewer was shown.
    hollow = "func TestX(t *testing.T) {}  // no assertions"
    fn = make_adversarial_review_fn(
        llm=StubLLM(json.dumps(
            {"allow": False, "reason": "hard-coded", "quote": hollow}
        ))
    )
    allow, reason = await fn(rc(artifact=hollow))
    assert allow is False
    assert reason == "hard-coded"


async def test_review_bad_json_is_fail_closed() -> None:
    fn = make_adversarial_review_fn(llm=StubLLM("not json"))
    allow, reason = await fn(rc())
    assert allow is False  # never accept on parse failure
    assert "unavailable" in reason


async def test_review_llm_exception_is_fail_closed() -> None:
    fn = make_adversarial_review_fn(llm=StubLLM(raise_exc=RuntimeError("down")))
    allow, _ = await fn(rc())
    assert allow is False


async def test_changed_source_reaches_the_reviewer() -> None:
    # Phase 5.1 core: the reviewer must SEE the source (not just pass/fail),
    # so a hollow passing test can be judged on its content.
    stub = StubLLM(json.dumps({"allow": False, "reason": "empty test"}))
    fn = make_adversarial_review_fn(llm=stub)
    await fn(rc(artifact="func TestX(t *testing.T) {}  // no assertions"))
    sent = stub.last_messages[-1]["content"]
    assert "CHANGED SOURCE" in sent
    assert "func TestX(t *testing.T) {}" in sent  # the hollow source is in-prompt
    assert "untrusted" in sent.lower()  # claim still marked untrusted


# -- SSC self-critique -------------------------------------------------------


async def test_ssc_reports_when_gamed() -> None:
    fn = make_ssc_critique_fn(
        llm=StubLLM(
            json.dumps({"gamed": True, "critique": "tests pinned", "spec_patch": "add prop check"})
        )
    )
    note = await fn(spec(), "x", 2)
    assert "tests pinned" in note
    assert "proposed: add prop check" in note


async def test_ssc_empty_when_not_gamed() -> None:
    fn = make_ssc_critique_fn(
        llm=StubLLM(json.dumps({"gamed": False, "critique": "", "spec_patch": ""}))
    )
    assert await fn(spec(), "x", 2) == ""


async def test_ssc_exception_is_swallowed() -> None:
    fn = make_ssc_critique_fn(llm=StubLLM(raise_exc=RuntimeError("down")))
    assert await fn(spec(), "x", 2) == ""  # advisory only


async def test_manifest_reaches_reviewer() -> None:
    stub = StubLLM(json.dumps({
        "allow": False, "reason": "core file omitted",
        "quote": "server.go (400 lines, 12000 B) [baseline] [omitted: cap]",
    }))
    fn = make_adversarial_review_fn(llm=stub)
    art = (
        "SOURCE MANIFEST - 2 file(s), 1 with content shown below:\n"
        "- server.go (400 lines, 12000 B) [baseline] [omitted: cap]\n"
        "- x_test.go (10 lines, 120 B) [changed] [shown]\n\n=== x_test.go ===\n..."
    )
    allow, _ = await fn(rc(artifact=art))
    sent = stub.last_messages[-1]["content"]
    assert "SOURCE MANIFEST" in sent
    assert "[omitted: cap]" in sent  # reviewer can see what it cannot see
    assert allow is False  # quote present in artifact -> block stands


# -- weak-reviewer confabulation guard --------------------------------------


async def test_confabulated_block_is_overridden() -> None:
    # the Gemini-2.0-flash failure: reviewer invents a violation not present
    # in what it was shown. Deterministic validation already passed, so an
    # uncitable block must NOT veto it.
    fn = make_adversarial_review_fn(llm=StubLLM(json.dumps({
        "allow": False,
        "reason": "binds to fixed port 127.0.0.1:8080 and omits futures-util",
        "quote": "TcpListener::bind(\"127.0.0.1:8080\")",  # never in artifact
    })))
    allow, reason = await fn(rc(artifact='bind("127.0.0.1:0") futures-util'))
    assert allow is True
    assert "unsubstantiated" in reason


async def test_substantiated_block_still_blocks() -> None:
    hollow = "#[tokio::test] async fn t() {} // asserts nothing"
    fn = make_adversarial_review_fn(llm=StubLLM(json.dumps({
        "allow": False, "reason": "test asserts nothing",
        "quote": "async fn t() {} // asserts nothing",
    })))
    allow, reason = await fn(rc(artifact=hollow))
    assert allow is False
    assert reason == "test asserts nothing"


async def test_block_with_too_short_quote_is_unsubstantiated() -> None:
    fn = make_adversarial_review_fn(llm=StubLLM(json.dumps({
        "allow": False, "reason": "bad", "quote": "x"  # < 12 chars
    })))
    allow, reason = await fn(rc(artifact="x marks something"))
    assert allow is True
    assert "unsubstantiated" in reason


async def test_block_quote_match_is_whitespace_case_insensitive() -> None:
    # the model recased and reflowed (newline + extra spaces = one ws run)
    # the copied line; _norm collapses ws runs + lowercases so it matches.
    fn = make_adversarial_review_fn(llm=StubLLM(json.dumps({
        "allow": False, "reason": "always-true assertion",
        "quote": "ASSERT_EQ(expected,\n   expected)",
    })))
    allow, _ = await fn(rc(artifact="...\nassert_eq(expected, expected);\n..."))
    assert allow is False
