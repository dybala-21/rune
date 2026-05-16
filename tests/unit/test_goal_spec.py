"""Tests for rune.agent.goal_spec — freeform request -> testable GoalSpec.

Driven by a scripted stub LLM (litellm-shaped dict); no provider is touched.
"""

from __future__ import annotations

import json
from typing import Any

from rune.agent.goal_spec import crystallize_goal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubLLM:
    """Returns a fixed completion. ``content`` is the model's raw text;
    ``raise_exc`` simulates a provider/transport failure."""

    def __init__(self, content: str = "", *, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def completion(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append({"messages": messages, **kwargs})
        if self._raise is not None:
            raise self._raise
        return {"choices": [{"message": {"content": self._content}}]}


def good_payload(**over: Any) -> str:
    base: dict[str, Any] = dict(
        goal="Add a /goal command",
        acceptance_criteria=["pytest passes", "ruff clean"],
        constraints=["no auto-commit"],
        validation_commands=["uv run pytest -q", "uv run ruff check ."],
        ambiguous=False,
        clarifications=[],
        notice="2 criteria derived",
    )
    base.update(over)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_well_formed_spec() -> None:
    llm = StubLLM(good_payload())

    res = await crystallize_goal("add a goal command", llm=llm)

    assert res.ambiguous is False
    assert res.spec.goal == "Add a /goal command"
    assert res.spec.acceptance_criteria == ["pytest passes", "ruff clean"]
    assert res.spec.constraints == ["no auto-commit"]
    assert res.spec.validation_commands == ["uv run pytest -q", "uv run ruff check ."]
    assert llm.calls and llm.calls[0]["max_tokens"] >= 4096  # room for reasoning models


async def test_markdown_fenced_json_is_parsed() -> None:
    llm = StubLLM("```json\n" + good_payload() + "\n```")

    res = await crystallize_goal("x", llm=llm)

    assert res.ambiguous is False
    assert res.spec.acceptance_criteria == ["pytest passes", "ruff clean"]


async def test_explicitly_ambiguous_request() -> None:
    llm = StubLLM(
        good_payload(
            acceptance_criteria=["app builds"],
            ambiguous=True,
            clarifications=["Which platform?", "What is 'done'?"],
        )
    )

    res = await crystallize_goal("make it better", llm=llm)

    assert res.ambiguous is True
    assert res.clarifications == ["Which platform?", "What is 'done'?"]


async def test_no_acceptance_criteria_forces_ambiguous() -> None:
    # Model claims not ambiguous but gave no verifiable criteria — the loop
    # must not be allowed to 'verify' on the inner gate alone.
    llm = StubLLM(good_payload(acceptance_criteria=[], ambiguous=False, clarifications=[]))

    res = await crystallize_goal("do the thing", llm=llm)

    assert res.ambiguous is True
    assert res.spec.acceptance_criteria == []
    assert res.clarifications  # auto-populated


async def test_list_sanitation_drops_non_strings() -> None:
    llm = StubLLM(
        good_payload(
            acceptance_criteria=["  keep me  ", 42, None, "", "second"],
            validation_commands="not a list",
        )
    )

    res = await crystallize_goal("x", llm=llm)

    assert res.spec.acceptance_criteria == ["keep me", "second"]
    assert res.spec.validation_commands == []


async def test_malformed_json_falls_back_conservatively() -> None:
    llm = StubLLM("totally not json {")

    res = await crystallize_goal("build a parser", llm=llm)

    assert res.ambiguous is True
    assert res.spec.goal == "build a parser"  # raw request preserved
    assert res.spec.acceptance_criteria == []
    assert res.clarifications


async def test_empty_response_falls_back() -> None:
    llm = StubLLM("")

    res = await crystallize_goal("build a parser", llm=llm)

    assert res.ambiguous is True
    assert res.notice.startswith("crystallization fell back")


async def test_llm_exception_is_fail_closed() -> None:
    llm = StubLLM(raise_exc=RuntimeError("provider down"))

    res = await crystallize_goal("build a parser", llm=llm)

    assert res.ambiguous is True
    assert res.spec.goal == "build a parser"


async def test_empty_request_short_circuits() -> None:
    llm = StubLLM(good_payload())

    res = await crystallize_goal("   ", llm=llm)

    assert res.ambiguous is True
    assert llm.calls == []  # never called the model


async def test_no_validation_commands_flags_notice() -> None:
    # acceptance present but zero validation commands -> notice must warn that
    # verification relies on adversarial review only (Phase 5.1 #B).
    llm = StubLLM(good_payload(validation_commands=[]))
    res = await crystallize_goal("write a design doc", llm=llm)
    assert res.ambiguous is False
    assert res.spec.validation_commands == []
    assert res.notice.startswith("[no machine validation")
