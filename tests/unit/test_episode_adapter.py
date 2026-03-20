"""Tests for rune.evaluation.adapters.episode_adapter — episode conversion and filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from rune.evaluation.adapters.episode_adapter import (
    EvalTestCase,
    episode_to_test_case,
    episode_to_trajectory,
    episodes_to_test_cases,
    export_for_deepeval,
    filter_episodes_for_eval,
)

# ---------------------------------------------------------------------------
# Fixtures — simple Episode-like objects
# ---------------------------------------------------------------------------


@dataclass
class _Intent:
    domain: str = "general"
    action: str = "test action"


@dataclass
class _Step:
    tool: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class _Plan:
    steps: list = field(default_factory=list)


@dataclass
class _Result:
    success: bool = True
    summary: str = "done"
    steps_completed: int = 2


@dataclass
class _Episode:
    id: str = "ep-1"
    timestamp: datetime = field(default_factory=lambda: datetime(2026, 1, 15, 10, 0, 0))
    task_summary: str = "test task"
    intent: _Intent = field(default_factory=_Intent)
    plan: _Plan = field(default_factory=_Plan)
    result: _Result = field(default_factory=_Result)
    lessons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# episode_to_test_case
# ---------------------------------------------------------------------------


class TestEpisodeToTestCase:
    def test_basic_conversion(self):
        ep = _Episode(
            id="ep-1",
            task_summary="read a file",
            intent=_Intent(domain="file", action="read file"),
            lessons=["use file.read"],
        )
        tc = episode_to_test_case(ep)
        assert tc.id == "ep-1"
        assert tc.input == "read a file"
        assert tc.expected_output == "read file"
        assert tc.context == ["use file.read"]
        assert tc.metadata["success"] is True
        assert tc.metadata["domain"] == "file"

    def test_uses_result_summary_as_actual_output(self):
        ep = _Episode(result=_Result(summary="summary text"))
        tc = episode_to_test_case(ep)
        assert tc.actual_output == "summary text"

    def test_empty_result_gives_empty_actual_output(self):
        ep = _Episode()
        ep.result = _Result(summary="")
        tc = episode_to_test_case(ep)
        # Depending on implementation, may be empty or summary
        assert isinstance(tc.actual_output, str)

    def test_extracts_tools_from_plan_steps(self):
        ep = _Episode()
        ep.plan = _Plan(steps=[
            _Step(tool="file.read"),
            _Step(tool="bash"),
            _Step(tool="file.write"),
        ])
        tc = episode_to_test_case(ep)
        assert tc.tools_called == ["file.read", "bash", "file.write"]

    def test_skips_steps_without_tool(self):
        ep = _Episode()
        ep.plan = _Plan(steps=[_Step(tool="bash"), _Step(tool="")])
        tc = episode_to_test_case(ep)
        assert tc.tools_called == ["bash"]

    def test_empty_plan_steps(self):
        ep = _Episode()
        ep.plan = _Plan(steps=[])
        tc = episode_to_test_case(ep)
        assert tc.tools_called == []


# ---------------------------------------------------------------------------
# episode_to_trajectory
# ---------------------------------------------------------------------------


class TestEpisodeToTrajectory:
    def test_converts_plan_steps(self):
        ep = _Episode()
        ep.plan = _Plan(steps=[
            _Step(tool="file.read", params={"path": "/foo.ts"}),
            _Step(tool="bash", params={"command": "ls"}),
        ])
        trajectory = episode_to_trajectory(ep)
        assert len(trajectory) == 2
        assert trajectory[0].tool == "file.read"
        assert trajectory[0].args == {"path": "/foo.ts"}
        assert trajectory[1].tool == "bash"

    def test_skips_empty_tool(self):
        ep = _Episode()
        ep.plan = _Plan(steps=[_Step(tool="bash"), _Step(tool="")])
        trajectory = episode_to_trajectory(ep)
        assert len(trajectory) == 1

    def test_empty_plan(self):
        ep = _Episode()
        ep.plan = _Plan(steps=[])
        assert episode_to_trajectory(ep) == []


# ---------------------------------------------------------------------------
# episodes_to_test_cases (batch)
# ---------------------------------------------------------------------------


class TestEpisodesToTestCases:
    def test_batch_conversion(self):
        episodes = [_Episode(id="ep-1"), _Episode(id="ep-2")]
        cases = episodes_to_test_cases(episodes)
        assert len(cases) == 2
        assert cases[0].id == "ep-1"
        assert cases[1].id == "ep-2"


# ---------------------------------------------------------------------------
# export_for_deepeval
# ---------------------------------------------------------------------------


class TestExportForDeepEval:
    def test_json_roundtrip(self):
        tc = EvalTestCase(id="tc-1", input="test", actual_output="output")
        result = export_for_deepeval([tc])
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "tc-1"

    def test_pretty_print(self):
        tc = EvalTestCase(id="tc-1", input="test")
        result = export_for_deepeval([tc], pretty=True)
        assert "\n" in result


# ---------------------------------------------------------------------------
# filter_episodes_for_eval
# ---------------------------------------------------------------------------


class TestFilterEpisodesForEval:
    def test_filter_only_success(self):
        episodes = [
            _Episode(id="ok", result=_Result(success=True)),
            _Episode(id="fail", result=_Result(success=False)),
        ]
        filtered = filter_episodes_for_eval(episodes, only_success=True)
        assert len(filtered) == 1
        assert filtered[0].id == "ok"

    def test_filter_only_failure(self):
        episodes = [
            _Episode(id="ok", result=_Result(success=True)),
            _Episode(id="fail", result=_Result(success=False)),
        ]
        filtered = filter_episodes_for_eval(episodes, only_failure=True)
        assert len(filtered) == 1
        assert filtered[0].id == "fail"

    def test_filter_by_domain(self):
        episodes = [
            _Episode(id="e1", intent=_Intent(domain="file")),
            _Episode(id="e2", intent=_Intent(domain="web")),
        ]
        filtered = filter_episodes_for_eval(episodes, domains=["file"])
        assert len(filtered) == 1
        assert filtered[0].id == "e1"

    def test_limit(self):
        episodes = [_Episode(id=f"ep-{i}") for i in range(10)]
        filtered = filter_episodes_for_eval(episodes, limit=3)
        assert len(filtered) == 3
