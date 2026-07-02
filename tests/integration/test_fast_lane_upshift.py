"""Fast-lane in-run upshift.

Failover only reacts to exceptions, so gate pressure is the lane's only
escape hatch. A scripted fake LiteLLMAgent forces that path
deterministically. See docs/design/simple-query-fast-path.md.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.config.loader import get_config
from rune.types import AgentConfig

PRIMARY = "anthropic/claude-opus-4-6"


def _fast_model() -> str:
    # Derived from config, not hardcoded — CI configs may override tiers.
    return f"anthropic/{get_config().llm.models.anthropic.fast}"


class _FakeStream:
    """Minimal stand-in for the LiteLLM stream result the loop consumes."""

    def __init__(self, text: str) -> None:
        self._text = text

    def inject_failure_state(self, streak: Any, seen: Any) -> None:
        pass

    async def stream_text(self, delta: bool = True):
        if self._text:
            yield self._text

    def get_failure_state(self) -> tuple[dict, set]:
        return {}, set()

    def usage(self) -> Any:
        return SimpleNamespace(input_tokens=100, output_tokens=50)

    def all_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
        if self._text:
            msgs.append({"role": "assistant", "content": self._text})
        return msgs


class _FakeAgentFactory:
    """Records every LiteLLMAgent construction and scripts step outputs."""

    def __init__(self, step_texts: list[str]) -> None:
        self.constructions: list[dict[str, Any]] = []
        self._step_texts = step_texts
        self._step = 0

    def __call__(self, model: str, **kwargs: Any) -> Any:
        self.constructions.append({"model": model, **kwargs})
        factory = self

        class _FakeAgent:
            @asynccontextmanager
            async def run_stream(self, goal: str, **kw: Any):
                text = (
                    factory._step_texts[factory._step]
                    if factory._step < len(factory._step_texts)
                    else factory._step_texts[-1]
                )
                factory._step += 1
                yield _FakeStream(text)

        return _FakeAgent()


@pytest.fixture()
def anthropic_lane(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "active_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "active_model", "claude-opus-4-6")
    monkeypatch.setattr(cfg.llm, "route_simple_queries", True)
    monkeypatch.setattr(cfg.llm, "simple_query_confidence", 0.8)
    return cfg


def _web_classification() -> ClassificationResult:
    return ClassificationResult(
        goal_type="web", confidence=0.95, tier=2, reason="test",
    )


def _loop() -> NativeAgentLoop:
    loop = NativeAgentLoop(
        config=AgentConfig(max_iterations=6, timeout_seconds=30, model="test-model")
    )
    loop._token_budget.total = 300_000
    return loop


ANSWER = "샌디스크 주가는 오늘 11% 하락한 45.2달러에 거래를 마쳤습니다. " * 3


@pytest.mark.asyncio
async def test_gate_pressure_upshifts_to_primary(anthropic_lane) -> None:
    """Two empty steps block the gate twice -> primary model restored."""
    # Steps 1-2: no text, no tools -> full gate blocks (intent unresolved,
    # web evidence missing). Step 3: a real answer -> completes.
    factory = _FakeAgentFactory(step_texts=["", "", ANSWER])
    loop = _loop()

    with patch("rune.agent.loop.LiteLLMAgent", factory):
        trace = await loop._execute_loop(
            goal="오늘 샌디스크 주가",
            system_prompt="test",
            tools=["web_search", "web_fetch", "think"],
            max_iterations=6,
            classification=_web_classification(),
        )

    assert trace.reason == "completed"
    # First construction: the lane model with the tight round cap.
    assert factory.constructions[0]["model"] == _fast_model()
    assert factory.constructions[0]["max_tool_rounds"] == 3
    # Upshift rebuilt the agent on the primary model with normal rounds.
    assert len(factory.constructions) == 2
    assert factory.constructions[1]["model"] == PRIMARY
    assert factory.constructions[1]["max_tool_rounds"] >= 12
    # The restored model starts with a fresh block budget.
    assert loop._gate_blocked_count == 0


@pytest.mark.asyncio
async def test_clean_run_never_upshifts(anthropic_lane) -> None:
    """An immediate good answer stays on the lane model end to end."""
    factory = _FakeAgentFactory(step_texts=[ANSWER])
    loop = _loop()

    with patch("rune.agent.loop.LiteLLMAgent", factory):
        trace = await loop._execute_loop(
            goal="오늘 샌디스크 주가",
            system_prompt="test",
            tools=["web_search", "web_fetch", "think"],
            max_iterations=6,
            classification=_web_classification(),
        )

    assert trace.reason == "completed"
    assert len(factory.constructions) == 1
    assert factory.constructions[0]["model"] == _fast_model()


@pytest.mark.asyncio
async def test_light_primary_upshift_still_lifts_round_cap(anthropic_lane, monkeypatch) -> None:
    """On a light-primary session the upshift can't change the model,
    but it must still lift the 3-round cap."""
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "active_model", cfg.llm.models.anthropic.fast)
    factory = _FakeAgentFactory(step_texts=["", "", ANSWER])
    loop = _loop()

    with patch("rune.agent.loop.LiteLLMAgent", factory):
        trace = await loop._execute_loop(
            goal="오늘 샌디스크 주가",
            system_prompt="test",
            tools=["web_search", "web_fetch", "think"],
            max_iterations=6,
            classification=_web_classification(),
        )

    assert trace.reason == "completed"
    assert factory.constructions[0]["model"] == _fast_model()
    assert factory.constructions[0]["max_tool_rounds"] == 3
    # Same model both times — but the rebuild must restore the round budget.
    assert len(factory.constructions) == 2
    assert factory.constructions[1]["model"] == _fast_model()
    assert factory.constructions[1]["max_tool_rounds"] >= 12


class _SearchingAgentFactory(_FakeAgentFactory):
    """Like _FakeAgentFactory, but performs one real web_search tool call
    (through the adapter, so evidence counters update) before each step's
    text. Requires the capability registry to be patched."""

    def __call__(self, model: str, **kwargs: Any) -> Any:
        self.constructions.append({"model": model, **kwargs})
        factory = self
        tools = kwargs.get("tools") or []
        search = next(
            (
                t.function
                for t in tools
                if getattr(t, "name", getattr(t, "__name__", "")) == "web_search"
            ),
            None,
        )
        assert search is not None, "web_search tool not built"

        class _FakeAgent:
            @asynccontextmanager
            async def run_stream(self, goal: str, **kw: Any):
                if factory._step == 0:
                    await search(query="샌디스크 주가")
                text = (
                    factory._step_texts[factory._step]
                    if factory._step < len(factory._step_texts)
                    else factory._step_texts[-1]
                )
                factory._step += 1
                yield _FakeStream(text)

        return _FakeAgent()


async def _fake_execute(cap_name: str, params: dict[str, Any]) -> Any:
    from rune.types import CapabilityResult

    return CapabilityResult(
        success=True, output="SNDK $45.20 (-11.17%) — finance snippet"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lane_on", [True, False])
async def test_search_only_answer_passes_gate_only_on_the_lane(
    anthropic_lane, monkeypatch, lane_on
) -> None:
    """The lane's grounding relaxation, exercised behaviorally: one search,
    zero fetches, short answer. Lane on -> gate passes; lane off -> the
    web-evidence gate demands a fetch and blocks."""
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "route_simple_queries", lane_on)
    # Short answer (< 50 chars) so the full completion gate evaluates
    # instead of the text-plus-evidence fast path.
    factory = _SearchingAgentFactory(step_texts=["45.20달러입니다."])
    loop = _loop()

    from rune.capabilities.registry import get_capability_registry

    monkeypatch.setattr(get_capability_registry(), "execute", _fake_execute)
    with patch("rune.agent.loop.LiteLLMAgent", factory):
        trace = await loop._execute_loop(
            goal="오늘 샌디스크 주가",
            system_prompt="test",
            tools=["web_search", "web_fetch", "think"],
            max_iterations=4,
            classification=_web_classification(),
        )

    if lane_on:
        assert trace.reason == "completed"
        assert trace.final_step == 1
    else:
        # Without the lane the same run must not sail through at step 1.
        assert not (trace.reason == "completed" and trace.final_step == 1)


@pytest.mark.asyncio
async def test_non_simple_goal_runs_primary_from_the_start(anthropic_lane) -> None:
    """research goals must never see the lane model at all."""
    factory = _FakeAgentFactory(step_texts=[ANSWER])
    loop = _loop()

    with patch("rune.agent.loop.LiteLLMAgent", factory):
        trace = await loop._execute_loop(
            goal="이 리포의 아키텍처를 분석해줘",
            system_prompt="test",
            tools=["file_read", "think"],
            max_iterations=6,
            classification=ClassificationResult(
                goal_type="research", confidence=0.99, tier=2, reason="test",
            ),
        )

    assert trace.reason == "completed"
    assert all(c["model"] == PRIMARY for c in factory.constructions)
