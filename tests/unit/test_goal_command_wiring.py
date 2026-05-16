"""Tests for the /goal command + schema wiring (design Phase 4)."""

from __future__ import annotations

from rune.config.schema import GoalLoopConfig, RuneConfig
from rune.ui.commands import _ALIAS_MAP, COMMANDS, _goal_handler


async def test_goal_handler_usage_when_empty() -> None:
    out = await _goal_handler("   ")
    assert out is not None and not out.startswith("__ACTION__")
    assert "/goal" in out


async def test_goal_handler_emits_action() -> None:
    out = await _goal_handler("ship the parser, pytest must pass")
    assert out == "__ACTION__:goal_loop:ship the parser, pytest must pass"


def test_goal_command_registered_with_alias() -> None:
    assert "/goal" in COMMANDS
    cmd = COMMANDS["/goal"]
    assert cmd.handler is _goal_handler
    assert "/g" in cmd.aliases
    assert _ALIAS_MAP.get("/g") == "/goal"


def test_goal_loop_config_defaults_and_wired_into_root() -> None:
    gc = GoalLoopConfig()
    assert gc.enabled is True
    assert gc.max_iterations == 10
    assert gc.stagnation_window == 3
    assert 0.0 < gc.evidence_threshold <= 1.0
    assert gc.adversarial_review is True  # hardened by default (Phase 5)
    assert gc.ssc_interval == 0  # opt-in (per-iteration cost)

    root = RuneConfig()
    assert isinstance(root.goal_loop, GoalLoopConfig)
    assert root.goal_loop.max_total_tokens == 2_000_000
