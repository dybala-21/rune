"""Tests for /escalate: explicit one-task cloud escalation.

With a local default model, data leaves the machine only on an explicit
/escalate, the switch is announced first, and the session is restored to the
previous model afterwards, including on failure.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from rune.config import get_config
from rune.config.schema import LLMConfig
from rune.ui.app import RuneApp
from rune.ui.commands import _ALIAS_MAP, COMMANDS, _escalate_handler

# Command wiring


async def test_escalate_handler_emits_action_with_task() -> None:
    out = await _escalate_handler("  fix the flaky test  ")
    assert out == "__ACTION__:escalate:fix the flaky test"


async def test_escalate_handler_empty_means_last_message() -> None:
    # Empty is valid: "escalate what I just asked" is the natural flow.
    assert await _escalate_handler("   ") == "__ACTION__:escalate:"


def test_escalate_registered_with_alias() -> None:
    cmd = COMMANDS["/escalate"]
    assert cmd.handler is _escalate_handler
    assert "/esc" in cmd.aliases
    assert _ALIAS_MAP.get("/esc") == "/escalate"


# Config schema


def test_llm_config_escalation_fields_default_unset() -> None:
    cfg = LLMConfig()
    assert cfg.escalation_provider is None
    assert cfg.escalation_model is None


def test_llm_config_escalation_aliases() -> None:
    cfg = LLMConfig(
        **{"escalationProvider": "anthropic", "escalationModel": "claude-opus-4-8"}
    )
    assert cfg.escalation_provider == "anthropic"
    assert cfg.escalation_model == "claude-opus-4-8"


async def test_action_dispatch_routes_to_do_escalate() -> None:
    app = RuneApp.__new__(RuneApp)
    called: list[str] = []

    async def fake_esc(task: str) -> None:
        called.append(task)

    app._do_escalate = fake_esc
    await app._handle_action_result("__ACTION__:escalate:do it")
    await app._handle_action_result("__ACTION__:escalate:")
    assert called == ["do it", ""]


# _do_escalate behavior


def _make_app(provider: str = "ollama", model: str = "llama3.2") -> RuneApp:
    app = RuneApp.__new__(RuneApp)
    app.console = Console(record=True, force_terminal=False)
    app._provider = provider
    app._model = model
    app._user_message_history = []
    app._agent_controller = None
    return app


def _output(app: RuneApp) -> str:
    return app.console.export_text()


async def test_escalate_without_profile_prints_guidance(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", None)
    app = _make_app()
    ran: list[str] = []

    async def fake_run(text: str) -> None:
        ran.append(text)

    app._run_agent = fake_run
    await app._do_escalate("hard task")
    assert ran == []  # nothing left the machine
    assert "escalationProvider" in _output(app)


async def test_escalate_runs_once_on_profile_and_restores(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "escalation_model", "claude-opus-4-8")
    monkeypatch.setattr(cfg.llm, "active_provider", None)
    monkeypatch.setattr(cfg.llm, "active_model", None)
    app = _make_app(provider="ollama", model="llama3.2")
    seen: list[tuple[str, str, str, str | None]] = []

    async def fake_run(text: str) -> None:
        # Capture the session state during the escalated run.
        seen.append((text, app._provider, app._model, cfg.llm.active_provider))

    app._run_agent = fake_run
    await app._do_escalate("hard task")

    assert seen == [("hard task", "anthropic", "claude-opus-4-8", "anthropic")]
    # Fully restored afterwards — session stays local-default.
    assert (app._provider, app._model) == ("ollama", "llama3.2")
    assert cfg.llm.active_provider is None
    assert cfg.llm.active_model is None
    # Announced before sending, named the destination provider.
    assert "anthropic:claude-opus-4-8" in _output(app)


async def test_escalate_empty_task_uses_last_message(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "escalation_model", "claude-opus-4-8")
    app = _make_app()
    app._user_message_history = ["first", "the hard one"]
    ran: list[str] = []

    async def fake_run(text: str) -> None:
        ran.append(text)

    app._run_agent = fake_run
    await app._do_escalate("")
    assert ran == ["the hard one"]


async def test_escalate_empty_task_no_history_is_noop(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", "anthropic")
    app = _make_app()
    ran: list[str] = []

    async def fake_run(text: str) -> None:
        ran.append(text)

    app._run_agent = fake_run
    await app._do_escalate("   ")
    assert ran == []
    out = _output(app)
    assert "Nothing to escalate" in out
    # "[task]" must survive Rich markup rendering (needs the \\[ escape)
    assert "/escalate [task]" in out


async def test_escalate_model_defaults_to_provider_best_tier(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "escalation_model", None)
    app = _make_app()
    seen: list[str] = []

    async def fake_run(text: str) -> None:
        seen.append(app._model)

    app._run_agent = fake_run
    await app._do_escalate("hard task")
    assert seen == [cfg.llm.models.anthropic.best]


async def test_escalate_restores_even_when_run_fails(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "escalation_model", "claude-opus-4-8")
    monkeypatch.setattr(cfg.llm, "active_provider", "ollama")
    monkeypatch.setattr(cfg.llm, "active_model", "llama3.2")
    app = _make_app(provider="ollama", model="llama3.2")

    async def fake_run(text: str) -> None:
        raise RuntimeError("provider exploded")

    app._run_agent = fake_run
    with pytest.raises(RuntimeError):
        await app._do_escalate("hard task")
    assert (app._provider, app._model) == ("ollama", "llama3.2")
    assert (cfg.llm.active_provider, cfg.llm.active_model) == ("ollama", "llama3.2")


async def test_escalate_same_model_is_plain_run(monkeypatch) -> None:
    cfg = get_config()
    monkeypatch.setattr(cfg.llm, "escalation_provider", "anthropic")
    monkeypatch.setattr(cfg.llm, "escalation_model", "claude-opus-4-8")
    app = _make_app(provider="anthropic", model="claude-opus-4-8")
    ran: list[str] = []

    async def fake_run(text: str) -> None:
        ran.append(text)

    app._run_agent = fake_run
    await app._do_escalate("hard task")
    assert ran == ["hard task"]
    assert "no-op" in _output(app)
