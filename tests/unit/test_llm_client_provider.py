"""LLM provider resolution uses the session choice.

Auxiliary subsystems (classifier, gates, learning) call the client with no
explicit provider. They should follow the session-active provider (set by ``-p``
/ ``/model``) rather than fall back to ``default_provider``, so a user who chose a
local provider does not have calls go to the default cloud provider.
"""

from __future__ import annotations

from rune.config.loader import get_config
from rune.llm.client import LLMClient
from rune.types import ModelTier, Provider


def test_no_provider_follows_active_provider():
    cfg = get_config()
    cfg.llm.default_provider = "openai"
    cfg.llm.active_provider = "anthropic"  # user chose anthropic this session
    client = LLMClient()
    # FAST tier, no explicit provider -> must resolve the ANTHROPIC fast model,
    # not the openai default (gpt-5-mini).
    assert client._effective_provider(None) == Provider.ANTHROPIC
    assert client.resolve_model(ModelTier.FAST) == "claude-haiku-4-5-20251001"


def test_no_active_falls_back_to_default():
    cfg = get_config()
    cfg.llm.default_provider = "openai"
    cfg.llm.active_provider = None
    client = LLMClient()
    assert client._effective_provider(None) == Provider.OPENAI
    assert client.resolve_model(ModelTier.FAST) == "gpt-5-mini"


def test_explicit_provider_overrides_active():
    cfg = get_config()
    cfg.llm.default_provider = "openai"
    cfg.llm.active_provider = "anthropic"
    client = LLMClient()
    # An explicit provider arg always wins over the session-active one.
    assert client._effective_provider(Provider.OPENAI) == Provider.OPENAI
    assert client.resolve_model(ModelTier.FAST, Provider.OPENAI) == "gpt-5-mini"


def test_invalid_active_provider_falls_back():
    cfg = get_config()
    cfg.llm.default_provider = "openai"
    cfg.llm.active_provider = "not-a-real-provider"
    client = LLMClient()
    assert client._effective_provider(None) == Provider.OPENAI  # safe fallback


def test_cli_flags_set_session_active_selection():
    """`rune -p X -m Y` sets both halves of the session selection.

    The failover primary profile (the model the agent loop runs) is built from
    active_provider/active_model; if -m does not land in active_model, the loop
    runs the default provider's best tier instead of the requested model.
    """
    from typer.testing import CliRunner

    from rune.cli.main import app

    cfg = get_config()
    prev = (cfg.llm.active_provider, cfg.llm.active_model)
    try:
        result = CliRunner().invoke(
            app, ["-p", "anthropic", "-m", "claude-haiku-4-5-20251001", "--version"]
        )
        assert result.exit_code == 0
        assert cfg.llm.active_provider == "anthropic"
        assert cfg.llm.active_model == "claude-haiku-4-5-20251001"
    finally:
        cfg.llm.active_provider, cfg.llm.active_model = prev


def test_local_provider_needs_no_api_key(monkeypatch):
    """A fully-local (ollama) session must start WITHOUT any cloud API key.

    Live-found: `rune -p ollama` refused to start ("No API key configured"),
    forcing local-only users to create a cloud key — contradicting the
    local/private design.
    """
    from rune.cli.main import _ensure_llm_key

    cfg = get_config()
    monkeypatch.setattr(cfg, "openai_api_key", None)
    monkeypatch.setattr(cfg, "anthropic_api_key", None)
    monkeypatch.setattr(cfg, "gemini_api_key", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    monkeypatch.setattr(cfg.llm, "active_provider", "ollama")
    assert _ensure_llm_key() is True  # local needs no key

    monkeypatch.setattr(cfg.llm, "active_provider", None)
    monkeypatch.setattr(cfg.llm, "default_provider", "ollama")
    assert _ensure_llm_key() is True  # local default too

    monkeypatch.setattr(cfg.llm, "default_provider", "openai")
    assert _ensure_llm_key() is False  # cloud with no key still blocked
