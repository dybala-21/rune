"""Tests for _guided_for_model: guided decoding must apply to weak local ollama
models but NOT to cloud-served ('-cloud') models, which it otherwise breaks
(strips native tools -> text-form calls -> 1-step finalize with no work)."""

from __future__ import annotations

from rune.agent import litellm_adapter as la

_OLLAMA = {"api_base": "http://localhost:11434/v1"}


def test_guided_on_for_local_ollama(monkeypatch):
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "1")
    assert la._guided_for_model("qwen2.5-coder:7b", True, _OLLAMA) is True


def test_guided_off_for_cloud_model(monkeypatch):
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "1")
    # The bug: this was True (guided applied to the strong cloud model -> broke it).
    assert la._guided_for_model("qwen3-coder:480b-cloud", True, _OLLAMA) is False
    assert la._guided_for_model("deepseek-v3.1:671b-cloud", True, _OLLAMA) is False


def test_guided_off_when_env_disabled(monkeypatch):
    monkeypatch.delenv("RUNE_GUIDED_TOOLS", raising=False)
    assert la._guided_for_model("qwen2.5-coder:7b", True, _OLLAMA) is False


def test_guided_off_for_non_ollama(monkeypatch):
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "1")
    assert la._guided_for_model("claude-sonnet-4-5", True, {"api_base": "https://api.anthropic.com"}) is False


def test_guided_off_without_tools(monkeypatch):
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "1")
    assert la._guided_for_model("qwen2.5-coder:7b", False, _OLLAMA) is False
