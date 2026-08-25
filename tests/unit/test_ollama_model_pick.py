"""Which local model actually runs, when config and reality disagree.

The shipped ollama defaults sat broken for months: the configured names
predated native tool calling, nobody had them installed, and a fresh
install that picked the local provider got an agent that could not act. A
name written in config goes stale on its own schedule, so the health check
now keeps the server's own list of installed models and resolution prefers
it — the configured name decides only when the server has not answered yet,
where it doubles as the recommendation of what to pull.
"""

from __future__ import annotations

import pytest

import rune.llm.client as client
from rune.llm.client import pick_ollama_model


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(client, "_ollama_installed", None)


class TestBeforeTheServerHasAnswered:
    def test_the_configured_name_stands(self):
        assert pick_ollama_model("qwen3-coder:30b") == "qwen3-coder:30b"


class TestWithTheInstalledListKnown:
    def test_an_installed_configured_model_is_kept(self, monkeypatch):
        monkeypatch.setattr(client, "_ollama_installed",
                            ["other:7b", "qwen3-coder:30b"])
        assert pick_ollama_model("qwen3-coder:30b") == "qwen3-coder:30b"

    def test_a_stale_configured_name_yields_to_what_exists(self, monkeypatch):
        # The months-long failure this exists to end: config names a model
        # nobody has, the machine holds one that works.
        monkeypatch.setattr(client, "_ollama_installed", ["qwen3-coder:30b"])
        assert pick_ollama_model("codellama") == "qwen3-coder:30b"

    def test_the_newest_installed_model_wins(self, monkeypatch):
        # The list arrives newest first; the most recently pulled model is
        # the best guess at the one the user means.
        monkeypatch.setattr(client, "_ollama_installed",
                            ["new:30b", "old:7b"])
        assert pick_ollama_model("gone:1b") == "new:30b"

    def test_an_embedding_model_never_drives_the_loop(self, monkeypatch):
        monkeypatch.setattr(client, "_ollama_installed",
                            ["nomic-embed-text:latest", "qwen3-coder:30b"])
        assert pick_ollama_model("gone:1b") == "qwen3-coder:30b"

    def test_a_server_with_only_embeddings_falls_back(self, monkeypatch):
        monkeypatch.setattr(client, "_ollama_installed",
                            ["nomic-embed-text:latest"])
        assert pick_ollama_model("qwen3-coder:30b") == "qwen3-coder:30b"

    def test_an_empty_server_falls_back(self, monkeypatch):
        monkeypatch.setattr(client, "_ollama_installed", [])
        assert pick_ollama_model("qwen3-coder:30b") == "qwen3-coder:30b"
