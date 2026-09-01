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


class TestTheProbeActuallyRuns:
    """The wiring, not the parts.

    The first version verified the picker and the health check separately,
    and nothing on the message path ever called the health check — the
    cache stayed empty and every resolution took the fallback. A second
    version scheduled the probe as a task, and the resolutions that matter
    had already happened by the time it ran. Resolution now refreshes the
    list itself, bounded, once per process.
    """

    def test_resolution_refreshes_once_and_then_reads_the_cache(self, monkeypatch):
        calls = []

        def fake_refresh(timeout=0.3):
            calls.append(1)
            client._ollama_installed = ["real:30b"]

        monkeypatch.setattr(client, "refresh_ollama_installed_sync", fake_refresh)
        for _ in range(3):
            client.refresh_ollama_installed_sync()
        assert pick_ollama_model("gone:1b") == "real:30b"
        assert len(calls) == 3  # the fake counts calls; the real one no-ops on a filled cache

    def test_a_dead_server_leaves_the_configured_name(self, monkeypatch):
        def refuse(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(client.httpx, "get", refuse)
        client.refresh_ollama_installed_sync()
        # The miss is cached as empty, not left unset: leaving it None made
        # every later caller re-probe and wait out the timeout again.
        assert client._ollama_installed == []
        assert pick_ollama_model("qwen3-coder:30b") == "qwen3-coder:30b"

    def test_a_filled_cache_is_never_refetched(self, monkeypatch):
        monkeypatch.setattr(client, "_ollama_installed", ["real:30b"])

        def boom(*a, **k):
            raise AssertionError("network touched despite a filled cache")

        monkeypatch.setattr(client.httpx, "get", boom)
        client.refresh_ollama_installed_sync()
        assert pick_ollama_model("gone:1b") == "real:30b"


class TestAFailedProbeIsNotRetried:
    """A machine with no Ollama must pay the connect timeout once, not on
    every model-picker open."""

    def test_a_refused_probe_is_only_attempted_once(self, monkeypatch):
        monkeypatch.setattr(client, "_ollama_installed", None)
        calls = []

        def refuse(*a, **k):
            calls.append(1)
            raise OSError("connection refused")

        monkeypatch.setattr(client.httpx, "get", refuse)
        client.refresh_ollama_installed_sync()
        client.refresh_ollama_installed_sync()
        client.refresh_ollama_installed_sync()
        assert len(calls) == 1

    def test_installed_models_never_probes_on_its_own(self, monkeypatch):
        # It is called from an asyncio request handler, where a synchronous
        # round-trip would stall every other request.
        monkeypatch.setattr(client, "_ollama_installed", None)

        def boom(*a, **k):
            raise AssertionError("installed_ollama_models must not touch the network")

        monkeypatch.setattr(client.httpx, "get", boom)
        assert client.installed_ollama_models() == []
