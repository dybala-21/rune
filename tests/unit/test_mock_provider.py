"""Tests for rune.evaluation.providers.mock_provider — recording, playback, passthrough."""

from __future__ import annotations

from pathlib import Path

import pytest

from rune.evaluation.providers.mock_provider import (
    MockProvider,
    MockProviderOptions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages(content: str = "test") -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


async def _real_provider(messages, system_prompt):
    return "response from LLM"


# ---------------------------------------------------------------------------
# Playback mode
# ---------------------------------------------------------------------------


class TestPlaybackMode:
    @pytest.mark.asyncio
    async def test_throws_when_no_recording_exists(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="playback",
        ))
        with pytest.raises(RuntimeError, match="No recording found"):
            await provider.call(_messages(), "system")

    @pytest.mark.asyncio
    async def test_returns_recorded_response(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")

        # Record first
        recorder = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="recording",
        ))

        async def real(msgs, sp):
            return "new response"

        await recorder.call(_messages("test"), "system", real)
        recorder.flush()

        # Playback
        player = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="playback",
        ))
        response = await player.call(_messages("test"), "system")
        assert response == "new response"

    @pytest.mark.asyncio
    async def test_fallback_to_real_provider(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="playback", fallback_to_real=True,
        ))

        async def real(msgs, sp):
            return "real response"

        response = await provider.call(_messages(), "system", real)
        assert response == "real response"


# ---------------------------------------------------------------------------
# Recording mode
# ---------------------------------------------------------------------------


class TestRecordingMode:
    @pytest.mark.asyncio
    async def test_calls_real_provider_and_saves(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="recording",
        ))

        async def real(msgs, sp):
            return "response from LLM"

        response = await provider.call(_messages(), "system", real)
        assert response == "response from LLM"

        provider.flush()
        assert Path(recording_path).exists()

    @pytest.mark.asyncio
    async def test_throws_without_real_provider(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="recording",
        ))
        with pytest.raises(RuntimeError, match="Real provider required"):
            await provider.call(_messages(), "system")


# ---------------------------------------------------------------------------
# Passthrough mode
# ---------------------------------------------------------------------------


class TestPassthroughMode:
    @pytest.mark.asyncio
    async def test_always_calls_real_provider(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="passthrough",
        ))

        async def real(msgs, sp):
            return "live response"

        response = await provider.call(_messages(), "system", real)
        assert response == "live response"

    @pytest.mark.asyncio
    async def test_throws_without_real_provider(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="passthrough",
        ))
        with pytest.raises(RuntimeError, match="Real provider required"):
            await provider.call(_messages(), "system")


# ---------------------------------------------------------------------------
# create_caller
# ---------------------------------------------------------------------------


class TestCreateCaller:
    @pytest.mark.asyncio
    async def test_creates_callable(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="recording",
        ))

        async def real(msgs, sp):
            return "response"

        caller = provider.create_caller(real)
        response = await caller(_messages(), "system")
        assert response == "response"


# ---------------------------------------------------------------------------
# recording_count
# ---------------------------------------------------------------------------


class TestRecordingCount:
    @pytest.mark.asyncio
    async def test_tracks_recording_count(self, tmp_path):
        recording_path = str(tmp_path / "recordings.json")
        provider = MockProvider(MockProviderOptions(
            recording_path=recording_path, mode="recording",
        ))

        async def real(msgs, sp):
            return "response"

        assert provider.recording_count == 0
        await provider.call(_messages("test1"), "system", real)
        await provider.call(_messages("test2"), "system", real)
        assert provider.recording_count == 2
