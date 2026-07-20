"""Tests for contrastive winner-vs-loser rule distillation.

Covers rule_learner.learn_from_contrast (extraction, decline, cap, lifecycle
source) and the best_of wiring (_learn_from_contrast reads artifacts and
degrades gracefully).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from rune.memory import rule_learner
from rune.memory.rule_learner import learn_from_contrast

WINNER = "def round_units(x):\n    from decimal import Decimal, ROUND_HALF_UP\n    return int(Decimal(str(x)).quantize(0, rounding=ROUND_HALF_UP))\n"
LOSER = ("def round_units(x):\n    return round(x)\n", "assert 2 == 3")
RULE = (
    "half_rounding: round .5 away from zero; round() is banker's "
    "(round(2.5)==2, wrong); use Decimal ROUND_HALF_UP (2.5 -> 3)."
)


def _llm_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))


class TestLearnFromContrast:
    @pytest.mark.asyncio
    async def test_extracts_and_saves_rule(self):
        client = AsyncMock()
        client.completion = AsyncMock(return_value=_llm_response(RULE))
        with patch("rune.llm.client.get_llm_client", return_value=client):
            key = await learn_from_contrast(WINNER, [LOSER])
        assert key == "half_rounding"
        # Saved with the contrastive source so demotion/decay see it.
        meta = rule_learner.load_fact_meta()
        entry = next(v for v in meta.values() if isinstance(v, dict)
                     and v.get("human_key") == "half_rounding")
        assert entry["source"] == "contrastive_distill"
        assert entry["confidence"] == rule_learner._CRISP_INITIAL_CONFIDENCE

    @pytest.mark.asyncio
    async def test_uses_best_tier(self):
        client = AsyncMock()
        client.completion = AsyncMock(return_value=_llm_response(RULE))
        with patch("rune.llm.client.get_llm_client", return_value=client):
            await learn_from_contrast(WINNER, [LOSER])
        from rune.types import ModelTier

        assert client.completion.call_args.kwargs["tier"] == ModelTier.BEST

    @pytest.mark.asyncio
    async def test_none_reply_declines(self):
        client = AsyncMock()
        client.completion = AsyncMock(return_value=_llm_response("NONE"))
        with patch("rune.llm.client.get_llm_client", return_value=client):
            assert await learn_from_contrast(WINNER, [LOSER]) is None

    @pytest.mark.asyncio
    async def test_empty_inputs_decline_without_llm(self):
        client = AsyncMock()
        with patch("rune.llm.client.get_llm_client", return_value=client):
            assert await learn_from_contrast("", [LOSER]) is None
            assert await learn_from_contrast(WINNER, []) is None
        client.completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self):
        client = AsyncMock()
        client.completion = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("rune.llm.client.get_llm_client", return_value=client):
            assert await learn_from_contrast(WINNER, [LOSER]) is None

    @pytest.mark.asyncio
    async def test_rule_prefix_stripped(self):
        client = AsyncMock()
        client.completion = AsyncMock(return_value=_llm_response(f"Rule: {RULE}"))
        with patch("rune.llm.client.get_llm_client", return_value=client):
            assert await learn_from_contrast(WINNER, [LOSER]) == "half_rounding"

    @pytest.mark.asyncio
    async def test_contrast_rules_are_demotable(self):
        """A learnable source that demotion ignores could never be voted out."""
        client = AsyncMock()
        client.completion = AsyncMock(return_value=_llm_response(RULE))
        with patch("rune.llm.client.get_llm_client", return_value=client):
            await learn_from_contrast(WINNER, [LOSER])
        before = next(v["confidence"] for v in rule_learner.load_fact_meta().values()
                      if isinstance(v, dict) and v.get("human_key") == "half_rounding")
        rule_learner.update_rules_from_outcome(
            "code_modify",
            task_success=False,
            goal="rounding task about half values away from zero banker's Decimal",
        )
        after = next(v["confidence"] for v in rule_learner.load_fact_meta().values()
                     if isinstance(v, dict) and v.get("human_key") == "half_rounding")
        assert after < before


class TestBestOfWiring:
    @pytest.mark.asyncio
    async def test_reads_artifacts_and_calls_learner(self, tmp_path):
        from rune.cli.best_of import AttemptArtifact, _learn_from_contrast

        wdir = tmp_path / "w"
        ldir = tmp_path / "l"
        wdir.mkdir()
        ldir.mkdir()
        (wdir / "solution.py").write_text("winner code")
        (ldir / "solution.py").write_text("loser code")
        winner = AttemptArtifact(index=0, workdir=str(wdir), stdout="",
                                 returncode=0, produced=["solution.py"])
        loser = AttemptArtifact(index=1, workdir=str(ldir), stdout="",
                                returncode=0, produced=["solution.py"])
        with patch("rune.memory.rule_learner.learn_from_contrast",
                   new=AsyncMock(return_value="k")) as m:
            key = await _learn_from_contrast(
                winner, [loser], {str(ldir): "assert 2 == 3"}
            )
        assert key == "k"
        args = m.call_args.args
        assert args[0] == "winner code"
        assert args[1] == [("loser code", "assert 2 == 3")]

    @pytest.mark.asyncio
    async def test_no_loser_code_declines(self, tmp_path):
        from rune.cli.best_of import AttemptArtifact, _learn_from_contrast

        wdir = tmp_path / "w"
        wdir.mkdir()
        (wdir / "solution.py").write_text("winner code")
        winner = AttemptArtifact(index=0, workdir=str(wdir), stdout="",
                                 returncode=0, produced=["solution.py"])
        empty = AttemptArtifact(index=1, workdir=str(tmp_path / "missing"),
                                stdout="", returncode=0, produced=["solution.py"])
        with patch("rune.memory.rule_learner.learn_from_contrast",
                   new=AsyncMock()) as m:
            assert await _learn_from_contrast(winner, [empty], {}) is None
        m.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_learner_exception_is_swallowed(self, tmp_path):
        from rune.cli.best_of import AttemptArtifact, _learn_from_contrast

        wdir = tmp_path / "w"
        ldir = tmp_path / "l"
        wdir.mkdir()
        ldir.mkdir()
        (wdir / "s.py").write_text("w")
        (ldir / "s.py").write_text("l")
        winner = AttemptArtifact(index=0, workdir=str(wdir), stdout="", returncode=0, produced=["s.py"])
        loser = AttemptArtifact(index=1, workdir=str(ldir), stdout="", returncode=0, produced=["s.py"])
        with patch("rune.memory.rule_learner.learn_from_contrast",
                   new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert await _learn_from_contrast(winner, [loser], {}) is None
