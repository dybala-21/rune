"""Tests for spec-driven verification (deterministic sampling + run + compare)."""

from __future__ import annotations

import pytest

from rune.agent.evidence_spec import (
    VerificationSpec,
    build_disjoint_sample,
    parse_spec,
    run_spec,
)


def test_parse_spec_valid():
    spec = parse_spec(
        '{"input_path":"/app/in.csv","expected_path":"/app/exp.csv",'
        '"artifact_path":"/app/a.vim",'
        '"run_command":"vim -Es {INPUT} -S /app/a.vim","compare":"in_place",'
        '"row_independent":true}'
    )
    assert spec is not None
    assert spec.valid
    assert spec.row_independent is True


def test_parse_spec_no_check():
    assert parse_spec("NO_CHECK") is None
    assert parse_spec("") is None


def test_parse_spec_requires_input_placeholder():
    # run_command without {INPUT} cannot be driven by the code → invalid.
    assert parse_spec(
        '{"input_path":"/i","expected_path":"/e","run_command":"vim -Es /i"}'
    ) is None


def test_parse_spec_strips_fences_and_prose():
    spec = parse_spec(
        'here is the spec:\n```json\n'
        '{"input_path":"/i","expected_path":"/e","run_command":"run {INPUT}"}\n```'
    )
    assert spec is not None and spec.valid


def test_build_disjoint_sample_splices_first_mid_last():
    inp = [b"i%d\n" % i for i in range(1000)]
    exp = [b"e%d\n" % i for i in range(1000)]
    out = build_disjoint_sample(inp, exp, rows_per_slice=10)
    assert out is not None
    in_s, exp_s = out
    # 3 slices x 10 rows = 30 lines each
    assert in_s.count(b"\n") == 30 and exp_s.count(b"\n") == 30
    # must include a first, a middle, and a last row (disjoint coverage)
    assert b"i0\n" in in_s and b"i999\n" in in_s
    assert any(b"i%d\n" % m in in_s for m in range(490, 510))


def test_build_disjoint_sample_mismatched_counts_returns_none():
    assert build_disjoint_sample([b"a\n"], [b"a\n", b"b\n"], 10) is None


@pytest.mark.asyncio
async def test_run_spec_pass_in_place(tmp_path):
    # artifact emulated with `tr a-z A-Z` applied to the copy in place
    inp = tmp_path / "in"
    inp.write_bytes(b"abc\ndef\n")
    exp = tmp_path / "exp"
    exp.write_bytes(b"ABC\nDEF\n")
    spec = VerificationSpec(
        input_path=str(inp), expected_path=str(exp), artifact_path=None,
        run_command="tr a-z A-Z < {INPUT} > {INPUT}.o && mv {INPUT}.o {INPUT}",
        compare="in_place", row_independent=True,
    )
    state, _ = await run_spec(spec)
    assert state == "pass"


@pytest.mark.asyncio
async def test_run_spec_fail_reports_mismatch(tmp_path):
    inp = tmp_path / "in"
    inp.write_bytes(b"abc\n")
    exp = tmp_path / "exp"
    exp.write_bytes(b"XYZ\n")
    spec = VerificationSpec(
        input_path=str(inp), expected_path=str(exp), artifact_path=None,
        run_command="tr a-z A-Z < {INPUT} > {INPUT}.o && mv {INPUT}.o {INPUT}",
        compare="in_place", row_independent=True,
    )
    state, evidence = await run_spec(spec)
    assert state == "fail"
    assert "mismatch" in evidence


@pytest.mark.asyncio
async def test_run_spec_timeout_is_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_EVIDENCE_GATE_TIMEOUT_MS", "1000")
    inp = tmp_path / "in"
    inp.write_bytes(b"x\n")
    exp = tmp_path / "exp"
    exp.write_bytes(b"x\n")
    spec = VerificationSpec(
        input_path=str(inp), expected_path=str(exp), artifact_path=None,
        run_command="sleep 5; cat {INPUT} > /dev/null",
        compare="in_place", row_independent=True,
    )
    state, _ = await run_spec(spec)
    assert state == "skip"


@pytest.mark.asyncio
async def test_run_spec_stdout_compare(tmp_path):
    inp = tmp_path / "in"
    inp.write_bytes(b"abc\n")
    exp = tmp_path / "exp"
    exp.write_bytes(b"ABC\n")
    spec = VerificationSpec(
        input_path=str(inp), expected_path=str(exp), artifact_path=None,
        run_command="tr a-z A-Z < {INPUT}",
        compare="stdout", row_independent=True,
    )
    state, _ = await run_spec(spec)
    assert state == "pass"


@pytest.mark.asyncio
async def test_run_spec_unreadable_input_is_skip(tmp_path):
    spec = VerificationSpec(
        input_path=str(tmp_path / "nope"), expected_path=str(tmp_path / "nope2"),
        artifact_path=None, run_command="cat {INPUT}",
        compare="in_place", row_independent=True,
    )
    state, _ = await run_spec(spec)
    assert state == "skip"
