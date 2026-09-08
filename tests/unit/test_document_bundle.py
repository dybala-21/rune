"""Outcome checks for office versions, failures, source integrity and tool wiring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.verified_workflows.demo import SALES, specification
from rune.capabilities.document import DocSheet, DocumentCreateParams, _read_pdf, document_create
from rune.capabilities.document_bundle import document_bundle


@pytest.fixture
def office(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_BUNDLE_UPDATE_ENABLED", "1")
    guardian = SimpleNamespace(
        validate_file_path=lambda _: SimpleNamespace(allowed=True),
        validate_file_read_path=lambda _: SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr("rune.capabilities.document.get_guardian", lambda: guardian)
    monkeypatch.setattr("rune.capabilities.document_bundle.get_guardian", lambda: guardian)
    source = tmp_path / "sales.csv"
    import csv

    with source.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(SALES)
    params = specification(source, tmp_path / "bundle", confirmed_only=False)
    # Font availability is tested separately; core bundle tests are portable.
    params.documents = [d for d in params.documents if d.format != "pdf"]
    return params


@pytest.mark.asyncio
async def test_condition_change_updates_every_native_file(office):
    from openpyxl import load_workbook

    from rune.capabilities.document import _read_docx, _read_pptx

    initial = await document_bundle(office)
    assert initial.success, initial.error
    before = Path(initial.metadata["manifest"]).read_bytes()
    updated = specification(Path(office.source_path), Path(office.directory), confirmed_only=True)
    updated.documents = [d for d in updated.documents if d.format != "pdf"]
    result = await document_bundle(updated)
    assert result.success, result.error
    manifest = json.loads(Path(result.metadata["manifest"]).read_text())
    assert manifest["metrics"] == {"amount": 23_000_000, "orders": 3}
    assert manifest["rows_selected"] == 3
    assert manifest["source"]["sha256"] == hashlib.sha256(Path(office.source_path).read_bytes()).hexdigest()
    files = {a["format"]: Path(a["path"]) for a in manifest["artifacts"]}
    wb = load_workbook(files["xlsx"], data_only=True)
    assert wb["집계"]["B2"].value == 23_000_000
    assert wb["집계"]["B3"].value == 3
    wb.close()
    for reader, fmt in [(_read_docx, "docx"), (_read_pptx, "pptx")]:
        text = reader(files[fmt])
        assert "23,000,000원" in text and "3건" in text
        assert "34,000,000" not in text and "{{" not in text
    assert Path(initial.metadata["manifest"]).read_bytes() == before
    current = json.loads((Path(office.directory) / "current.json").read_text())
    assert current["revision"] == result.metadata["revision"]
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["render", "readback", "source_changed", "reference"])
async def test_failed_update_preserves_previous_version(office, monkeypatch, failure):
    import rune.capabilities.document_bundle as bundle

    initial = await document_bundle(office)
    assert initial.success
    pointer = Path(office.directory) / "current.json"
    before = pointer.read_bytes()
    if failure == "reference":
        office.documents[1].title = "{{unknown}}"
    else:
        original = bundle._RENDERERS["docx"][0]

        def render(path, params):
            if failure == "render":
                raise RuntimeError("injected renderer failure")
            original(path, params)
            if failure == "readback":
                path.write_bytes(b"broken native artifact")
            else:
                Path(office.source_path).write_text("changed", encoding="utf-8")

        monkeypatch.setitem(bundle._RENDERERS, "docx", (render, "python-docx"))
        from rune.capabilities.document_worker import render_snapshot

        async def injected_render(payload):
            return render_snapshot(payload)

        # Inject at the renderer boundary; other tests use the real spawn worker.
        monkeypatch.setattr(bundle, "_render_staged", injected_render)
    result = await document_bundle(office)
    assert not result.success
    assert pointer.read_bytes() == before
    assert len(list((pointer.parent / "versions").iterdir())) == 1


@pytest.mark.asyncio
async def test_formula_source_rejected_and_path_cannot_escape(office, monkeypatch, tmp_path):
    source = tmp_path / "formulas.xlsx"
    result = await document_create(DocumentCreateParams(
        path=str(source), format="xlsx", sheets=[DocSheet(name="주문", rows=[["value"], ["=1+1"]])],
    ))
    assert result.success
    office.source_path = str(source)
    result = await document_bundle(office)
    assert not result.success and "formulas" in result.error
    office.source_path = str(tmp_path / "sales.csv")
    office.documents[0].filename = "../escape.xlsx"
    result = await document_bundle(office)
    assert not result.success and "filename" in result.error
    office.documents[0].filename = "summary.xlsx"
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path / "isolated"))
    result = await document_bundle(office)
    assert not result.success and "Isolation" in result.error
    assert not (tmp_path / "escape.xlsx").exists()


@pytest.mark.asyncio
async def test_tool_is_available_and_anchors_workspace_paths(office, monkeypatch, tmp_path):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.document_bundle import register_document_bundle_capability
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import get_allowed_tools

    assert "document_bundle" in get_allowed_tools("rune")
    assert "document_bundle" not in get_allowed_tools("safe")
    assert "document_read" in get_allowed_tools("safe")
    for goal_type in ("chat", "research"):
        selected = NativeAgentLoop()._select_tools(ClassificationResult(goal_type=goal_type, confidence=1, tier=1))
        assert "document_bundle" in selected
    reg = CapabilityRegistry()
    register_document_bundle_capability(reg)
    tools = build_tool_set(ToolAdapterOptions(workspace_root=str(tmp_path), enable_guardian=False), registry=reg)
    spec = office.model_dump()
    spec["source_path"] = "sales.csv"
    spec["directory"] = "anchored"
    output = await tools["document_bundle"].function(**spec)
    assert (tmp_path / "anchored/current.json").exists(), output


@pytest.mark.asyncio
async def test_pdf_preserves_unicode_and_long_cells(office, monkeypatch, tmp_path):
    from rune.capabilities.document_fonts import pdf_font

    text = "한글 주문 확인 문장이 잘리지 않고 끝까지 보여야 합니다"
    try:
        font = pdf_font(text)
    except ValueError:
        pytest.skip("No locally installed Korean font")
    assert font
    out = tmp_path / "korean.pdf"
    result = await document_create(DocumentCreateParams(
        path=str(out), format="pdf", title="주문 현황",
        blocks=[{"type": "table", "rows": [["내용", "금액"], [text, 23000000]]}],
    ))
    assert result.success, result.error
    assert "".join(text.split()) in "".join(_read_pdf(out).split())
    monkeypatch.setenv("RUNE_DOCUMENT_FONT", str(tmp_path / "missing.ttf"))
    result = await document_create(DocumentCreateParams(path=str(out), format="pdf", title="한글"))
    assert not result.success and "RUNE_DOCUMENT_FONT" in result.error
