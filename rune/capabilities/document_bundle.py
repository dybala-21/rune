"""Render and check an office bundle before publishing its version.

Readers resolve current.json once to get the files for one version. Concurrent
publishes use the last completed version. Checks cover saved content and metric
consistency; they do not validate prose accuracy, formulas, or visual layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rune.agent.isolation import enforce
from rune.capabilities.bundle_data import (
    BundleMetric,
    RowFilter,
    calculate,
    read_table,
    resolve_references,
)
from rune.capabilities.document import (
    _READERS,
    _RENDERERS,
    DocBlock,
    DocSheet,
    DocumentCreateParams,
)
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.safety.guardian import get_guardian
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)
BundleFormat = Literal["xlsx", "docx", "pptx", "pdf"]


class BundleDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(description="Simple filename including format extension, no directories")
    format: BundleFormat
    title: str = ""
    font_family: str = ""
    blocks: list[DocBlock] = Field(default_factory=list)
    sheets: list[DocSheet] = Field(default_factory=list)


class DocumentBundleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str = Field(min_length=1, description="Bundle output folder, with current.json and versions/")
    source_path: str = Field(min_length=1, description="CSV or values-only XLSX source, first row contains headers")
    sheet: str | None = Field(default=None, description="Explicit source sheet if workbook has multiple tabs")
    filters: list[RowFilter] = Field(default_factory=list, description="All filters apply together (AND)")
    metrics: list[BundleMetric] = Field(min_length=1, max_length=100)
    documents: list[BundleDocument] = Field(
        min_length=1, max_length=12,
        description="Use {{metric_id}} references in prose/table cells; resend spec to update all files",
    )


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)


def _verify_file(path: Path, params: DocumentCreateParams) -> None:
    """Compare saved cells or extracted text with the requested content."""
    if params.format == "xlsx":
        from openpyxl import load_workbook  # type: ignore[import-untyped]

        wb = load_workbook(path, read_only=True, data_only=False)
        try:
            if wb.sheetnames != [s.name for s in params.sheets]:
                raise ValueError("Saved sheet names differ from the requested names")
            for spec in params.sheets:
                actual = list(wb[spec.name].values)
                # Excel reads empty strings as None and pads ragged rows.
                width = max(len(r) for r in spec.rows)
                expected = [tuple(None if v == "" else v for v in row) +
                            (None,) * (width - len(row)) for row in spec.rows]
                if actual != expected:
                    raise ValueError(f"Saved values differ in sheet {spec.name}")
        finally:
            wb.close()
        return
    text = " ".join(_READERS[params.format][0](path).split())
    fragments = [params.title]
    for block in params.blocks:
        if block.type in ("heading", "paragraph"):
            fragments.append(block.text)
        elif block.type == "bullets":
            fragments.extend(block.items)
        elif block.type == "table":
            fragments.extend(str(c) for row in block.rows for c in row)
    for fragment in fragments:
        if " ".join(fragment.split()) not in text:
            raise ValueError(f"Saved {params.format} lost supplied text: {fragment[:80]!r}")


def _prepare_documents(
    params: DocumentBundleParams, values: dict[str, int | float], display: dict[str, str],
) -> list[DocumentCreateParams]:
    prepared = []
    names: set[str] = set()
    for doc in params.documents:
        name = doc.filename
        if (not name or name.startswith(".") or "/" in name or "\\" in name
                or Path(name).suffix != f".{doc.format}" or name.casefold() in names):
            raise ValueError(f"Invalid or duplicate document filename: {name}")
        names.add(name.casefold())
        spec = doc.model_dump(exclude={"filename", "format"})
        resolved = resolve_references(spec, values, display)
        # Keep prose fields as text when a reference resolves to a number.
        resolved["title"] = str(resolved["title"])
        for block in resolved["blocks"]:
            block["text"] = str(block["text"])
            block["items"] = [str(v) for v in block["items"]]
        if doc.format == "xlsx":
            if not resolved["sheets"] or any(not s["rows"] for s in resolved["sheets"]):
                raise ValueError("Bundle XLSX requires explicit nonempty sheets")
            for sheet in resolved["sheets"]:
                for row in sheet["rows"]:
                    if any(isinstance(v, str) and v.startswith("=") for v in row):
                        raise ValueError("Bundle output cells must be values, not unevaluated formulas")
        prepared.append(DocumentCreateParams(path=name, format=doc.format, **resolved))
    return prepared


async def document_bundle(params: DocumentBundleParams) -> CapabilityResult:
    try:
        guardian = get_guardian()
        root = Path(params.directory).expanduser().resolve()
        source = Path(params.source_path).expanduser().resolve()
        for check in (guardian.validate_file_path(str(root)),
                      guardian.validate_file_read_path(str(source))):
            if not check.allowed:
                return CapabilityResult(success=False, error=check.reason)
        if error := enforce(str(root)):
            return CapabilityResult(success=False, error=error)
        if root == source or root in source.parents:
            raise ValueError("Keep the source outside the bundle output directory")
        if source.stat().st_size > 10_000_000:
            raise ValueError("Source exceeds 10 MB")
        data = source.read_bytes()
        rows = read_table(data, source.suffix.lower(), params.sheet)
        values, display, selected = calculate(rows, params.filters, params.metrics)
        documents = _prepare_documents(params, values, display)
        # Reject symlinks that redirect writes outside the bundle.
        versions = root / "versions"
        if versions.is_symlink() or (root / "current.json").is_symlink():
            raise ValueError("Bundle control paths must not be symlinks")
        versions.mkdir(parents=True, exist_ok=True)
        revision = uuid.uuid4().hex
        final = versions / revision
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as temp:
            stage = Path(temp)
            staged_version = stage / "version"
            staged_version.mkdir()
            artifacts = []
            for doc in documents:
                output = staged_version / doc.path
                _RENDERERS[doc.format][0](output, doc)
                _verify_file(output, doc)
                artifacts.append({"path": str(final / doc.path), "format": doc.format,
                                  "sha256": _hash(output.read_bytes()), "bytes": output.stat().st_size})
            if _hash(source.read_bytes()) != _hash(data):
                raise ValueError("Source changed during rendering; regenerate from its new version")
            manifest = {
                "schema_version": 1, "revision": revision,
                "source": {"path": str(source), "sha256": _hash(data), "sheet": params.sheet},
                "rows_total": len(rows), "rows_selected": selected,
                "metrics": values, "display_values": display,
                "spec": params.model_dump(), "artifacts": artifacts,
                "verification": {"status": "pass", "checks": ["native_readback", "metric_resolution"],
                                 "visual_review": "not_performed"},
            }
            (staged_version / "manifest.json").write_text(_json(manifest), encoding="utf-8")
            staged_version.rename(final)
            # Publish all files by replacing the version pointer once.
            pointer = stage / "current.json"
            pointer.write_text(_json({"revision": revision, "manifest": str(final / "manifest.json")}),
                               encoding="utf-8")
            os.replace(pointer, root / "current.json")
        paths = [a["path"] for a in artifacts]
        return CapabilityResult(
            success=True,
            output=_json({"revision": revision, "metrics": values, "files": paths,
                          "manifest": str(final / "manifest.json"),
                          "verification": manifest["verification"]}),
            metadata={"paths": paths + [str(root / "current.json")],
                      "manifest": str(final / "manifest.json"), "revision": revision,
                      "verified": True, "metrics": values},
        )
    except Exception as exc:
        log.warning("document_bundle_failed", error=str(exc))
        return CapabilityResult(success=False, error=f"Bundle was not published: {exc}")


def register_document_bundle_capability(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="document_bundle",
        description=("Create/update consistent XLSX, DOCX, PPTX and PDF files from one CSV/XLSX source. "
                     "Specify row filters, aggregates and {{metric_id}} references in document content. "
                     "Reopens every file and publishes a complete version with source/output hashes. "
                     "Resend the full spec with changed filters to update all deliverables together."),
        domain=Domain.FILE, risk_level=RiskLevel.MEDIUM, group="write",
        parameters_model=DocumentBundleParams, execute=document_bundle,
    ))
