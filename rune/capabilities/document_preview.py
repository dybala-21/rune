"""Render one saved document page without opening a desktop application."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import shlex
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from rune.capabilities.document_inspection import inspect_bytes, read_snapshot
from rune.safety.execution_environment import execution_workspace
from rune.safety.verification import run_check
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


class DocumentPreviewParams(BaseModel):
    path: str = Field(description="Saved PDF, DOCX, PPTX or XLSX file")
    page: int = Field(default=1, ge=1, le=500, description="One-based rendered page to inspect")


def _check_package(data: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if "vbaproject" in member.filename.lower():
                raise ValueError("Macro-bearing documents are not rendered")
            if member.filename.endswith(".rels"):
                root = ElementTree.fromstring(archive.read(member))
                for relationship in root:
                    if (relationship.get("TargetMode") == "External"
                            and not relationship.get("Type", "").endswith("/hyperlink")):
                        raise ValueError("Documents with external linked content are not rendered")


async def document_preview(params: DocumentPreviewParams) -> CapabilityResult:
    try:
        path, data = await asyncio.to_thread(read_snapshot, params.path)
        fmt = path.suffix.lower().lstrip(".")
        if fmt not in {"pdf", "docx", "pptx", "xlsx"}:
            raise ValueError("Preview supports PDF, DOCX, PPTX and XLSX")
        await asyncio.to_thread(inspect_bytes, data, fmt, 1000)
        if fmt != "pdf":
            await asyncio.to_thread(_check_package, data)
        workspace = Path(execution_workspace()).resolve()
        with tempfile.TemporaryDirectory(prefix=".rune-preview-", dir=workspace) as temporary:
            root = Path(temporary)
            source = root / f"input.{fmt}"
            source.write_bytes(data)
            pdf = source
            async with asyncio.timeout(65):
                if fmt != "pdf":
                    profile = root / "profile"
                    (profile / "user").mkdir(parents=True)
                    (profile / "user" / "registrymodifications.xcu").write_text(
                        '<oor:items xmlns:oor="http://openoffice.org/2001/registry">'
                        '<item oor:path="/org.openoffice.Office.Common/Security/Scripting">'
                        '<prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop>'
                        '</item></oor:items>')
                    command = shlex.join(["soffice", "--headless", "--norestore", "--nodefault",
                                          f"-env:UserInstallation={profile.as_uri()}", "--convert-to", "pdf",
                                          "--outdir", str(root), str(source)])
                    result = await run_check(command, str(root), 45, limit=100_000)
                    pdf = root / "input.pdf"
                    if result.code != 0 or not pdf.is_file():
                        raise ValueError("Document renderer unavailable or blocked: " + result.error[-500:])
                _, rendered = await asyncio.to_thread(read_snapshot, pdf)
                evidence = await asyncio.to_thread(inspect_bytes, rendered, "pdf", 1000)
                pages = evidence["facts"]["page_count"]
                if params.page > pages:
                    raise ValueError(f"Rendered document has {pages} pages; requested page {params.page}")
                prefix = root / "page"
                command = shlex.join(["pdftoppm", "-f", str(params.page), "-l", str(params.page),
                                      "-singlefile", "-scale-to", "1400", "-png", str(pdf), str(prefix)])
                result = await run_check(command, str(root), 15, limit=100_000)
                if result.code != 0 or not prefix.with_suffix(".png").is_file():
                    raise ValueError("Page renderer unavailable or blocked: " + result.error[-500:])
                image = prefix.with_suffix(".png").read_bytes()
                from rune.agent.tool_output import ToolImage

                ToolImage.from_bytes(image)
            # Rendering is evidence for inspection, not an automatic quality verdict.
            facts = {"source_sha256": hashlib.sha256(data).hexdigest(), "page_count": pages,
                     "page": params.page, "renderer": "PDF rasterizer" if fmt == "pdf" else "LibreOffice and PDF rasterizer",
                     "visual_verified": False}
            return CapabilityResult(success=True, output=json.dumps(facts) +
                                    "\nInspect the attached page for clipping, overlap and requested layout. Other pages remain uninspected.",
                                    metadata={"image_base64": base64.b64encode(image).decode(), "preview": facts})
    except Exception as exc:
        log.debug("document_preview_unavailable", path=params.path, error=str(exc))
        return CapabilityResult(success=False, error=f"Preview was not verified: {exc}")
