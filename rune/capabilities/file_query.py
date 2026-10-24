"""Compute record queries from an authorized file snapshot."""

import csv
import hashlib
import io
from pathlib import Path

from pydantic import BaseModel, Field

from rune.capabilities.file import FileReadParams, file_read
from rune.capabilities.json_records import RecordQuery, dump_json, query_records
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel


class FileQueryParams(BaseModel):
    path: str = Field(description="Source CSV, TSV or JSON file")
    query: RecordQuery


async def file_query(params: FileQueryParams) -> CapabilityResult:
    source = await file_read(FileReadParams(path=params.path, raw=True, maxSize=1_000_000))
    if not source.success:
        return source
    text = source.output or ""
    suffix = Path(source.metadata["path"]).suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            if params.query.path:
                raise ValueError("CSV/TSV queries use the root array; omit query.path")
            reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter="," if suffix == ".csv" else "\t")
            headers = reader.fieldnames or []
            if not headers or len(headers) != len(set(headers)):
                raise ValueError("The table needs unique column names")
            rows = []
            for row in reader:
                if len(rows) >= 10_000 or None in row or any(value is None for value in row.values()):
                    raise ValueError("The table has malformed rows or exceeds 10000 records")
                rows.append(row)
            text = dump_json(rows)
        elif suffix != ".json":
            raise ValueError("Use CSV, TSV or JSON for record queries")
        result, receipt = query_records(text, params.query, numeric_strings=suffix in {".csv", ".tsv"})
        receipt.update(scope="file_snapshot", path=source.metadata["path"],
                       sha256=hashlib.sha256((source.output or "").encode()).hexdigest())
        return CapabilityResult(success=True, output='{"result":' + result + ',"source":' + dump_json(receipt) + '}',
                                metadata={"path": source.metadata["path"], "record_query": receipt})
    except (ValueError, csv.Error) as exc:
        return CapabilityResult(success=False, error=str(exc))


def register_file_query(registry) -> None:
    registry.register(CapabilityDefinition(
        name="file_query", description="Compute an exact count/sum/mean/min/max or latest record from CSV/TSV/JSON, optionally filtered. CSV/TSV filter values are strings. Returns the result and source query; never writes a file.",
        domain=Domain.FILE, risk_level=RiskLevel.LOW, group="file",
        parameters_model=FileQueryParams, execute=file_query,
    ))
