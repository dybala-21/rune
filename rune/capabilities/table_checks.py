"""Agent tools for fixing table requirements and checking saved results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


class TableRequirementsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_path: str = Field(min_length=1, description="Original CSV or values-only XLSX source; never a generated substitute")
    sheet: str | None = None


class TableVerifyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str = Field(min_length=1)
    sheet: str | None = None
    header_row: int = Field(default=1, ge=1, le=100)


async def table_requirements(params: TableRequirementsParams) -> CapabilityResult:
    return await _call("requirements", params.model_dump())


async def table_verify(params: TableVerifyParams) -> CapabilityResult:
    return await _call("verify", params.model_dump())


async def _call(method: str, params: dict) -> CapabilityResult:
    from rune.agent.table_acceptance import active_acceptance

    state = active_acceptance()
    if state is None:
        return CapabilityResult(success=False, error="Table verification needs the current agent request context")
    try:
        return await getattr(state, method)(**params)
    except Exception as exc:
        log.info("table_check_failed", method=method, error=str(exc))
        return CapabilityResult(success=False, error=str(exc))


def register_table_checks(registry: CapabilityRegistry) -> None:
    for name, params, execute, description in (
        ("table_requirements", TableRequirementsParams, table_requirements,
         "Before producing an aggregate CSV/XLSX from source data, fix the data requirements from the original user request. "
         "Returns an immutable contract, exact output columns, filters, duplicate policy and unsupported requirements. "
         "The agent cannot supply or change the acceptance conditions. Use table_verify after creating the output."),
        ("table_verify", TableVerifyParams, table_verify,
         "Check a saved CSV/XLSX aggregate table against a table_requirements contract. Independently recalculates "
         "source totals, filters, duplicates and groups, and detects missing/extra/wrong rows. Fix failures and rerun "
         "after every change. Checks the selected table's data; does not certify prose, formulas or visual layout."),
    ):
        registry.register(CapabilityDefinition(name=name, description=description, parameters_model=params,
                                               execute=execute, domain=Domain.FILE, risk_level=RiskLevel.LOW, group="read"))
