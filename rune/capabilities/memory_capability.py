"""Memory tool capabilities for RUNE.

Ported from src/capabilities/memory.ts - wraps the MemoryManager
to provide semantic search and fact storage as agent tools.

MemGPT pattern: the agent manages its own memory.  ``memory_read``
(semantic search) and ``memory_save`` (with section-normalised types
and schema validation) give the model self-memory capabilities.
``memory_tune`` adjusts retrieval policy parameters.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

# MemGPT-style section normalisation

MemoryType = Literal["preference", "environment", "decision", "pattern", "note"]

_MEMORY_SECTION_MAP: dict[MemoryType, str] = {
    "preference": "Preferences",
    "environment": "Environment",
    "decision": "Decisions",
    "pattern": "Patterns",
    "note": "Notes",
}


def _normalise_memory_section(mem_type: MemoryType) -> str:
    """Map a memory type to its canonical section heading."""
    return _MEMORY_SECTION_MAP.get(mem_type, "Notes")


# Parameter schemas

class MemorySearchParams(BaseModel):
    """Parameters for memory.search (MemGPT memory_read)."""

    query: str = Field(description="Semantic search query (natural language or keywords)")
    type: Literal["episode", "fact", "command"] | None = Field(
        default=None,
        description="Filter by entry type (omit to search all)",
    )
    limit: int = Field(default=5, alias="maxResults", description="Maximum results (default: 5)")
    min_score: float = Field(default=0.3, alias="minScore",
                             description="Minimum similarity score (0-1)")


class MemorySaveParams(BaseModel):
    """Parameters for memory.save (MemGPT memory_save).

    The *type* field determines which section heading the entry is filed
    under, following the TS ``normalizeMemorySection`` logic.
    """

    content: str = Field(description="Content to store")
    type: MemoryType = Field(description="Memory type (preference/decision/pattern/note/environment)")
    key: str | None = Field(
        default=None,
        description="Key identifier (used for preference / environment types)",
    )

    # Keep backward compat aliases
    value: str | None = Field(default=None, exclude=True)
    category: str | None = Field(default=None, exclude=True)
    scope: str = Field(default="user", description="Scope (user/project)")

    @field_validator("content", mode="before")
    @classmethod
    def _backfill_content(cls, v: str, info: Any) -> str:  # type: ignore[override]
        """Accept legacy ``value`` field as ``content``."""
        if not v and info.data.get("value"):
            return info.data["value"]
        return v


# Allow pydantic to accept extra fields for backward compat
class _MemorySaveParamsCompat(MemorySaveParams):
    model_config = ConfigDict(populate_by_name=True)


class MemoryTuneParams(BaseModel):
    """Parameters for memory.tune to adjust retrieval policy knobs."""

    scope: Literal["user", "project"] = Field(
        default="project", description="Settings storage scope"
    )
    preset: Literal["speed", "balanced", "accuracy"] | None = Field(
        default=None, description="Memory tuning preset"
    )
    policy_mode: Literal["auto", "legacy", "shadow", "balanced", "strict"] | None = Field(
        default=None, description="Memory policy mode"
    )
    uncertain_score_threshold: float | None = Field(
        default=None, ge=0, le=1,
        description="Uncertain intent score threshold (0-1)",
    )
    uncertain_relevance_floor: float | None = Field(
        default=None, ge=0, le=1,
        description="Uncertain intent relevance floor (0-1)",
    )
    uncertain_semantic_limit: int | None = Field(
        default=None, ge=1, le=20,
        description="Semantic search result count for uncertain intents (1-20)",
    )
    uncertain_semantic_min_score: float | None = Field(
        default=None, ge=0, le=1,
        description="Minimum semantic score for uncertain intents (0-1)",
    )


# Implementations

async def memory_search(params: MemorySearchParams) -> CapabilityResult:
    """Semantic search across the memory store."""
    log.debug("memory_search", query=params.query, limit=params.limit)

    try:
        from rune.memory.manager import get_memory_manager

        manager = get_memory_manager()
        await manager.initialize()

        results = await manager.search(params.query, k=params.limit)

        # Filter by minimum score
        results = [r for r in results if r.score >= params.min_score]

        if not results:
            # Fall back to exact match in working memory
            working = manager.working
            matches: list[dict] = []
            query_lower = params.query.lower()
            for key, value in working.facts.items():
                if query_lower in key.lower() or query_lower in value.lower():
                    matches.append({"key": key, "value": value, "score": 1.0})

            # Also search project MEMORY.md file
            try:
                import os

                from rune.memory.project_memory import read_project_memory_head
                md_content = read_project_memory_head(os.getcwd())
                if md_content:
                    for line in md_content.splitlines():
                        line = line.strip()
                        if line and query_lower in line.lower():
                            matches.append({"key": "project_memory", "value": line, "score": 0.8})
            except Exception:
                pass

            if not matches:
                return CapabilityResult(
                    success=True,
                    output="No matching memories found.",
                    metadata={"query": params.query, "count": 0},
                )

            lines: list[str] = [f"Found {len(matches)} matching fact(s):"]
            for m in matches[: params.limit]:
                lines.append(f"  [{m['key']}] {m['value']}")

            return CapabilityResult(
                success=True,
                output="\n".join(lines),
                metadata={"query": params.query, "count": len(matches)},
            )

        # Format vector search results
        lines: list[str] = [f"Found {len(results)} memory result(s):"]
        for r in results:
            score_str = f"{r.score:.2f}"
            summary = r.metadata.summary if r.metadata else ""
            entry_type = r.metadata.type if r.metadata else "unknown"
            lines.append(f"  [{score_str}] ({entry_type}) {summary}")

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "query": params.query,
                "count": len(results),
                "results": [
                    {
                        "score": r.score,
                        "type": r.metadata.type if r.metadata else "",
                        "summary": r.metadata.summary if r.metadata else "",
                    }
                    for r in results
                ],
            },
        )

    except Exception as exc:
        log.warning("memory_search_failed", error=str(exc))
        return CapabilityResult(
            success=False,
            error=f"Memory search failed: {exc}",
        )


async def memory_save(params: MemorySaveParams) -> CapabilityResult:
    """Save a memory entry with MemGPT-style section normalisation.

    The *type* field determines which section heading the entry is filed
    under (Preferences, Environment, Decisions, Patterns, Notes).
    Schema validation is enforced by the Pydantic model.
    """
    section = _normalise_memory_section(params.type)
    key = params.key or params.type
    log.debug("memory_save", key=key, section=section, scope=params.scope)

    try:
        from rune.memory.manager import get_memory_manager
        from rune.memory.store import Fact

        manager = get_memory_manager()
        await manager.initialize()

        fact = Fact(
            key=key,
            value=params.content,
            category=params.type,
            source=f"agent:{params.scope}",
        )

        await manager.save_fact(fact)

        # Also attempt project memory append for preference/environment types
        if params.type in ("preference", "environment"):
            try:
                import os

                from rune.memory.project_memory import append_project_memory

                workspace = os.getcwd()
                append_project_memory(workspace, section, f"- {params.content}")
            except Exception:
                pass  # best-effort

        return CapabilityResult(
            success=True,
            output=(
                f"Saved to [{section}]: {params.content}"
                + (f" (key: {key})" if params.key else "")
            ),
            metadata={
                "key": key,
                "type": params.type,
                "section": section,
                "scope": params.scope,
            },
        )

    except Exception as exc:
        log.warning("memory_save_failed", error=str(exc))
        return CapabilityResult(
            success=False,
            error=f"Memory save failed: {exc}",
        )


async def memory_tune(params: MemoryTuneParams) -> CapabilityResult:
    """Adjust memory retrieval policy parameters.

    Ported from the TS ``memory.tune`` capability.  Accepts a preset
    (speed/balanced/accuracy) or individual knobs for the uncertain-query
    pipeline.
    """
    log.info("memory_tune", scope=params.scope, preset=params.preset)

    applied: dict[str, Any] = {}

    # Presets override individual knobs
    if params.preset == "speed":
        applied.update(
            uncertain_score_threshold=0.6,
            uncertain_relevance_floor=0.4,
            uncertain_semantic_limit=3,
        )
    elif params.preset == "accuracy":
        applied.update(
            uncertain_score_threshold=0.3,
            uncertain_relevance_floor=0.2,
            uncertain_semantic_limit=10,
        )
    elif params.preset == "balanced":
        applied.update(
            uncertain_score_threshold=0.45,
            uncertain_relevance_floor=0.3,
            uncertain_semantic_limit=5,
        )

    # Individual knobs override preset values
    if params.uncertain_score_threshold is not None:
        applied["uncertain_score_threshold"] = params.uncertain_score_threshold
    if params.uncertain_relevance_floor is not None:
        applied["uncertain_relevance_floor"] = params.uncertain_relevance_floor
    if params.uncertain_semantic_limit is not None:
        applied["uncertain_semantic_limit"] = params.uncertain_semantic_limit
    if params.uncertain_semantic_min_score is not None:
        applied["uncertain_semantic_min_score"] = params.uncertain_semantic_min_score
    if params.policy_mode is not None:
        applied["policy_mode"] = params.policy_mode

    if not applied:
        return CapabilityResult(
            success=False,
            error="At least one tuning field or preset must be provided",
        )

    # Persist via env or config
    try:
        from rune.utils.env import set_env

        for k, v in applied.items():
            env_key = f"RUNE_MEMORY_{k.upper()}"
            set_env(env_key, str(v))
    except Exception:
        pass  # best-effort persistence

    return CapabilityResult(
        success=True,
        output=f"Memory tuning applied ({params.scope}): {applied}",
        metadata={"scope": params.scope, "applied": applied},
    )


# Registration

def register_memory_capabilities(registry: CapabilityRegistry) -> None:
    """Register memory capabilities (search, save, tune)."""
    registry.register(CapabilityDefinition(
        name="memory_search",
        description="Search past work, user preferences, and learned patterns from memory",
        domain=Domain.MEMORY,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=MemorySearchParams,
        execute=memory_search,
    ))
    registry.register(CapabilityDefinition(
        name="memory_save",
        description=(
            "Save a memory entry (preference/decision/pattern/note/environment) "
            "with MemGPT-style self-memory management"
        ),
        domain=Domain.MEMORY,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=MemorySaveParams,
        execute=memory_save,
    ))
    registry.register(CapabilityDefinition(
        name="memory_tune",
        description="Adjust memory retrieval policy parameters and thresholds",
        domain=Domain.MEMORY,
        risk_level=RiskLevel.MEDIUM,
        group="write",
        parameters_model=MemoryTuneParams,
        execute=memory_tune,
    ))
