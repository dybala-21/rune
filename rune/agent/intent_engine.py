"""Intent engine - hybrid regex + LLM intent classification.

Ported from src/agent/intent-engine.ts (270 lines) - unified intent
classification facade with contract-based completion verification.

classifyIntent(): Tier-1 regex, then optional Tier-2 LLM fallback.
resolveIntentContract(): Maps GoalClassification to a completion contract.
isExplicitRecallIntent(): Detects recall/status follow-ups that bypass evidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from rune.agent.goal_classifier import (
    ClassificationResult,
    classify_goal,
)
from rune.utils.logger import get_logger

# Compat wrapper - returns low-confidence "full" to trigger Tier 2 LLM classification

@dataclass(slots=True)
class RegexClassifyResult:
    """Compat wrapper with .result and .confidence attributes."""

    result: ClassificationResult
    confidence: float


def regex_classify(
    goal: str,
    _initial_messages: list[Any] | None = None,
) -> RegexClassifyResult:
    """Always returns low-confidence 'full' to defer to LLM classification."""
    fallback = ClassificationResult(
        goal_type="full",
        confidence=0.3,
        tier=2,
        reason="Deferred to LLM",
    )
    return RegexClassifyResult(result=fallback, confidence=0.3)

log = get_logger(__name__)

# Type aliases

IntentKind = Literal[
    "recall_followup",
    "chat",
    "knowledge_explain",
    "research",
    "code_read",
    "code_write",
    "browser_read",
    "browser_write",
    "mixed",
]

ToolRequirement = Literal["none", "read", "write"]
GroundingRequirement = Literal["none", "recommended", "required"]
IntentResolutionState = Literal["resolved", "unresolved"]
IntentUnresolvedReason = Literal[
    "tier1_low_confidence",
    "tier2_fallback",
    "classification_error",
]

OutputExpectation = Literal["text", "file", "either"]


# IntentContract

@dataclass(slots=True)
class IntentContract:
    """Contract describing what tools, grounding, and output are expected."""

    kind: IntentKind
    tool_requirement: ToolRequirement
    grounding_requirement: GroundingRequirement
    output_expectation: OutputExpectation
    requires_code_verification: bool
    requires_code_write_artifact: bool = False


# IntentMessage / Options

@dataclass(slots=True)
class IntentMessage:
    role: str
    content: str


@dataclass(slots=True)
class ClassifyIntentOptions:
    tier1: RegexClassifyResult | None = None
    allow_tier2: bool = True
    structural_context: dict[str, Any] | None = None


# IntentEngineResult

@dataclass(slots=True)
class IntentEngineResult:
    classification: ClassificationResult
    tier1: ClassificationResult
    tier1_confidence: float
    source: Literal["tier1", "tier2", "fallback"]
    intent: IntentContract
    resolution: IntentResolutionState
    unresolved_reason: IntentUnresolvedReason | None = None


# isExplicitRecallIntent

def is_explicit_recall_intent(
    _tier1_confidence: float,
    classification: ClassificationResult,
) -> bool:
    """Recall/status-style follow-up should bypass evidence gate.

    Centralized here so loop no longer hardcodes the predicate inline.
    """
    return (
        getattr(classification, "category", None) == "chat"
        and getattr(classification, "is_continuation", False)
        and not getattr(classification, "requires_code", False)
        and not getattr(classification, "requires_execution", False)
    )


# resolveIntentContract

def resolve_intent_contract(
    classification: ClassificationResult,
    tier1_confidence: float,
) -> IntentContract:
    """Map a GoalClassification to a deterministic IntentContract."""
    output_expectation: OutputExpectation = getattr(
        classification, "output_expectation", "either"
    ) or "either"

    # Recall / follow-up - bypass evidence gate
    if is_explicit_recall_intent(tier1_confidence, classification):
        return IntentContract(
            kind="recall_followup",
            tool_requirement="none",
            grounding_requirement="none",
            output_expectation="text",
            requires_code_verification=False,
        )

    requires_execution = getattr(classification, "requires_execution", False) or (
        getattr(classification, "goal_type", "") in ("execution", "code_modify")
    )
    # Map goal_type to legacy category names used by the contract mapping
    _gt = getattr(classification, "goal_type", "") or getattr(classification, "category", "")
    _CATEGORY_MAP = {
        "code_modify": "code",
        "research": "code",
        "execution": "code",
        "chat": "chat",
        "web": "web",
        "browser": "browser",
        "full": "full",
    }
    category = _CATEGORY_MAP.get(_gt, _gt)

    if requires_execution:
        if category == "code":
            return IntentContract(
                kind="code_write",
                tool_requirement="write",
                grounding_requirement="none",
                output_expectation=output_expectation,
                requires_code_verification=True,
                requires_code_write_artifact=True,
            )
        if category == "browser":
            return IntentContract(
                kind="browser_write",
                tool_requirement="write",
                grounding_requirement="none",
                output_expectation=output_expectation,
                requires_code_verification=True,
            )
        return IntentContract(
            kind="mixed",
            tool_requirement="write",
            grounding_requirement="none",
            output_expectation=output_expectation,
            requires_code_verification=False,
        )

    if category == "web":
        return IntentContract(
            kind="research",
            tool_requirement="read",
            grounding_requirement="required",
            output_expectation="text",
            requires_code_verification=False,
        )

    if category == "browser":
        return IntentContract(
            kind="browser_read",
            tool_requirement="read",
            grounding_requirement="none",
            output_expectation=output_expectation,
            requires_code_verification=False,
        )

    if category == "code":
        return IntentContract(
            kind="code_read",
            tool_requirement="read",
            grounding_requirement="none",
            output_expectation="text",
            requires_code_verification=False,
        )

    if category == "chat":
        is_continuation = getattr(classification, "is_continuation", False)
        tool_req: ToolRequirement = "read" if is_continuation else "none"
        return IntentContract(
            kind="chat",
            tool_requirement=tool_req,
            grounding_requirement="none",
            output_expectation="text",
            requires_code_verification=False,
        )

    if category == "full":
        action_type = getattr(classification, "action_type", "unspecified")
        complexity = getattr(classification, "complexity", "simple")
        is_analyze = action_type in ("analyze", "unspecified")
        is_non_simple = complexity != "simple"

        if is_analyze:
            return IntentContract(
                kind="research",
                tool_requirement="read",
                grounding_requirement="required" if is_non_simple else "recommended",
                output_expectation=output_expectation,
                requires_code_verification=False,
            )

        tool_req_full: ToolRequirement = "write" if output_expectation == "file" else "read"
        return IntentContract(
            kind="mixed" if output_expectation == "file" else "knowledge_explain",
            tool_requirement=tool_req_full,
            grounding_requirement="recommended",
            output_expectation=output_expectation,
            requires_code_verification=False,
        )

    # Exhaustive fallback
    return IntentContract(
        kind="knowledge_explain",
        tool_requirement="read",
        grounding_requirement="recommended",
        output_expectation=output_expectation,
        requires_code_verification=False,
    )


# classifyIntentTier1 (sync)

def classify_intent_tier1(
    goal: str,
    initial_messages: list[IntentMessage] | None = None,
) -> IntentEngineResult:
    """Tier-1 regex-only classification (sync, 0 LLM cost)."""
    tier1 = regex_classify(goal, initial_messages)
    intent = resolve_intent_contract(tier1.result, tier1.confidence)
    resolution: IntentResolutionState = "resolved" if tier1.confidence >= 0.8 else "unresolved"
    return IntentEngineResult(
        classification=tier1.result,
        tier1=tier1.result,
        tier1_confidence=tier1.confidence,
        source="tier1",
        intent=intent,
        resolution=resolution,
        unresolved_reason="tier1_low_confidence" if resolution == "unresolved" else None,
    )


# classifyIntent (async - Tier-1 + optional Tier-2 LLM)

async def classify_intent(
    goal: str,
    initial_messages: list[IntentMessage] | None = None,
    options: ClassifyIntentOptions | None = None,
) -> IntentEngineResult:
    """Unified intent engine facade.

    Step 1: Tier-1 regex classification.
    Step 2: If confidence < 0.8 and Tier-2 is allowed, call LLM-based classifier.
    """
    opts = options or ClassifyIntentOptions()
    tier1 = opts.tier1 or regex_classify(goal, initial_messages)

    classification = tier1.result
    source: Literal["tier1", "tier2"] = "tier1"
    resolution: IntentResolutionState = "resolved" if tier1.confidence >= 0.8 else "unresolved"
    unresolved_reason: IntentUnresolvedReason | None = (
        "tier1_low_confidence" if resolution == "unresolved" else None
    )

    if tier1.confidence < 0.8 and opts.allow_tier2:
        try:
            detailed = await classify_goal(goal)
            classification = detailed
            source = "tier2"
            # If LLM-based, mark resolved
            resolution = "resolved"
            unresolved_reason = None
        except Exception:
            classification = tier1.result
            source = "tier1"
            resolution = "unresolved"
            unresolved_reason = "classification_error"

    return IntentEngineResult(
        classification=classification,
        tier1=tier1.result,
        tier1_confidence=tier1.confidence,
        source=source,
        intent=resolve_intent_contract(classification, tier1.confidence),
        resolution=resolution,
        unresolved_reason=unresolved_reason,
    )
