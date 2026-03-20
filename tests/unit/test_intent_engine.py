"""Tests for rune.agent.intent_engine — intent classification and contract resolution."""


from rune.agent.intent_engine import (
    IntentContract,
    classify_intent_tier1,
    is_explicit_recall_intent,
    resolve_intent_contract,
)

# ---------------------------------------------------------------------------
# Helpers — fake ClassificationResult-like objects
# ---------------------------------------------------------------------------

class FakeClassification:
    """Minimal duck-typed ClassificationResult for resolve_intent_contract."""

    def __init__(self, **kwargs):
        self.category = kwargs.get("category", "full")
        self.requires_code = kwargs.get("requires_code", False)
        self.requires_execution = kwargs.get("requires_execution", False)
        self.complexity = kwargs.get("complexity", "simple")
        self.is_continuation = kwargs.get("is_continuation", False)
        self.action_type = kwargs.get("action_type", "unspecified")
        self.output_expectation = kwargs.get("output_expectation", "either")


# ---------------------------------------------------------------------------
# classifyIntentTier1
# ---------------------------------------------------------------------------

class TestClassifyIntentTier1:
    def test_returns_tier1_source(self):
        result = classify_intent_tier1("fix login bug")
        assert result.source == "tier1"

    def test_high_confidence_is_resolved(self):
        # A greeting like "hello" should match chat patterns with high confidence
        result = classify_intent_tier1("hello")
        if result.tier1_confidence >= 0.8:
            assert result.resolution == "resolved"
        else:
            assert result.resolution == "unresolved"

    def test_low_confidence_is_unresolved(self):
        # An ambiguous non-English phrase should have low tier1 confidence
        result = classify_intent_tier1("something very ambiguous xyz")
        if result.tier1_confidence < 0.8:
            assert result.resolution == "unresolved"
            assert result.unresolved_reason == "tier1_low_confidence"

    def test_returns_intent_contract(self):
        result = classify_intent_tier1("edit the main.py file")
        assert isinstance(result.intent, IntentContract)

    def test_classification_result_has_tier1(self):
        result = classify_intent_tier1("run pytest")
        assert result.tier1 is not None


# ---------------------------------------------------------------------------
# isExplicitRecallIntent
# ---------------------------------------------------------------------------

class TestIsExplicitRecallIntent:
    def test_recall_intent_for_chat_continuation(self):
        cls = FakeClassification(
            category="chat",
            is_continuation=True,
            requires_code=False,
            requires_execution=False,
        )
        assert is_explicit_recall_intent(0.9, cls) is True

    def test_not_recall_when_requires_code(self):
        cls = FakeClassification(
            category="chat",
            is_continuation=True,
            requires_code=True,
            requires_execution=False,
        )
        assert is_explicit_recall_intent(0.9, cls) is False

    def test_not_recall_when_not_continuation(self):
        cls = FakeClassification(
            category="chat",
            is_continuation=False,
            requires_code=False,
            requires_execution=False,
        )
        assert is_explicit_recall_intent(0.9, cls) is False

    def test_not_recall_when_category_is_code(self):
        cls = FakeClassification(
            category="code",
            is_continuation=True,
            requires_code=False,
            requires_execution=False,
        )
        assert is_explicit_recall_intent(0.9, cls) is False


# ---------------------------------------------------------------------------
# resolveIntentContract
# ---------------------------------------------------------------------------

class TestResolveIntentContract:
    def test_recall_followup_contract(self):
        cls = FakeClassification(
            category="chat",
            is_continuation=True,
            requires_code=False,
            requires_execution=False,
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "recall_followup"
        assert contract.tool_requirement == "none"
        assert contract.output_expectation == "text"

    def test_code_write_contract(self):
        cls = FakeClassification(
            category="code",
            requires_execution=True,
            output_expectation="file",
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "code_write"
        assert contract.tool_requirement == "write"
        assert contract.requires_code_verification is True

    def test_browser_write_contract(self):
        cls = FakeClassification(
            category="browser",
            requires_execution=True,
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "browser_write"
        assert contract.tool_requirement == "write"

    def test_mixed_write_contract(self):
        cls = FakeClassification(
            category="full",
            requires_execution=True,
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "mixed"
        assert contract.tool_requirement == "write"

    def test_web_research_contract(self):
        cls = FakeClassification(category="web")
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "research"
        assert contract.tool_requirement == "read"
        assert contract.grounding_requirement == "required"

    def test_code_read_contract(self):
        cls = FakeClassification(category="code", requires_execution=False)
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "code_read"
        assert contract.tool_requirement == "read"

    def test_full_analyze_complex_requires_grounding(self):
        cls = FakeClassification(
            category="full",
            action_type="analyze",
            complexity="complex",
            requires_execution=False,
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "research"
        assert contract.tool_requirement == "read"
        assert contract.grounding_requirement == "required"

    def test_full_analyze_simple_recommends_grounding(self):
        cls = FakeClassification(
            category="full",
            action_type="analyze",
            complexity="simple",
            requires_execution=False,
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "research"
        assert contract.grounding_requirement == "recommended"

    def test_chat_non_continuation_no_tools(self):
        cls = FakeClassification(
            category="chat",
            is_continuation=False,
            requires_code=False,
            requires_execution=False,
        )
        contract = resolve_intent_contract(cls, 0.9)
        assert contract.kind == "chat"
        assert contract.tool_requirement == "none"
