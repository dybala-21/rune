"""Check arithmetic grounding and reject executable or unquoted expressions."""

import pytest

from rune.agent.calculation import calculate, calculation_context
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.prompts import build_system_prompt


@pytest.mark.parametrize("expression,expected", [
    ("173 × 29 − 417", "4600"), ("0.1 + 0.2", "0.3"),
    ("(12.5 - 2.5) ÷ 4", "2.5"), ("-2 ** 2", "-4"),
    ("1 / 3", "1/3"), ("1e-6 * 3", "0.000003"), ("2 ** -3", "0.125"),
])
def test_exact_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('touch /tmp/unwanted')", "(1).__class__", "2^3",
    "[1] * 100000000", "True + 1", "1,234 + 1", "1 / 0", "1e999 + 1",
    "9 ** 999 ** 999", "2 ** 0.5", "1 + " * 500 + "1", "12345", "20% + 1",
])
def test_unsupported_or_unbounded_expressions_are_rejected(expression):
    with pytest.raises((ValueError, SyntaxError, ArithmeticError)):
        calculate(expression)


def test_prompt_uses_the_computed_value_without_changing_the_goal():
    goal = "173 × 29 − 417의 값을 숫자만으로 답해줘."
    classification = ClassificationResult("chat", 1, 2, calculation_expression="173 × 29 − 417")
    prompt = build_system_prompt(goal=goal, goal_category="chat", classification=classification)
    assert '"result": "4600"' in prompt
    assert goal in prompt


@pytest.mark.parametrize("goal,expression,intents", [
    ("Calculate 10 + 2", "10 + 3", frozenset()),
    ("Use Calculator for 10 + 2", "10 + 2", frozenset({"desktop"})),
    ("My account number is 12345", "12345", frozenset()),
    ("Summarize the notes", "", frozenset()),
])
def test_missing_or_inapplicable_evidence_is_not_injected(goal, expression, intents):
    classification = ClassificationResult("chat", 1, 2, calculation_expression=expression, intent_categories=intents)
    assert calculation_context(goal, classification) == ""
