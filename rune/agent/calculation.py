"""Evaluate quoted arithmetic without running model-generated code."""

from __future__ import annotations

import ast
import json
import re
from decimal import Decimal, localcontext
from fractions import Fraction

from rune.utils.logger import get_logger

log = get_logger(__name__)
_OPERATORS = str.maketrans({"×": "*", "÷": "/", "−": "-"})


def calculate(expression: str) -> str:
    if not expression or len(expression) > 512:
        raise ValueError("Expression exceeds the calculation limit")
    source = expression.translate(_OPERATORS).strip()
    if not re.fullmatch(r"[0-9eE.\s+*/()\-]+", source):
        raise ValueError("Only numeric arithmetic is supported")
    tree = ast.parse(source, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 128 or not any(isinstance(node, ast.BinOp) for node in nodes):
        raise ValueError("Expected a bounded arithmetic expression")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(ast.get_source_segment(source, node))
            if not value.is_finite() or abs(value.as_tuple().exponent) > 256:
                raise ValueError("Number exceeds the calculation limit")
            result = Fraction(value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            match node.op:
                case ast.Add():
                    result = left + right
                case ast.Sub():
                    result = left - right
                case ast.Mult():
                    result = left * right
                case ast.Div():
                    result = left / right
                case ast.Pow() if right.denominator == 1 and abs(right) <= 128:
                    result = left ** int(right)
                case _:
                    raise ValueError("Unsupported arithmetic operator")
        else:
            raise ValueError("Unsupported arithmetic syntax")
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > 1024:
            raise ValueError("Result exceeds the calculation limit")
        return result

    result = visit(tree.body)
    denominator = result.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return str(result)
    with localcontext() as context:
        context.prec = 1536
        text = format(Decimal(result.numerator) / Decimal(result.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def calculation_context(goal: str, classification) -> str:
    expression = getattr(classification, "calculation_expression", "")
    if (not getattr(classification, "available", True)
            or "desktop" in (getattr(classification, "intent_categories", ()) or ())
            or not isinstance(expression, str) or not expression.strip() or expression not in goal):
        return ""
    try:
        result = calculate(expression)
    except (ValueError, SyntaxError, ArithmeticError, RecursionError) as exc:
        log.debug("calculation_skipped", reason=type(exc).__name__)
        return ""
    return ("\nLocal arithmetic result (exact; only the quoted expression was evaluated): "
            + json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
            + "\nUse this value when answering the requested calculation. Preserve the user's requested format.")
