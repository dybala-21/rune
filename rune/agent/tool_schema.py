"""Remove redundant schema labels from model requests."""

from copy import deepcopy

CALL_RATIONALE = {"type": "string", "description": "Brief reasoning for this call (removed before execution)"}
_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_SCHEMAS = {"items", "additionalProperties", "additionalItems", "contains", "not", "if", "then", "else",
            "propertyNames", "unevaluatedProperties", "unevaluatedItems"}
_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}


def compact_parameters(schema: dict) -> dict:
    result = deepcopy(schema)

    def label(value):
        return "".join(char.lower() for char in value if char.isalnum())

    def visit(value, name=""):
        if not isinstance(value, dict):
            return
        title = value.get("title")
        if name and isinstance(title, str) and label(title) == label(name):
            value.pop("title")
        for key, child in value.items():
            if key in _MAPS and isinstance(child, dict):
                for field, subschema in child.items():
                    visit(subschema, field)
            elif key in _SCHEMAS and isinstance(child, dict):
                visit(child)
            elif (key in _LISTS or key == "items") and isinstance(child, list):
                for subschema in child:
                    visit(subschema)

    visit(result)
    return result


def without_call_rationale(schemas: list[dict] | None) -> list[dict] | None:
    """Remove Rune's optional reasoning field, preserving tool-defined fields."""
    if not schemas:
        return schemas
    result = []
    for tool in schemas:
        function = tool.get("function", {})
        params = function.get("parameters", {})
        properties = params.get("properties", {})
        if properties.get("think") == CALL_RATIONALE and "think" not in params.get("required", []):
            tool = {**tool, "function": {**function, "parameters": {
                **params, "properties": {key: value for key, value in properties.items() if key != "think"},
            }}}
        result.append(tool)
    return result
