"""Adapt function schemas without changing their local validation rules."""

from copy import deepcopy

_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_SCHEMAS = {"items", "additionalProperties", "additionalItems", "contains", "not", "if", "then", "else",
            "propertyNames", "unevaluatedProperties", "unevaluatedItems"}
_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}


def gemini_tools(model: str, tools: list[dict] | None, choice) -> list[dict] | None:
    provider, _, name = model.partition("/")
    forced = choice == "required" or isinstance(choice, dict) and choice.get("type") == "function"
    if (not tools or not forced or provider not in {"gemini", "vertex_ai", "vertex_ai_beta"}
            or not name.startswith("gemini-")):
        return tools
    result = deepcopy(tools)

    def visit(schema):
        if not isinstance(schema, dict):
            return
        maximum = schema.get("maxItems")
        if type(maximum) is int and maximum >= 0:
            # ANY can reject bounded arrays; execution still validates the original schema.
            schema.pop("maxItems")
            description = schema.get("description", "")
            schema["description"] = f"{description.rstrip()} At most {maximum} items.".lstrip()
        for key, child in schema.items():
            if key in _MAPS and isinstance(child, dict):
                for subschema in child.values():
                    visit(subschema)
            elif key in _SCHEMAS and isinstance(child, dict):
                visit(child)
            elif (key in _LISTS or key == "items") and isinstance(child, list):
                for subschema in child:
                    visit(subschema)
        variants = schema.get("anyOf")
        description = schema.get("description")
        if (isinstance(description, str) and isinstance(variants, list) and variants
                and all(isinstance(branch, dict) for branch in variants)):
            # LiteLLM otherwise replaces branch descriptions with the parent's text.
            schema.pop("description")
            for branch in variants:
                detail = branch.get("description", "")
                branch["description"] = description if detail == description else f"{description} {detail}".strip()

    for tool in result:
        if tool.get("type") == "function":
            visit(tool.get("function", {}).get("parameters"))
    return result
