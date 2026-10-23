"""Actionable errors raised before a tool is dispatched."""

import json

from pydantic import ValidationError


def validation_details(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                     for item in error.errors(include_input=False, include_url=False)[:3])[:1000]


class ToolPreparationError(ValueError):
    def __init__(self, code: str, recovery: str, **details):
        self.details = {"code": code, **details, "recovery": recovery}
        super().__init__(json.dumps(self.details, ensure_ascii=False))
