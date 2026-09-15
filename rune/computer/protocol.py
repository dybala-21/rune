"""Structured desktop requests with no script or shell execution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DesktopError(RuntimeError):
    def __init__(self, message: str, *, outcome: str = "not_executed") -> None:
        super().__init__(message)
        self.outcome = outcome


class DesktopTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    app: str = Field(min_length=1, max_length=200, description="Bundle ID returned by desktop_apps")


class DesktopCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["title", "control"]
    text: str = Field(max_length=2000, description="Expected visible text; an empty value is allowed only with equals")
    match: Literal["equals", "contains"] = "equals"
    role: str = Field(default="", max_length=100, description="Exact accessibility role, such as AXTextArea")
    name: str | None = Field(default=None, max_length=500, description="Optional exact control name to narrow the target")
    field: Literal["name", "value"] = "value"

    @model_validator(mode="after")
    def check_condition(self) -> DesktopCondition:
        if self.kind == "title" and (self.role or self.name is not None or self.field != "value"):
            raise ValueError("A title condition cannot select a control")
        if self.kind == "control" and not self.role.strip():
            raise ValueError("A control condition requires its accessibility role")
        if (self.kind == "title" or self.match == "contains") and not self.text.strip():
            raise ValueError("This condition requires nonempty text")
        return self


class DesktopWait(DesktopTarget):
    condition: DesktopCondition
    timeout_ms: int = Field(default=3000, ge=0, le=10000, description="Polling budget; zero checks once. An in-flight observation finishes under its separate 15-second transport limit.")


class DesktopAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    observation: str = Field(min_length=1, max_length=100)
    action: Literal["press", "set_value", "click", "type", "key", "scroll", "drag", "publish"]
    ref: str = Field(default="", max_length=100)
    text: str = Field(default="", max_length=2000)
    key: Literal["", "return", "tab", "escape", "space", "backspace", "up", "down", "left", "right", "a", "c", "v", "s", "n", "o", "w", "z"] = ""
    modifiers: list[Literal["command", "shift", "option", "control"]] = Field(default_factory=list, max_length=4)
    x: float | None = Field(default=None, ge=0, le=16384, allow_inf_nan=False)
    y: float | None = Field(default=None, ge=0, le=16384, allow_inf_nan=False)
    endX: float | None = Field(default=None, ge=0, le=16384, allow_inf_nan=False)
    endY: float | None = Field(default=None, ge=0, le=16384, allow_inf_nan=False)
    deltaX: int = Field(default=0, ge=-800, le=800)
    deltaY: int = Field(default=0, ge=-800, le=800)

    @model_validator(mode="after")
    def check_fields(self) -> DesktopAction:
        required = {
            "press": {"ref"}, "set_value": {"ref", "text"}, "click": {"x", "y"},
            "type": {"text"}, "key": {"key"}, "scroll": set(), "drag": {"x", "y", "endX", "endY"}, "publish": set(),
        }[self.action]
        allowed = required | {"observation", "action"}
        if self.action == "key":
            allowed.add("modifiers")
        if self.action == "scroll":
            allowed.update({"deltaX", "deltaY"})
        for name, value in self.model_dump().items():
            if name in required and (value is None or value == "" and name != "text"):
                raise ValueError(f"{name} is required for {self.action}")
            if name not in allowed and value not in (None, "", 0, []):
                raise ValueError(f"{name} does not apply to {self.action}")
        if self.action == "scroll" and not (self.deltaX or self.deltaY):
            raise ValueError("Scroll requires a nonzero delta")
        if len(set(self.modifiers)) != len(self.modifiers):
            raise ValueError("Duplicate modifiers")
        return self
