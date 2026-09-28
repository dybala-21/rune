"""Describe each batch step with the same arguments as its standalone tool."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserExtractParams,
    BrowserFindParams,
    BrowserObserveParams,
    BrowserScreenshotParams,
)
from rune.capabilities.browser.core import BrowserNavigateParams, BrowserOpenParams
from rune.capabilities.browser.discover import BrowserDiscoverApisParams


def _step_schema(schema: dict) -> None:
    tag = schema["properties"]["type"]
    tag["enum"] = [tag.pop("const")]


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_step_schema)


class NavigateStep(Step):
    type: Literal["navigate"]
    params: BrowserNavigateParams


class OpenStep(Step):
    type: Literal["open"]
    params: BrowserOpenParams


class ObserveStep(Step):
    type: Literal["observe"]
    params: BrowserObserveParams = Field(default_factory=BrowserObserveParams)


class ActStep(Step):
    type: Literal["act"]
    params: BrowserActParams


class ScreenshotStep(Step):
    type: Literal["screenshot"]
    params: BrowserScreenshotParams = Field(default_factory=BrowserScreenshotParams)


class ExtractStep(Step):
    type: Literal["extract"]
    params: BrowserExtractParams


class FindStep(Step):
    type: Literal["find"]
    params: BrowserFindParams


class DiscoverStep(Step):
    type: Literal["discover_apis"]
    params: BrowserDiscoverApisParams = Field(default_factory=BrowserDiscoverApisParams)


BrowserStep = Annotated[
    NavigateStep | OpenStep | ObserveStep | ActStep | ScreenshotStep | ExtractStep | FindStep | DiscoverStep,
    Field(discriminator="type"),
]
