"""Describe each batch step with the same arguments as its standalone tool."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserExtractParams,
    BrowserFindParams,
    BrowserObserveParams,
    BrowserScreenshotParams,
)
from rune.capabilities.browser.core import BrowserNavigateParams, BrowserOpenParams
from rune.capabilities.browser.discover import BrowserDiscoverApisParams


class NavigateStep(BaseModel):
    type: Literal["navigate"]
    params: BrowserNavigateParams


class OpenStep(BaseModel):
    type: Literal["open"]
    params: BrowserOpenParams


class ObserveStep(BaseModel):
    type: Literal["observe"]
    params: BrowserObserveParams = Field(default_factory=BrowserObserveParams)


class ActStep(BaseModel):
    type: Literal["act"]
    params: BrowserActParams


class ScreenshotStep(BaseModel):
    type: Literal["screenshot"]
    params: BrowserScreenshotParams = Field(default_factory=BrowserScreenshotParams)


class ExtractStep(BaseModel):
    type: Literal["extract"]
    params: BrowserExtractParams


class FindStep(BaseModel):
    type: Literal["find"]
    params: BrowserFindParams


class DiscoverStep(BaseModel):
    type: Literal["discover_apis"]
    params: BrowserDiscoverApisParams = Field(default_factory=BrowserDiscoverApisParams)


BrowserStep = Annotated[
    NavigateStep | OpenStep | ObserveStep | ActStep | ScreenshotStep | ExtractStep | FindStep | DiscoverStep,
    Field(discriminator="type"),
]
