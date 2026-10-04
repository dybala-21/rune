"""The connector policy and request format shared by the controller and broker."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def checked_path(value: str) -> str:
    # Reject ambiguous paths before a proxy or server can normalize them differently.
    if (not re.fullmatch(r"/[A-Za-z0-9_./~:@!$&'()*+,;=-]*", value)
            or "//" in value or any(part in {".", ".."} for part in value.split("/"))):
        raise ValueError("Use an absolute, unescaped API path without dot segments")
    return value


class ConnectorPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    origin: str
    path_prefix: str = "/"
    methods: list[Literal["GET", "POST", "PUT", "PATCH", "DELETE"]] = Field(min_length=1)
    header: Literal["Authorization", "X-API-Key"] = "Authorization"
    scheme: Literal["Bearer", "raw"] = "Bearer"

    @field_validator("origin")
    @classmethod
    def origin_only(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.port not in {None, 443} or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or "\\" in value):
            raise ValueError("An HTTPS origin on port 443 is required")
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        if not re.fullmatch(r"[a-z0-9.-]+", host) or host.endswith("."):
            raise ValueError("Use a DNS hostname without a trailing dot")
        return f"https://{host}"

    _path = field_validator("path_prefix")(checked_path)

    def permits(self, request: ConnectorRequest) -> bool:
        prefix = self.path_prefix.rstrip("/")
        return (request.connector == self.name and request.origin == self.origin and request.method in self.methods
                and (request.path == prefix or request.path.startswith(prefix + "/")))


class ConnectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connector: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    origin: str = Field(description="Exact HTTPS origin returned by connector_list; shown with the approval request")
    revision: str = Field(pattern=r"^[a-f0-9]{32}$", description="Exact revision returned by connector_list; changed policies or credentials need fresh approval")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    path: str = Field(max_length=2048)
    query: dict[str, str] = Field(default_factory=dict, max_length=64)
    body: str = Field(default="", max_length=262144)
    max_length: int = Field(default=10000, ge=100, le=50000)
    content_type: Literal["application/json", "application/x-www-form-urlencoded", "text/plain"] = "application/json"

    _path = field_validator("path")(checked_path)
    _origin = field_validator("origin")(ConnectorPolicy.origin_only.__func__)

    @field_validator("query")
    @classmethod
    def bounded_query(cls, value: dict[str, str]) -> dict[str, str]:
        if sum(len(k) + len(v) for k, v in value.items()) > 8192:
            raise ValueError("Query exceeds 8192 characters")
        return value
