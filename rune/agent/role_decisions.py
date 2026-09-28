"""Reuse file-role decisions during one request without caching file state."""

from pathlib import Path

from rune.agent.provenance import ArtifactLedger


class RoleDecisions:
    def __init__(self, request: str, workspace: str):
        self.request = request
        self.workspace = Path(workspace).resolve()
        self.names = ArtifactLedger.for_request(request).referenced
        self.roles: dict[str, str] = {}

    def matches(self, request: str, workspace: str) -> bool:
        return self.request == request and self.workspace == Path(workspace).resolve()

    def remember(self, roles: dict[str, str]) -> None:
        for name, role in roles.items():
            if name in self.names and role in {"input", "output", "preserve"}:
                self.roles.setdefault(name, role)
