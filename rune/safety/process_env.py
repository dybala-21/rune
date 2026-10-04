"""Keep host credentials out of child processes unless explicitly supplied."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

_CREDENTIAL = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:_|$)", re.I)
_PRIVATE = frozenset({"SSH_AUTH_SOCK", "GOOGLE_APPLICATION_CREDENTIALS", "DATABASE_URL"})


def child_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items()
           if not _CREDENTIAL.search(name) and name not in _PRIVATE}
    env.update(overrides or {})
    return env
