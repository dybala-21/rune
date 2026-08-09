"""Track which files a run actually opened, so it cannot invent one.

Asked to fix the bug described in a file that does not exist, an agent
wrote that file itself, invented a bug, edited unrelated code and reported
the task done. Its own output said the file was missing.

Reading the final answer does not catch this: a wrong answer is phrased
just as confidently as a right one. The read record does. A file the
request treats as already existing, that no read ever found, cannot be the
file the run then writes from scratch.

Two questions, both answered from the tool-call record — no model in the
loop, nothing phrased in any particular language:

  - is this write inventing a file the request treated as existing?
  - did the run finish with such a file still unaccounted for?

File names are matched with a regex because they are a structured format;
nothing here tries to parse the sentence around them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_ARTIFACT_PROVENANCE"

# A file name: a stem plus a short extension. Deliberately narrow — a bare
# word must not be mistaken for a path. The tail is "not another ASCII
# alphanumeric" rather than a word boundary, because in languages that
# attach particles directly ("BUGREPORT.md에") there is no boundary there.
_PATH_RE = re.compile(
    r"(?:[\w./\\-]*[/\\])?[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,7}(?![A-Za-z0-9])"
)

# Extensions that name a document/artifact rather than an inline example.
_SKIP_SUFFIXES = frozenset({
    ".com", ".org", ".net", ".io", ".dev", ".ai", ".co", ".kr", ".jp",
})

_READ_TOOLS = frozenset({"file_read", "file_search", "file_list"})
_WRITE_TOOLS = frozenset({"file_write", "file_edit"})


def provenance_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def referenced_paths(text: str) -> set[str]:
    """File names the request talks about, keyed by base name."""
    out: set[str] = set()
    for m in _PATH_RE.finditer(text or ""):
        token = m.group(0)
        name = os.path.basename(token.replace("\\", "/"))
        suffix = os.path.splitext(name)[1].lower()
        if not suffix or suffix in _SKIP_SUFFIXES:
            continue
        out.add(name)
    return out


def _key(path: str) -> str:
    return os.path.basename(str(path).replace("\\", "/"))


@dataclass
class ArtifactLedger:
    """What the run has actually seen, versus what it was asked about."""

    referenced: set[str] = field(default_factory=set)
    read_ok: set[str] = field(default_factory=set)
    looked_up: set[str] = field(default_factory=set)
    created: set[str] = field(default_factory=set)
    refused: set[str] = field(default_factory=set)
    roles: dict[str, str] = field(default_factory=dict)
    # Paths this run positively observed to be missing. Only these can have
    # been fabricated later; anything else that exists was already there.
    known_absent: set[str] = field(default_factory=set)
    # The tree the request is about. A request names a bare file, so the
    # ledger keys on bare names — which means a file of the same name
    # somewhere else would otherwise answer for it. Empty disables the
    # check, for callers with no workspace to speak of.
    root: str = ""

    @classmethod
    def for_request(cls, request: str, root: str = "") -> ArtifactLedger:
        return cls(referenced=referenced_paths(request), root=root)

    def _within_root(self, path: str) -> bool:
        """Whether *path* is the workspace's copy and not a namesake.

        Observed: asked to fix the bug in a BUGREPORT.md that did not exist,
        the agent searched the parent directory, found an unrelated file of
        that name, and read it. Keyed on the bare name, that read counted as
        having found the requested input — so the guard against writing a
        file the request assumed already existed stopped applying, and the
        run authored one and reported success. Reading someone else's file
        proves nothing about this task's.
        """
        if not self.root:
            return True
        try:
            base = Path(self.root).expanduser().resolve()
            return Path(path).expanduser().resolve().is_relative_to(base)
        except (OSError, ValueError):
            return False

    def record_read(self, path: str, ok: bool) -> None:
        k = _key(path)
        self.looked_up.add(k)
        if ok and self._within_root(path):
            self.read_ok.add(k)
            self.known_absent.discard(k)
        elif not ok:
            self.known_absent.add(k)

    def record_lookup(self, blob: str) -> None:
        """Any call that names a referenced artifact is a search for it.

        Agents locate a file however they like — a read, a glob, `find`,
        `ls | grep`. Tying this to the file tools alone left the obvious
        hole: look with a shell command, then write the file anyway.
        """
        for name in self.referenced:
            if name in blob:
                self.looked_up.add(name)

    def record_write(self, path: str, existed: bool) -> None:
        if not existed:
            self.created.add(_key(path))

    def is_phantom(self, path: str) -> bool:
        """True when writing *path* would conjure an artifact the request
        treated as pre-existing.

        Once the request has been classified, an input that is not there
        is a phantom outright — waiting for the agent to search first
        misses the runs that write immediately. Without a classification
        (offline, a model that could not answer) the older and weaker
        test still applies: it counts only if the run looked and failed.
        Existence is settled by the caller against the filesystem.
        """
        k = _key(path)
        if k not in self.referenced or k in self.read_ok:
            return False
        role = self.roles.get(k)
        if role == "output":
            return False
        if role == "input":
            return True
        return k in self.looked_up

    def unresolved(self) -> list[str]:
        """Inputs the run went looking for and never found."""
        return sorted(
            k for k in self.referenced
            if k in self.looked_up and k not in self.read_ok
            and self.roles.get(k, "input") == "input"
        )


_CLASSIFY_TIMEOUT_S = 20.0

_CLASSIFY_PROMPT = """\
For each file name listed, decide from the request whether it is an INPUT
(the request assumes it already exists and its contents are to be used) or
an OUTPUT (the request asks for it to be produced).

Request:
---
{request}
---

File names: {names}

Reply with JSON only: {{"<name>": "input"|"output", ...}}
"""


async def classify_roles(
    request: str, names: list[str], model: str | None, provider: str | None
) -> dict[str, str]:
    """Ask the model which named files the request treats as inputs.

    Input-versus-output lives in the sentence, so a model decides it — the
    alternative is per-language phrasing rules, which do not survive
    contact with a second language. Any provider will do; a failure here
    returns nothing and the caller falls back to what the tool record
    alone can prove.
    """
    import asyncio
    import json as _json

    from rune.agent.litellm_adapter import _resolve_litellm_model, litellm

    if not names:
        return {}
    resolved, extra = _resolve_litellm_model(
        f"{provider}:{model}" if provider and model else (model or "")
    )
    prompt = _CLASSIFY_PROMPT.format(request=request[:4000],
                                     names=", ".join(sorted(names)))
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=resolved,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                **extra,
            ),
            timeout=_CLASSIFY_TIMEOUT_S,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        log.debug("artifact_role_error", error=str(exc)[:120])
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        raw = _json.loads(text[start:end + 1])
    except ValueError:
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items() if isinstance(raw, dict) else []:
        if isinstance(v, str) and v.strip().lower() in ("input", "output"):
            out[_key(k)] = v.strip().lower()
    return out


def circumvented_note(name: str) -> str:
    return (
        f"REVERTED: {name} was created after that write was refused, so it "
        f"has been removed again. The file genuinely does not exist. Say so "
        f"and stop instead of routing around the refusal."
    )


def phantom_write_error(name: str) -> str:
    return (
        f"BLOCKED: {name} was referred to as an existing file, but every "
        f"attempt to read it failed. Creating it here would replace the "
        f"user's content with invented content. Report that {name} is "
        f"missing and stop; do not reconstruct it."
    )


def unresolved_stop_note(missing: list[str]) -> str:
    listed = ", ".join(missing)
    return (
        f"These files were part of the request but could not be read: "
        f"{listed}. Do not describe the task as done. State plainly which "
        f"inputs are missing and what could not be determined without them."
    )


def path_exists(path: str) -> bool:
    try:
        return Path(path).expanduser().exists()
    except OSError:
        return False
