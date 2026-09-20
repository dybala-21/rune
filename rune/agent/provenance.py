"""Track requested files against the run's read and write records.

The ledger detects attempts to replace a missing input with a generated
file and reports inputs that remain unresolved at completion.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
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
_WEB_LINK_RE = re.compile(r"\[[^\]\n]*\]\(\s*https?://[^)\n]*\)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>`\"']+", re.IGNORECASE)
_FRAGMENT_RE = re.compile(r"(?<![\w/\\])#[^\s<>`\"')\]]+")


def _local_path_tokens(text: str) -> Iterator[str]:
    # A page or download URL is not evidence that a local file should exist.
    text = _URL_RE.sub(" ", _WEB_LINK_RE.sub(" ", text or ""))
    text = _FRAGMENT_RE.sub(" ", text)
    return (match.group(0) for match in _PATH_RE.finditer(text))

# Extensions that name a document/artifact rather than an inline example.
_SKIP_SUFFIXES = frozenset({
    ".com", ".org", ".net", ".io", ".dev", ".ai", ".co", ".kr", ".jp",
})

_READ_TOOLS = frozenset({"file_read", "document_read"})
_WRITE_TOOLS = frozenset({"file_write", "file_edit"})


def provenance_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def referenced_paths(text: str) -> set[str]:
    """File names the request talks about, keyed by base name."""
    out: set[str] = set()
    for token in _local_path_tokens(text):
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
    unreadable: set[str] = field(default_factory=set)
    # The tree the request is about. A request names a bare file, so the
    # ledger keys on bare names — which means a file of the same name
    # somewhere else would otherwise answer for it. Empty disables the
    # check, for callers with no workspace to speak of.
    root: str = ""
    observed_paths: dict[str, str] = field(default_factory=dict)
    requested_paths: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def for_request(cls, request: str, root: str = "") -> ArtifactLedger:
        ledger = cls(referenced=referenced_paths(request), root=root)
        for path in _local_path_tokens(request):
            name = _key(path)
            if name in ledger.referenced:
                ledger.requested_paths.setdefault(name, set()).add(path)
        return ledger

    def resolve_path(self, path: str) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute() and self.root:
            target = Path(self.root) / target
        return target.resolve()

    def _within_root(self, path: str) -> bool:
        if not self.root:
            return True
        try:
            return self.resolve_path(path).is_relative_to(Path(self.root).resolve())
        except (OSError, ValueError):
            return False

    def record_read(self, path: str, ok: bool, *, missing: bool = True) -> None:
        if not self._within_root(path):
            return
        k = _key(path)
        declared = self.requested_paths.get(k, set())
        explicit = [p for p in declared if Path(p).parent != Path(".")]
        if explicit and not any(self.resolve_path(p) == self.resolve_path(path) for p in explicit):
            return
        self.observed_paths[k] = str(self.resolve_path(path))
        self.looked_up.add(k)
        if ok:
            self.read_ok.add(k)
            self.known_absent.discard(k)
            self.unreadable.discard(k)
        elif missing:
            self.known_absent.add(k)
        else:
            self.unreadable.add(k)

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
        if role in {"input", "preserve"}:
            return True
        return k in self.looked_up

    def unresolved(self) -> list[str]:
        """Required inputs with evidence of absence or a failed read.

        A shell command may inspect a file or hash it without using a read
        tool. Mentioning that path is not evidence that the file is missing.
        """
        return sorted(
            k for k in self.referenced
            if k in self.looked_up and k not in self.read_ok
            and self.roles.get(k, "input") == "input"
            and (k in self.known_absent or k in self.unreadable
                 or not any(self.resolve_path(p).exists()
                            for p in self.requested_paths.get(k, {k})))
        )


_CLASSIFY_TIMEOUT_S = 20.0

_CLASSIFY_PROMPT = """\
For each file name listed, decide from the request whether it is an INPUT
(the request assumes it already exists and its contents are to be used) or
an OUTPUT (the request asks for it to be produced), or PRESERVE (mentioned
only to keep it unchanged, without a task that needs its contents).
If the request both uses a file's contents and preserves it, choose INPUT.
Checking a file's hash to prove it was not changed does not require reading
its contents into the answer.

Request:
---
{request}
---

File names: {names}

Reply with JSON only: {{"<name>": "input"|"output"|"preserve", ...}}
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
    from rune.llm.reasoning import reasoning_control
    from rune.llm.request_params import compatible_completion

    if not names:
        return {}
    resolved, extra = _resolve_litellm_model(
        f"{provider}:{model}" if provider and model else (model or "")
    )
    prompt = _CLASSIFY_PROMPT.format(request=request[:4000],
                                     names=", ".join(sorted(names)))
    efforts = reasoning_control(resolved).efforts
    effort = next((level for level in ("none", "minimal", "low") if level in efforts), None)
    try:
        resp = await asyncio.wait_for(
            compatible_completion(litellm.acompletion, litellm.BadRequestError, {
                "model": resolved,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1200,
                "reasoning_effort": effort,
                "timeout": _CLASSIFY_TIMEOUT_S,
                **extra,
            }),
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
        if isinstance(v, str) and v.strip().lower() in ("input", "output", "preserve"):
            if _key(k) in names:
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


def unresolved_stop_note(missing: list[str], request: str = "") -> str:
    listed = ", ".join(missing)
    return (
        f"Required inputs could not be accessed: {listed}. Check the tool "
        f"evidence before summarizing. Distinguish a missing file from an "
        f"existing file whose read failed. State what remains unresolved "
        f"without discarding work that was independently verified. This "
        f"is internal check feedback, not a new user request. Use the "
        f"language of the original request quoted below for the final answer.\n\n"
        f"Original user request:\n---\n{request[:4000]}\n---"
    )


def path_exists(path: str, root: str = "") -> bool:
    try:
        target = Path(path).expanduser()
        if not target.is_absolute() and root:
            target = Path(root) / target
        return target.exists()
    except OSError:
        return False
