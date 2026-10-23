"""Recognize shell commands that install packages and whether they change anything outside the workspace.

Installs outside the workspace outlive the task and run agent-chosen code, so they need approval.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PackageInstall:
    tool: str
    outside_workspace: bool
    reason: str


_SEGMENT = re.compile(r"&&|\|\||[;|&\n]")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
# Package managers whose install target is always outside the project.
_SYSTEM_MANAGERS = {
    "brew": {"install", "reinstall", "upgrade", "tap"},
    "apt": {"install", "upgrade"},
    "apt-get": {"install", "upgrade"},
    "dnf": {"install", "upgrade"},
    "yum": {"install", "update"},
    "port": {"install", "upgrade"},
    "gem": {"install", "update"},
    "cargo": {"install"},
    "go": {"install"},
    "pipx": {"install", "upgrade", "reinstall"},
    "conda": {"install", "update"},
    "mamba": {"install", "update"},
}
# Project-scoped managers: they resolve the project's own environment.
_PROJECT_MANAGERS = {
    "uv": {"add", "sync"},
    "poetry": {"add", "install"},
    "pipenv": {"install"},
    "pdm": {"add", "install"},
}


def _segments(command: str) -> list[list[str]]:
    words = []
    for part in _SEGMENT.split(command):
        try:
            tokens = shlex.split(part)
        except ValueError:
            continue
        while tokens and _ASSIGNMENT.fullmatch(tokens[0]):
            tokens.pop(0)
        if tokens[:1] in (["sudo"], ["env"]):
            tokens.pop(0)
        if tokens:
            words.append(tokens)
    return words


def _inside(path: str, root: str) -> bool:
    if not root:
        return False
    resolved = os.path.normpath(os.path.join(root, os.path.expanduser(path)))
    return resolved == root or resolved.startswith(root.rstrip("/") + "/")


def _pip(words: list[str], root: str, venv_active: bool = False) -> PackageInstall | None:
    """pip / pip3 / python -m pip / uv pip install, judged by which environment receives the packages."""
    program = os.path.basename(words[0])
    if program == "uv":
        if words[1:3] != ["pip", "install"]:
            return None
        if "--user" in words:
            return PackageInstall("uv pip", True, "Installs Python packages into the user site-packages, outside the workspace.")
        outside = "--system" in words
        return PackageInstall("uv pip", outside, "Installs Python packages into the system interpreter." if outside else "")
    if re.fullmatch(r"pip\d*(\.\d+)?", program):
        args, interpreter = words[1:], words[0]
    elif re.fullmatch(r"python\d*(\.\d+)?", program) and words[1:3] == ["-m", "pip"]:
        args, interpreter = words[3:], words[0]
    else:
        return None
    if not args or args[0] != "install":
        return None
    if "--user" in args:
        return PackageInstall("pip", True, "Installs Python packages into the user site-packages, outside the workspace.")
    for flag in ("--target", "-t", "--prefix", "--root"):
        if flag in args and args.index(flag) + 1 < len(args):
            if not _inside(args[args.index(flag) + 1], root):
                return PackageInstall("pip", True, "Installs Python packages to a directory outside the workspace.")
            return PackageInstall("pip", False, "")
    # A bare `pip`/`python3` resolves through PATH to an unknown interpreter unless a workspace venv was activated.
    if ("/" not in interpreter and not venv_active) or ("/" in interpreter and not _inside(interpreter, root)):
        return PackageInstall("pip", True,
                              "Installs Python packages with an interpreter outside the workspace; use the project's "
                              "virtual environment (for example .venv/bin/python -m pip) to keep it local.")
    return PackageInstall("pip", False, "")


def install_scope(command: str, cwd: str) -> PackageInstall | None:
    """The first package install in a shell command, or None when it installs nothing."""
    root = os.path.normpath(cwd) if cwd else ""
    venv_active = False
    for words in _segments(command):
        program = os.path.basename(words[0])
        sub = next((word for word in words[1:] if not word.startswith("-")), "")
        if program in {"source", "."} and len(words) > 1 and words[1].endswith("/bin/activate"):
            venv_active = _inside(words[1], root)
            continue
        if found := _pip(words, root, venv_active):
            return found
        if program in {"npm", "pnpm", "yarn", "bun"}:
            if program == "yarn" and sub == "global":
                return PackageInstall("yarn", True, "Installs packages globally, outside the workspace.")
            if sub in {"install", "i", "add"}:
                if {"-g", "--global"} & set(words):
                    return PackageInstall(program, True, "Installs packages globally, outside the workspace.")
                return PackageInstall(program, False, "")
        if program in _SYSTEM_MANAGERS and sub in _SYSTEM_MANAGERS[program]:
            return PackageInstall(program, True, f"`{program} {sub}` changes software outside the workspace.")
        if program in _PROJECT_MANAGERS and sub in _PROJECT_MANAGERS[program]:
            if program == "uv" and "--system" in words:
                return PackageInstall(program, True, "Installs Python packages into the system interpreter.")
            return PackageInstall(program, False, "")
    return None
