"""Identify check invocations whose exit status the shell preserves."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath

from rune.agent.bash_parsing import is_test_command, is_verification_command, strip_runner_prefix


@dataclass(frozen=True, slots=True)
class CheckCommand:
    command: str
    cwd: str
    kind: str
    reliable_exit: bool

    @property
    def key(self) -> tuple[str, str]:
        words = shlex.split(self.command)
        if words and re.fullmatch(r"python(?:[23](?:\.\d+)?)?", PurePosixPath(words[0]).name):
            # -B changes bytecode writes, not which tests run.
            while words[1:2] == ["-B"]:
                del words[1]
        return self.cwd, shlex.join(words)


def pytest_scope(command: str) -> tuple[str, ...] | None:
    """Match pytest arguments across direct and Python-module invocations."""
    try:
        words = strip_runner_prefix(tuple(shlex.split(command)))
    except ValueError:
        return None
    if not words:
        return None
    head = PurePosixPath(words[0]).name
    if head in {"pytest", "py.test"}:
        return words[1:]
    if head in {"python", "python3", "python2"} and words[1:3] == ("-m", "pytest"):
        return words[3:]
    return None


@lru_cache(maxsize=1)
def _parser():
    import tree_sitter_bash
    from tree_sitter import Language, Parser

    return Parser(Language(tree_sitter_bash.language()))


def check_commands(command: str, cwd: str = "") -> list[CheckCommand]:
    """Match checks by command arguments and working directory.

    Trust the exit status only for simple checks joined by &&.
    Piped or compound commands need a direct rerun.
    """
    source = command.encode()
    root = _parser().parse(source).root_node
    found: list[tuple[str, str, str]] = []
    reliable = not root.has_error
    directory = os.path.normpath(cwd or ".")

    def text(node) -> str:
        return source[node.start_byte:node.end_byte].decode()

    def visit(node) -> None:
        nonlocal directory, reliable
        if node.type in {"comment", "function_definition"}:
            return
        if node.type == "program":
            statements = [c for c in node.named_children if c.type != "comment"]
            if len(statements) != 1 or any(c.type == "&" for c in node.children):
                reliable = False
        elif node.type == "list":
            if any(c.type != "&&" for c in node.children if not c.is_named):
                reliable = False
        elif node.type == "redirected_statement":
            visit(node.child_by_field_name("body"))
            return
        elif node.type == "command":
            try:
                words = shlex.split(text(node))
            except ValueError:
                reliable = False
                return
            if not words:
                return
            if any(word in {"--help", "--version", "--collect-only", "--collectonly"} for word in words):
                return
            if words[0] == "cd" and len(words) == 2:
                directory = os.path.normpath(os.path.join(directory, words[1]))
                return
            normalized = shlex.join(words)
            if is_verification_command(normalized):
                kind = "test" if is_test_command(normalized) else "check"
                found.append((normalized, directory, kind))
            else:
                # An unrelated command can alter cwd, shell options or code.
                reliable = False
            return
        else:
            reliable = False
        for child in node.named_children:
            visit(child)

    visit(root)
    return [CheckCommand(cmd, directory, kind, reliable) for cmd, directory, kind in found]
