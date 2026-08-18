"""What the shell would actually run, read off a parse rather than a pattern.

Every hole found in the command classifier was structural. `rm -f` was read
as recursive because a flag pattern let the `r` be optional. A backslash and
a newline hid `rm -rf /etc` because whitespace was collapsed before the
continuation was removed. `rm a.txt && rm -rf /etc` passed because the
search stopped at the first deletion. None of those are hard questions for
something that knows where one command ends and the next begins.

So the tree is asked, and it is asked for one thing only: **is there a
deletion here worse than the patterns already found?** The answer can raise
the verdict and can never lower it. That is the whole safety argument. A
parser that misreads a command cannot open a hole, because nothing it says
is believed when it says "safer" — it is not consulted for that. What it
can do is over-block, which is measurable, and is measured.

Two things it deliberately does not do. It does not replace normalisation:
`${IFS}` and brace expansion happen when the shell runs, not when it parses,
so `rm${IFS}-rf${IFS}/etc` is one word to any grammar and stays the
normaliser's job. And it does not try to be the shell — where the grammar
reports an error, this contributes nothing and the existing checks stand
alone, which is the same answer they would have given anyway.

Precedent for the shape: layered command defences put a fast structural
filter first and let later stages escalate but never override an earlier
block (CASCADE, arXiv 2604.17125).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_SHELL_AST"
# Commands whose argument is itself a command to run.
_SHELL_RUNNERS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
# Wrappers that pass the rest of the line through untouched.
_TRANSPARENT = frozenset({"sudo", "env", "nohup", "time", "timeout", "doas",
                          "command", "exec", "xargs", "nice", "ionice"})
_MAX_DEPTH = 3          # how far to follow `bash -c "…"` chains


def shell_ast_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


@dataclass(frozen=True)
class SimpleCommand:
    """One command the shell would run, with its own working directory.

    `cwd` is whatever a `cd` earlier in the same chain set, which is how
    `cd / && rm -rf etc` gets read as a deletion of /etc rather than of a
    relative path that happens to be named etc.
    """
    name: str
    args: tuple[str, ...]
    cwd: str


@dataclass(frozen=True)
class ShellReading:
    commands: tuple[SimpleCommand, ...]
    understood: bool


@lru_cache(maxsize=1)
def _parser():
    """The bash parser, or None when it is not installed.

    Absence is not an error: this layer only ever adds a reason to refuse,
    so a missing parser leaves the existing checks exactly as they were.
    """
    try:
        import tree_sitter_bash
        from tree_sitter import Language, Parser
    except ImportError:                      # pragma: no cover - env dependent
        log.debug("shell_ast_parser_unavailable")
        return None
    return Parser(Language(tree_sitter_bash.language()))


def _unquote(word: str) -> str:
    """Take off matching outer quotes, one pair at a time.

    Stripping quote characters from both ends independently breaks the very
    strings it is meant to clean: `"rm -rf 'build'"` ends with `'"`, and
    removing both leaves an unbalanced quote that no parse survives.
    """
    while len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
        word = word[1:-1]
    return word


def read(command: str) -> ShellReading:
    """The commands a line would run, in order, with cd applied."""
    parser = _parser()
    if parser is None or not command.strip():
        return ShellReading((), False)
    src = command.encode()
    try:
        tree = parser.parse(src)
    except Exception:                        # pragma: no cover - defensive
        log.debug("shell_ast_parse_failed")
        return ShellReading((), False)

    out: list[SimpleCommand] = []
    cwd = "."

    def text(node) -> str:
        return src[node.start_byte:node.end_byte].decode(errors="replace")

    def visit(node, depth: int) -> None:
        nonlocal cwd
        if node.type == "command":
            name, args = "", []
            for child in node.children:
                if child.type == "command_name":
                    name = _unquote(text(child))
                elif child.type in ("word", "string", "raw_string", "number",
                                    "concatenation", "simple_expansion",
                                    "expansion"):
                    args.append(_unquote(text(child)))
            if not name:
                return
            if name == "cd" and args:
                cwd = args[0]
                return
            out.append(SimpleCommand(name, tuple(args), cwd))
            return
        for child in node.children:
            visit(child, depth)

    visit(tree.root_node, 0)
    return ShellReading(tuple(out), not tree.root_node.has_error)


def worst_deletion(command: str,
                   depth: int = 0) -> Literal["critical", "high"] | None:
    """The worst deletion the parse can see, or None.

    Only deletions, and only ever as an escalation. Wrappers are stepped
    through so `sudo rm -rf /etc` is read as the `rm` it is, and a `cd`
    earlier in the chain decides what a relative target means.
    """
    if not shell_ast_enabled():
        return None
    # Nothing here reports anything but `rm`, so a line without one has
    # nothing to find and does not need to be normalised or parsed. Only
    # ANSI-C decoding can produce those two letters where they were not
    # written, which is what the second token looks for. `ls -la` is the
    # common case and it was paying 68us to be told it deletes nothing.
    if "rm" not in command and "$'" not in command:
        return None
    from rune.safety.analyzer import classify_rm_target, normalize_command

    # The normalised text, not the raw line. Expansions are a runtime thing
    # and no grammar resolves them, so `rm${IFS}-rf${IFS}build` is one word
    # to the parser and `$'\x72\x6d'` is not a command name at all. Reading
    # what the normaliser produced puts both back within reach.
    worst: Literal["critical", "high"] | None = None
    here = os.getcwd()
    for cmd in read(normalize_command(command)).commands:
        name, args = cmd.name, list(cmd.args)
        # Step through wrappers to the command they carry. `timeout 5 rm …`
        # puts an operand of its own in the way, so anything that is not a
        # plausible command word is stepped over too.
        while name in _TRANSPARENT and args:
            name, args = args[0], args[1:]
            while name and (name.startswith("-") or name.isdigit()) and args:
                name, args = args[0], args[1:]
        base = name.rsplit("/", 1)[-1]
        # `bash -c "…"` carries a command in its argument, and a wrapper in
        # front of it is why this is asked here rather than at parse time:
        # in `env bash -c "rm -rf /etc"` the command name is `env`.
        if base in _SHELL_RUNNERS and depth < _MAX_DEPTH:
            for i, a in enumerate(args):
                if a == "-c" and i + 1 < len(args):
                    inner = worst_deletion(args[i + 1], depth + 1)
                    if inner == "critical":
                        return "critical"
                    if inner == "high":
                        worst = "high"
            continue
        if base != "rm":
            continue
        prefix = "" if cmd.cwd in (".", "") else cmd.cwd.rstrip("/") + "/"
        # The targets are classified directly. Rebuilding `rm <flags>
        # <target>` and handing it back to the regex re-parsed the same
        # command once per file: forty object files cost 2.7ms of a path
        # that is supposed to stay under a millisecond. Duplicates are
        # dropped for the same reason — a sweep names the same directory
        # many times.
        flags = " ".join(a for a in args if a.startswith("-"))
        seen: set[str] = set()
        for arg in args:
            if arg.startswith("-"):
                continue
            target = arg if arg.startswith(("/", "~", "$")) else prefix + arg
            if target in seen:
                continue
            seen.add(target)
            risk = classify_rm_target(flags, target, here)
            if risk == "critical":
                return "critical"
            if risk == "high":
                worst = "high"
    return worst
