"""Resolve explicit shell write targets for the shared file policy.

This inspects shell syntax, not the effects of arbitrary programs. It adds
checks to Guardian; it does not provide process isolation.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from rune.safety.shell_ast import _parser


@dataclass
class WriteTargets:
    paths: set[str] = field(default_factory=set)
    uncertain: bool = False


def is_raw_python(command: str) -> bool:
    """Recognize Python source only when it is not valid shell syntax."""
    import ast

    if len(command) > 128_000:
        return False
    try:
        ast.parse(command)
    except (SyntaxError, ValueError, RecursionError):
        return False
    parser = _parser()
    if parser is None:
        return False
    root = parser.parse(command.encode()).root_node
    raw_python = root.has_error
    # The parser accepts f(value), but Bash requires an operator before a subshell.
    pending = [root]
    while pending:
        node = pending.pop()
        # An unfinished heredoc is still shell input, even if Python can parse it.
        if node.type == "heredoc_start":
            return False
        if node.type == "command" and any(child.type == "subshell" for child in node.named_children):
            raw_python = True
        pending.extend(node.named_children)
    return raw_python


def shell_write_targets(command: str, cwd: str = "", *, home: str = "", depth: int = 0) -> WriteTargets:
    found = WriteTargets()
    parser = _parser()
    if parser is None or depth > 3:
        found.uncertain = True
        return found
    source = command.encode()
    tree = parser.parse(source)
    home = home or str(Path.home())

    def text(node) -> str:
        return source[node.start_byte:node.end_byte].decode()

    def literal(node) -> str | None:
        if node is None:
            return None
        if node.type == "raw_string":
            return text(node)[1:-1]
        if node.type in {"simple_expansion", "expansion"}:
            return home if text(node) in {"$HOME", "${HOME}"} else None
        if node.type in {"string", "concatenation", "command_name"}:
            parts = [literal(child) for child in node.named_children]
            return "".join(parts) if all(part is not None for part in parts) else None
        if node.type in {"word", "number", "string_content"}:
            raw = text(node)
            if node.type == "word":
                # Unquoted globbing and brace expansion can name several targets.
                escaped = False
                for char in raw:
                    if not escaped and char in "*?[{}":
                        return None
                    escaped = not escaped and char == "\\"
                if raw == "~" or raw.startswith("~/"):
                    raw = home + raw[1:]
            try:
                return shlex.split('"' + raw + '"' if node.type == "string_content" else raw)[0]
            except (ValueError, IndexError):
                return None
        return None

    def target(value: str | None, directories: set[str]) -> None:
        if value is None or not directories:
            found.uncertain = True
        elif value and value != "-":
            found.paths.update(str(Path(directory, value).resolve()) for directory in directories)

    def redirects(nodes, directories: set[str]) -> None:
        for node in nodes:
            if node.type != "file_redirect":
                continue
            operator = next((text(c) for c in node.children if not c.is_named), "")
            if operator not in {">", ">>", ">|", "&>", "&>>", "<>", ">&"}:
                continue
            value = literal(node.child_by_field_name("destination"))
            if operator == ">&" and value is not None and (value.isdigit() or value == "-"):
                continue
            # These redirect to a sink or an existing descriptor, not a file.
            if value == "/dev/null" or value in {"/dev/stdout", "/dev/stderr"}:
                continue
            target(value, directories)

    def visit(node, directories: set[str], extra=()) -> set[str]:
        if node.type in {"comment", "heredoc_body", "heredoc_redirect", "function_definition"}:
            return directories
        if node.type == "redirected_statement":
            body = node.child_by_field_name("body")
            redirs = tuple(c for c in node.named_children if c != body) + tuple(extra)
            if body is None:
                redirects(redirs, directories)
                return directories
            return visit(body, directories, redirs)
        if node.type in {"program", "list"}:
            children = [c for c in node.named_children if c.type != "comment"]
            for index, child in enumerate(children):
                before = directories
                directories = visit(child, directories, extra if index == len(children) - 1 else ())
                # A failed cd followed by ';' or '||' leaves the old cwd in use.
                following = children[index + 1] if index + 1 < len(children) else None
                between = source[child.end_byte:following.start_byte].strip() if following else b""
                if between != b"&&":
                    directories = directories | before
            return directories
        if node.type == "command":
            redirects(extra, directories)
            name = literal(node.child_by_field_name("name"))
            args = [literal(c) for c in node.children_by_field_name("argument")]
            name = os.path.basename(name or "")
            while name in {"command", "exec", "nohup", "env"} and args:
                if name == "env":
                    while args and args[0] is not None and "=" in args[0] and not args[0].startswith("-"):
                        # Environment assignments may change how paths expand.
                        found.uncertain = True
                        args.pop(0)
                if not args or args[0] is None or args[0].startswith("-"):
                    found.uncertain = True
                    return directories
                name = os.path.basename(args.pop(0))
            if name == "cd":
                value = args[-1] if args else home
                if value is None or value == "-":
                    found.uncertain = True
                    return set()
                return {str(Path(directory, value).resolve()) for directory in directories}
            script_flag = next((i for i, arg in enumerate(args)
                                if arg is not None and arg.startswith("-")
                                and not arg.startswith("--") and "c" in arg), None)
            if name in {"bash", "sh", "zsh", "dash", "ksh"} and script_flag is not None:
                index = script_flag + 1
                script = args[index] if index < len(args) else None
                if script is None or not directories:
                    found.uncertain = True
                else:
                    for directory in directories:
                        nested = shell_write_targets(script, directory, home=home, depth=depth + 1)
                        found.paths.update(nested.paths)
                        found.uncertain |= nested.uncertain
            elif name in {"tee", "touch", "mkdir", "rm", "rmdir", "truncate"}:
                options = True
                for arg in args:
                    if options and arg == "--":
                        options = False
                    elif options and arg is not None and arg.startswith("-"):
                        continue
                    else:
                        target(arg, directories)
            elif name in {"cp", "mv", "install", "ln"}:
                operands = []
                destination = None
                options = True
                index = 0
                while index < len(args):
                    arg = args[index]
                    index += 1
                    if options and arg == "--":
                        options = False
                    elif options and arg in {"-t", "--target-directory"}:
                        destination = args[index] if index < len(args) else None
                        index += 1
                        if destination is None:
                            found.uncertain = True
                    elif options and arg is not None and arg.startswith("--target-directory="):
                        destination = arg.split("=", 1)[1]
                    elif options and arg is not None and arg.startswith("-"):
                        continue
                    else:
                        operands.append(arg)
                if destination is None and operands:
                    destination = operands.pop()
                target(destination, directories)
                for operand in operands:
                    if operand is None:
                        found.uncertain = True
                        continue
                    if name == "mv":
                        target(operand, directories)
                    # Copying into a directory writes beneath that directory.
                    if destination is not None:
                        target(os.path.join(destination, os.path.basename(operand)), directories)
            return directories
        redirects(extra, directories)
        inner = directories
        for child in node.named_children:
            inner = visit(child, inner)
        # Changes in a subshell or pipeline do not change the parent cwd.
        if node.type in {"subshell", "pipeline"}:
            return directories
        return directories | inner

    found.uncertain = tree.root_node.has_error
    visit(tree.root_node, {str(Path(cwd or Path.cwd()).expanduser().resolve())})
    return found
