"""Identify executable names for activity suggestions, without parsing shell programs."""

import re
import shlex
from pathlib import PurePosixPath


def command_name(command: str) -> str:
    try:
        words = shlex.split(command)
    except ValueError:
        return ""
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
        words.pop(0)
    if words[:1] == ["env"]:
        words.pop(0)
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words.pop(0)
    if words[:2] in (["uv", "run"], ["poetry", "run"]):
        words = words[2:]
    if words[:1] in (["npx"], ["exec"]):
        words = words[1:]
    if not words:
        return ""
    name = PurePosixPath(words[0]).name
    if re.fullmatch(r"python(?:[23](?:\.\d+)?)?", name):
        args = words[1:]
        while args and args[0] in {"-B", "-I", "-u", "-E", "-s", "-S"}:
            args.pop(0)
        name = args[1] if len(args) > 1 and args[0] == "-m" else ""
    if name in {"bash", "sh", "zsh", "cd", "echo", "printf", "export", "source"}:
        return ""
    return name if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) else ""
