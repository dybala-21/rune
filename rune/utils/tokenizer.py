"""Token counting utilities using tiktoken.

Ported from src/utils/tokenizer.ts - Rust BPE engine, 3-6x faster than JS.
"""

from __future__ import annotations

import functools

import tiktoken


@functools.lru_cache(maxsize=8)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """Get or cache a tiktoken encoding for the given model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (GPT-4 / Claude compatible)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in *text* for *model*."""
    enc = _get_encoding(model)
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate *text* to at most *max_tokens*."""
    enc = _get_encoding(model)
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def truncate_to_last_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Return the *last* *max_tokens* tokens of *text*."""
    enc = _get_encoding(model)
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[-max_tokens:])


def encode(text: str, model: str = "gpt-4o") -> list[int]:
    """Encode *text* into token IDs."""
    enc = _get_encoding(model)
    return enc.encode(text, disallowed_special=())


def decode(tokens: list[int], model: str = "gpt-4o") -> str:
    """Decode token IDs back to text."""
    enc = _get_encoding(model)
    return enc.decode(tokens)
