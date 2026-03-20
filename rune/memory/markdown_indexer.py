"""Markdown-to-vector indexing pipeline for RUNE memory.

Chunks markdown files, hashes content for incremental updates,
embeds changed chunks, and maintains the FAISS index.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from rune.memory.markdown_store import (
    memory_dir,
    parse_daily_log,
    parse_learned_md,
    parse_memory_md,
    parse_rules_md,
    parse_user_profile,
)
from rune.memory.state import load_index_state, save_index_state
from rune.memory.types import VectorMetadata
from rune.memory.vector import VectorStore, get_vector_store
from rune.utils.logger import get_logger

log = get_logger(__name__)

_LEARNED_STRIP_RE = re.compile(r"^\[[\w]+\]\s*")
_CONFIDENCE_STRIP_RE = re.compile(r"\s*\([\d.]+\)\s*$")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_embeddable_learned(raw_value: str) -> str:
    """Strip [category] prefix and (confidence) suffix for cleaner embedding."""
    text = _LEARNED_STRIP_RE.sub("", raw_value)
    text = _CONFIDENCE_STRIP_RE.sub("", text)
    return text.strip()


# Chunking

def chunk_memory_md(path: Path | None = None) -> list[dict[str, Any]]:
    """Chunk MEMORY.md: one chunk per bullet line."""
    if path is None:
        path = memory_dir() / "MEMORY.md"
    sections = parse_memory_md(path)
    chunks = []
    for section, lines in sections.items():
        for line in lines:
            chunk_id = f"MEMORY.md::{section}::{line.split(':')[0].strip() if ':' in line else line[:30]}"
            embeddable = line
            chunks.append({
                "id": chunk_id,
                "text": embeddable,
                "hash": _sha256(embeddable),
                "type": "md_fact",
                "timestamp": "",
                "category": section.lower(),
                "source_file": "MEMORY.md",
            })
    return chunks


def chunk_learned_md(path: Path | None = None) -> list[dict[str, Any]]:
    """Chunk learned.md: one chunk per fact line."""
    if path is None:
        path = memory_dir() / "learned.md"
    facts = parse_learned_md(path)
    chunks = []
    for fact in facts:
        embeddable = f"{fact['key']}: {fact['value']}"
        chunk_id = f"learned.md::{fact['key']}"
        chunks.append({
            "id": chunk_id,
            "text": embeddable,
            "hash": _sha256(embeddable),
            "type": "md_fact",
            "timestamp": "",
            "category": fact["category"],
            "source_file": "learned.md",
        })
    return chunks


def chunk_daily_file(path: Path) -> list[dict[str, Any]]:
    """Chunk a daily log: one chunk per H2 section (task entry)."""
    entries = parse_daily_log(path)
    chunks = []
    for entry in entries:
        text_parts = [f"{entry['title']}."]
        for action in entry["actions"][:3]:
            text_parts.append(action)
        embeddable = " ".join(text_parts)

        date = path.stem
        chunk_id = f"daily/{path.name}::{entry['time']}"
        chunks.append({
            "id": chunk_id,
            "text": embeddable,
            "hash": _sha256(embeddable),
            "type": "md_daily",
            "timestamp": f"{date}T{entry['time']}:00Z" if entry["time"] else "",
            "category": "daily",
            "source_file": f"daily/{path.name}",
        })
    return chunks


def chunk_rules_md(path: Path) -> list[dict[str, Any]]:
    """Chunk rules.md: one chunk per H2 section."""
    rules = parse_rules_md(path)
    chunks = []
    for rule in rules:
        embeddable = f"{rule.get('name', '')}: {rule.get('pattern', '')}. {rule.get('reason', '')}"
        chunk_id = f"rules.md::{rule.get('name', 'unnamed')}"
        chunks.append({
            "id": chunk_id,
            "text": embeddable.strip(),
            "hash": _sha256(embeddable.strip()),
            "type": "md_rule",
            "timestamp": "",
            "category": "safety",
            "source_file": "rules.md",
        })
    return chunks


def chunk_user_profile(path: Path | None = None) -> list[dict[str, Any]]:
    """Chunk user-profile.md: one chunk per H1 section."""
    if path is None:
        path = memory_dir() / "user-profile.md"
    sections = parse_user_profile(path)
    chunks = []
    for section, lines in sections.items():
        if not lines:
            continue
        embeddable = f"{section}: " + ", ".join(lines[:10])
        chunk_id = f"user-profile.md::{section}"
        chunks.append({
            "id": chunk_id,
            "text": embeddable,
            "hash": _sha256(embeddable),
            "type": "md_profile",
            "timestamp": "",
            "category": section.lower(),
            "source_file": "user-profile.md",
        })
    return chunks


# Indexing

def collect_all_chunks() -> list[dict[str, Any]]:
    """Collect chunks from all markdown files in ~/.rune/memory/."""
    d = memory_dir()
    all_chunks: list[dict[str, Any]] = []

    # MEMORY.md
    mem_path = d / "MEMORY.md"
    if mem_path.exists():
        all_chunks.extend(chunk_memory_md(mem_path))

    # learned.md
    learned_path = d / "learned.md"
    if learned_path.exists():
        all_chunks.extend(chunk_learned_md(learned_path))

    # daily/*.md
    daily_dir = d / "daily"
    if daily_dir.exists():
        for daily_file in sorted(daily_dir.glob("*.md")):
            all_chunks.extend(chunk_daily_file(daily_file))

    # user-profile.md
    profile_path = d / "user-profile.md"
    if profile_path.exists():
        all_chunks.extend(chunk_user_profile(profile_path))

    return all_chunks


async def incremental_reindex(
    vectors: VectorStore | None = None,
) -> dict[str, int]:
    """Reindex only chunks whose content has changed.

    Compares SHA-256 hashes against index-state.json.
    Returns stats: {added, updated, removed, unchanged}.
    """
    if vectors is None:
        vectors = get_vector_store()

    state = load_index_state()
    chunks_state: dict[str, Any] = state.get("chunks", {})

    current_chunks = collect_all_chunks()
    current_ids = {c["id"] for c in current_chunks}

    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

    # Find new and changed chunks
    to_embed: list[dict[str, Any]] = []
    for chunk in current_chunks:
        old = chunks_state.get(chunk["id"])
        if old and old.get("hash") == chunk["hash"]:
            stats["unchanged"] += 1
            continue
        if old:
            stats["updated"] += 1
        else:
            stats["added"] += 1
        to_embed.append(chunk)

    # Find orphaned chunks (deleted from markdown)
    orphaned = set(chunks_state.keys()) - current_ids
    stats["removed"] = len(orphaned)

    # Embed and add new/changed chunks
    if to_embed:
        try:
            from rune.llm.local_embedding import get_embedding_provider
            provider = get_embedding_provider()

            for chunk in to_embed:
                embedding = await provider.embed_single(chunk["text"])
                meta = VectorMetadata(
                    type=chunk["type"],
                    id=chunk["id"],
                    timestamp=chunk.get("timestamp", ""),
                    summary=chunk["text"][:200],
                    category=chunk.get("category", ""),
                )
                vectors.add(embedding, meta)

                chunks_state[chunk["id"]] = {
                    "hash": chunk["hash"],
                    "indexed_at": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                }
        except Exception as exc:
            log.warning("indexer_embed_failed", error=str(exc), count=len(to_embed))

    # Remove orphaned entries from state (FAISS tombstoning handled by vector.py)
    for oid in orphaned:
        chunks_state.pop(oid, None)

    state["chunks"] = chunks_state
    save_index_state(state)

    if stats["added"] or stats["updated"] or stats["removed"]:
        log.info("incremental_reindex_done", **stats)

    return stats


async def full_rebuild(vectors: VectorStore | None = None) -> dict[str, int]:
    """Full reindex: clear state and re-embed everything."""
    save_index_state({"chunks": {}})

    if vectors is None:
        vectors = get_vector_store()

    return await incremental_reindex(vectors)
