"""Fill in the vectors the active embedding model doesn't have yet.

Vectors are derived data, kept per model fingerprint. When the model changes (nomic to EmbeddingGemma,
or a new prompt version) the new model's index starts empty, and only memories written from then
on get vectors. This embeds the rest in the background: episodes from SQLite, the markdown facts,
then conversation turns. Until it is done, hybrid search still finds the rest by keyword.

It runs once per long-lived process (web server, daemon), in small pieces so searches get the
embedding worker in between. It never starts a model download itself; it waits until the model is
on disk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# One embedding call: at most this many documents and characters (~2 s on a CPU, ~0.3 s on Apple
# GPUs), so a search waits at most one piece.
_PIECE_ITEMS = 8
_PIECE_CHARS = 2000
# Episodes to cover: the same window keyword search holds.
_EPISODES = 5000
# Save the vector index every this many pieces, so a stop loses little work.
_SAVE_EVERY = 25
# How often to look again while the model isn't on disk yet, or the worker is down.
_WAIT_SECONDS = 60.0


def _pieces[T](items: list[T], text: Callable[[T], str]) -> list[list[T]]:
    out: list[list[T]] = [[]]
    size = 0
    for item in items:
        if out[-1] and (size + len(text(item)) > _PIECE_CHARS or len(out[-1]) >= _PIECE_ITEMS):
            out.append([])
            size = 0
        out[-1].append(item)
        size += len(text(item))
    return [piece for piece in out if piece]


async def _episodes(provider: Any, vectors: Any) -> int:
    from rune.memory.store import get_memory_store
    from rune.memory.types import VectorMetadata

    loop = asyncio.get_running_loop()
    store = get_memory_store()
    episodes = [ep for ep in await loop.run_in_executor(None, store.get_recent_episodes, _EPISODES)
                if ep.task_summary]
    if not episodes:
        return 0
    # The first vector names the active model, and so which index to compare against.
    probe = await provider.embed([episodes[0].task_summary], kind="document")
    vectors.select_model(probe[0].fingerprint, len(probe[0]))
    have = vectors.indexed_ids({"episode"})
    todo = [ep for ep in episodes if ep.id not in have]
    done = 0
    try:
        for n, piece in enumerate(_pieces(todo, lambda ep: ep.task_summary)):
            found = await provider.embed([ep.task_summary for ep in piece], kind="document")
            # Searches and saves ran meanwhile: skip what they indexed, and stop if the model changed.
            if found[0].fingerprint != probe[0].fingerprint:
                raise RuntimeError("Embedding model changed during reindex")
            have = vectors.indexed_ids({"episode"})
            fresh = [(v, ep) for v, ep in zip(found, piece, strict=True) if ep.id not in have]
            if fresh:
                vectors.add_batch([v for v, _ in fresh], [
                    VectorMetadata(type="episode", id=ep.id, timestamp=ep.timestamp, summary=ep.task_summary)
                    for _, ep in fresh])
                done += len(fresh)
            if n % _SAVE_EVERY == _SAVE_EVERY - 1:
                vectors.save()
    finally:
        if done:
            vectors.save()
    return done


async def _turns(provider: Any) -> int:
    from rune.conversation.store import ConversationStore
    from rune.utils.paths import conversations_db_path

    store = ConversationStore(conversations_db_path())
    turns = store.fetch_searchable_turns()
    if not turns:
        return 0
    probe = await provider.embed([turns[0]["content"][:500]], kind="document")
    fingerprint = probe[0].fingerprint
    contents: dict[str, str] = {}
    for turn in turns:
        contents.setdefault(turn["content_hash"], turn["content"][:500])
    cached = store.get_cached_embeddings(list(contents), fingerprint=fingerprint)
    todo = [(h, c) for h, c in contents.items() if h not in cached]
    done = 0
    for piece in _pieces(todo, lambda item: item[1]):
        found = await provider.embed([c for _, c in piece], kind="document")
        if any(v.fingerprint != fingerprint for v in found):
            raise RuntimeError("Embedding model changed during reindex")
        store.cache_embeddings({h: v for (h, _), v in zip(piece, found, strict=True)}, fingerprint=fingerprint)
        done += len(piece)
    return done


async def fill_missing_vectors() -> dict[str, int]:
    """Embed every episode, markdown fact and conversation turn the active model has no vector for."""
    from rune.llm.local_embedding import get_embedding_provider
    from rune.memory.markdown_indexer import incremental_reindex
    from rune.memory.vector import get_vector_store

    provider = get_embedding_provider()
    vectors = get_vector_store()
    stats = {"episodes": await _episodes(provider, vectors)}
    markdown = await incremental_reindex(vectors)
    stats["markdown"] = markdown["added"] + markdown["updated"]
    stats["turns"] = await _turns(provider)
    return stats


def _model_on_disk() -> bool:
    from rune.llm.local_embedding import model_cached

    return model_cached()


async def keep_vectors_filled(
    *, fill: Callable[[], Awaitable[dict[str, int]]] = fill_missing_vectors,
    ready: Callable[[], bool] = _model_on_disk, wait: float = _WAIT_SECONDS,
) -> None:
    """Run `fill` once the model is on disk, retrying while the embedding worker can't answer."""
    import importlib.util

    from rune.llm.local_embedding import embeddings_enabled

    # Without these extras there are no vectors to fill, now or later in this process.
    if not embeddings_enabled() or not all(importlib.util.find_spec(m) for m in ("numpy", "faiss")):
        return
    while True:
        # Waiting first leaves startup to the work a user is waiting on.
        await asyncio.sleep(wait)
        if ready():
            try:
                stats = await fill()
                log.info("vectors_filled", **stats)
                return
            except Exception as exc:
                log.warning("vector_fill_failed", error=f"{type(exc).__name__}: {exc}"[:200])
