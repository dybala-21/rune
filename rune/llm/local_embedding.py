"""Embedding provider with a supervised native worker."""

from __future__ import annotations

import asyncio
import atexit
import functools
import hashlib
import importlib.util
import math
import os
import tempfile
import threading
import time
from array import array
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from filelock import FileLock

from rune.llm.embedding_models import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODELS,
    EmbeddingKind,
    EmbeddingModelConfig,
    EmbeddingVector,
    _models_dir,
)
from rune.llm.embedding_worker import serve
from rune.utils.env import env_flag
from rune.utils.logger import get_logger
from rune.utils.process_worker import ProcessWorker, WorkerUnavailable

log = get_logger(__name__)
__all__ = ["DEFAULT_MODEL", "EMBEDDING_DIM", "EMBEDDING_MODELS", "EmbeddingKind", "EmbeddingModelConfig",
           "LocalEmbeddingEngine", "LocalEmbeddingProvider", "get_embedding_engine",
           "get_embedding_provider", "dispose_embedding_engine", "embeddings_enabled", "embeddings_installed",
           "match_threshold", "model_cached", "warm_up_embedding"]

# Local embeddings run only when the user turns them on: the model is a download of a few hundred MB
# and a resident worker, which a user who never searches memory by meaning should not pay for.
LOCAL_EMBEDDINGS_ENV = "RUNE_LOCAL_EMBEDDINGS"

# Documents embedded again and again (every rule and skill on every lookup) are answered from
# here. 4,096 vectors of 768 float32 are 12 MB.
_DOCUMENT_CACHE_SIZE = 4096
# A download that goes this long without finishing is abandoned and retried later.
_DOWNLOAD_LIMIT_SECONDS = 1800
# Documents go to the worker in pieces of at most this many characters (or items), each with its
# own deadline: one 64-turn batch once ran past the 10 s deadline on a CPU and timed out.
# EmbeddingGemma on a CPU embeds ~2,800 tokens/s (4 threads, M4 Pro). Queries get the worker
# between pieces.
_DOCUMENT_PIECE_CHARS = 4000
# The worker holds the model (~480 MB resident with EmbeddingGemma); it is stopped after this long
# unused and loads again on the next request, in ~0.7 s (M4 Pro, CPU or Metal).
_IDLE_SECONDS = 600.0
_DOCUMENT_PIECE_ITEMS = 32


def _pieces(texts: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    size = 0
    for text in texts:
        if out[-1] and (size + len(text) > _DOCUMENT_PIECE_CHARS or len(out[-1]) >= _DOCUMENT_PIECE_ITEMS):
            out.append([])
            size = 0
        out[-1].append(text)
        size += len(text)
    return out


@functools.cache
def embeddings_installed() -> bool:
    """Whether the `embedding` extra is there; without it memory search runs on keywords."""
    return importlib.util.find_spec("llama_cpp") is not None


def embeddings_enabled() -> bool:
    """Whether local embeddings may run: the extra is installed and the user turned them on."""
    return env_flag(LOCAL_EMBEDDINGS_ENV) and embeddings_installed()


class LocalEmbeddingEngine:
    def __init__(self, *, worker: ProcessWorker | None = None) -> None:
        self._worker = worker or ProcessWorker(serve, idle_timeout=_IDLE_SECONDS)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rune-embed")
        self._downloads = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rune-model")
        self._download: Future[tuple[bool, float]] | None = None
        self._download_model_id: str | None = None
        self._download_failures = 0
        self._download_retry_at = 0.0
        self._slots = threading.BoundedSemaphore(32)
        self._sync_lock = threading.Lock()
        self._closed = threading.Event()
        self._model_id = DEFAULT_MODEL
        self._fingerprint: str | None = None
        self._model_stamp: tuple[int, int, int] | None = None
        # (fingerprint, text digest) -> vector; read and written under _sync_lock only.
        self._documents: OrderedDict[tuple[str, bytes], array[float]] = OrderedDict()

    @property
    def dimensions(self) -> int:
        return EMBEDDING_MODELS[self._model_id].dimensions

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint if self.is_ready() else None

    @property
    def health(self) -> dict[str, Any]:
        return {**self._worker.status, "model": self._model_id,
                "fingerprint": self.fingerprint, "dimensions": self.dimensions}

    def is_ready(self) -> bool:
        # Parked means stopped for being idle: the same model loads again on the next request.
        return self._fingerprint is not None and self._worker.status["state"] in ("ready", "parked")

    def _download_model(self, model_id: str) -> Path:
        import urllib.request

        config = EMBEDDING_MODELS[model_id]
        root = _models_dir()
        root.mkdir(parents=True, exist_ok=True)
        path = root / config.file
        with FileLock(str(path) + ".lock", timeout=1):
            if path.is_file() and path.stat().st_size > 0:
                return path
            started = time.monotonic()
            temp: str | None = None
            try:
                url = f"https://huggingface.co/{config.repo}/resolve/main/{config.file}"
                # The socket timeout catches a stalled download; the overall limit a crawling one.
                with urllib.request.urlopen(url, timeout=10) as response, tempfile.NamedTemporaryFile(
                    dir=root, prefix=".download-", delete=False,
                ) as output:
                    temp = output.name
                    total = 0
                    digest = hashlib.sha256()
                    while chunk := response.read(1 << 20):
                        if self._closed.is_set() or time.monotonic() - started > _DOWNLOAD_LIMIT_SECONDS:
                            raise WorkerUnavailable("Model download cancelled or timed out")
                        output.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)
                    if total == 0 or (response.headers.get("Content-Length") and
                                      total != int(response.headers["Content-Length"])):
                        raise ValueError("Incomplete model download")
                    if config.sha256 and digest.hexdigest() != config.sha256:
                        raise ValueError("Downloaded model does not match its pinned hash")
                os.replace(temp, path)
                log.info("embedding_model_downloaded", model=model_id, bytes=total)
            finally:
                if temp is not None:
                    Path(temp).unlink(missing_ok=True)
        return path

    def _background_download(self, model_id: str) -> tuple[bool, float]:
        try:
            self._download_model(model_id)
            return True, time.monotonic()
        except Exception as exc:
            log.warning("embedding_download_failed", model=model_id, error=type(exc).__name__)
            return False, time.monotonic()

    def _schedule_download(self) -> None:
        if self._download is not None and not self._download.done():
            return
        # The download may have finished since the caller checked the cache.
        path = _models_dir() / EMBEDDING_MODELS[self._model_id].file
        if path.is_file() and path.stat().st_size > 0:
            return
        if self._download is not None:
            succeeded, finished = self._download.result()
            self._download = None
            if self._download_model_id == self._model_id:
                self._download_failures = 0 if succeeded else self._download_failures + 1
                delay = min(60, 5 * 2 ** min(self._download_failures - 1, 4))
                self._download_retry_at = finished + (delay if not succeeded else 0)
            else:
                self._download_failures = 0
                self._download_retry_at = 0
        if self._download_model_id != self._model_id:
            self._download_retry_at = 0
            self._download_failures = 0
        if time.monotonic() >= self._download_retry_at:
            self._download_model_id = self._model_id
            self._download = self._downloads.submit(self._background_download, self._model_id)

    def _activate_model(self, model_id: str) -> None:
        self._model_id = model_id
        self._call([], kind="document", timeout=30, download=True)

    def initialize(self, model_id: str = DEFAULT_MODEL) -> None:
        if model_id not in EMBEDDING_MODELS:
            log.warning("unknown_embedding_model", requested=model_id, fallback=DEFAULT_MODEL)
            model_id = DEFAULT_MODEL
        with self._sync_lock:
            if self._closed.is_set():
                raise WorkerUnavailable("Embedding engine is closed")
            self._activate_model(model_id)

    def _call(self, texts: list[str], *, kind: EmbeddingKind, timeout: float,
              download: bool = False) -> list[EmbeddingVector]:
        if self._closed.is_set():
            raise WorkerUnavailable("Embedding engine is closed")
        if not embeddings_installed():
            # without llama.cpp the model could never load: don't fetch hundreds of MB of it
            self._worker.last_error = "embedding_extra_missing"
            raise WorkerUnavailable("Local embeddings need the `embedding` extra (llama-cpp-python)")
        if not env_flag(LOCAL_EMBEDDINGS_ENV):
            # off unless asked for: no download, no worker, searches stay on keywords
            self._worker.last_error = "local_embeddings_off"
            raise WorkerUnavailable(f"Local embeddings are off; set {LOCAL_EMBEDDINGS_ENV}=1 to turn them on")
        path = _models_dir() / EMBEDDING_MODELS[self._model_id].file
        if not path.is_file() or path.stat().st_size == 0:
            if download:
                path = self._download_model(self._model_id)
            else:
                self._schedule_download()
                if not path.is_file() or path.stat().st_size == 0:
                    self._worker.last_error = "model_not_cached"
                    raise WorkerUnavailable("Embedding model is not cached; download pending or waiting to retry")
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamp != self._model_stamp:
            self._fingerprint = None
        result = self._worker.request({"model_id": self._model_id, "model_path": str(path),
                                       "model_stamp": list(stamp), "texts": texts, "kind": kind,
                                       "cpu_only": env_flag("RUNE_EMBEDDING_CPU_ONLY")},
                                      timeout=timeout)
        vectors = result.get("vectors")
        fingerprint = result.get("fingerprint")
        if (result.get("dimensions") != self.dimensions or not isinstance(fingerprint, str)
                or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)
                or not isinstance(vectors, list)
                or len(vectors) != len(texts)
                or any(not isinstance(v, list) or len(v) != self.dimensions or
                       any(isinstance(n, bool) or not isinstance(n, (int, float)) or
                           not math.isfinite(n) for n in v) for v in vectors)):
            self._fingerprint = None
            self._worker.invalidate("Invalid embedding response")
            raise WorkerUnavailable("Invalid embedding response")
        self._fingerprint = fingerprint
        self._model_stamp = stamp
        return [EmbeddingVector(v, fingerprint) for v in vectors]

    def embed_sync(self, text: str, *, kind: EmbeddingKind = "document") -> EmbeddingVector:
        return self.embed_batch_sync([text], kind=kind)[0]

    def embed_batch_sync(self, texts: list[str], *, kind: EmbeddingKind = "document",
                         deadline: float | None = None) -> list[EmbeddingVector]:
        if not texts:
            return []
        if len(texts) > 256 or sum(len(t.encode()) for t in texts) > 1_000_000:
            raise WorkerUnavailable("Embedding batch exceeds input limit")
        deadline = deadline if deadline is not None else time.monotonic() + 10
        if not self._sync_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise WorkerUnavailable("Embedding queue deadline exceeded")
        try:
            if kind == "query":
                return self._call(texts, kind=kind, timeout=max(0, deadline - time.monotonic()))
            return self._documents_sync(texts, deadline)
        finally:
            self._sync_lock.release()

    def _documents_sync(self, texts: list[str], deadline: float) -> list[EmbeddingVector]:
        """Documents, from the cache where the active model already embedded the same text."""
        keys = [hashlib.blake2b(t.encode(), digest_size=16).digest() for t in texts]
        known = self._fingerprint
        found: dict[int, EmbeddingVector] = {}
        if known is not None:
            for i, key in enumerate(keys):
                if (cached := self._documents.get((known, key))) is not None:
                    self._documents.move_to_end((known, key))
                    found[i] = EmbeddingVector(cached.tolist(), known)
        missing = [i for i in range(len(texts)) if i not in found]
        if missing:
            fresh = self._call([texts[i] for i in missing], kind="document",
                               timeout=max(0, deadline - time.monotonic()))
            if fresh[0].fingerprint != known:
                # Another model answers now: what was cached for the old one can't be mixed in.
                self._documents.clear()
                if found:
                    found, missing = {}, list(range(len(texts)))
                    fresh = self._call(texts, kind="document", timeout=max(0, deadline - time.monotonic()))
            for i, vector in zip(missing, fresh, strict=True):
                found[i] = vector
                self._documents[(vector.fingerprint, keys[i])] = array("f", vector)
                self._documents.move_to_end((vector.fingerprint, keys[i]))
            while len(self._documents) > _DOCUMENT_CACHE_SIZE:
                self._documents.popitem(last=False)
        return [found[i] for i in range(len(texts))]

    async def embed_single(self, text: str, *, kind: EmbeddingKind = "document") -> EmbeddingVector:
        return (await self.embed([text], kind=kind))[0]

    async def embed(self, texts: list[str], *, kind: EmbeddingKind = "document") -> list[EmbeddingVector]:
        if kind == "query" or len(texts) <= 1:
            return await self._embed_piece(texts, kind)
        out: list[EmbeddingVector] = []
        for piece in _pieces(texts):
            vectors = await self._embed_piece(piece, kind)
            if out and vectors and vectors[0].fingerprint != out[0].fingerprint:
                raise WorkerUnavailable("Embedding model changed during a batch")
            out += vectors
        return out

    async def _embed_piece(self, texts: list[str], kind: EmbeddingKind) -> list[EmbeddingVector]:
        if self._closed.is_set() or not self._slots.acquire(blocking=False):
            raise WorkerUnavailable("Embedding queue unavailable")
        try:
            future = self._executor.submit(self.embed_batch_sync, texts, kind=kind,
                                           deadline=time.monotonic() + 10)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        # Cancelling the await cancels queued work; running work drains under its deadline.
        return await asyncio.wrap_future(future)

    def dispose(self) -> None:
        self._closed.set()
        self._worker.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._downloads.shutdown(wait=False, cancel_futures=True)
        self._fingerprint = None
        self._documents.clear()


_engine: LocalEmbeddingEngine | None = None
_engine_lock = threading.Lock()


def get_embedding_engine() -> LocalEmbeddingEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = LocalEmbeddingEngine()
        return _engine


async def dispose_embedding_engine() -> None:
    global _engine
    with _engine_lock:
        engine, _engine = _engine, None
    if engine is not None:
        await asyncio.to_thread(engine.dispose)


def _shutdown() -> None:
    if _engine is not None:
        _engine.dispose()


atexit.register(_shutdown)


class LocalEmbeddingProvider:
    """Every embedding caller goes through this contract.

    `kind` is "query" for text searched with and "document" for text stored and searched for. It
    defaults to "document"; a search that embeds its query as a document loses most of the recall
    the instruction-trained default model was chosen for.
    """

    @property
    def fingerprint(self) -> str | None:
        return get_embedding_engine().fingerprint

    def embed_sync(self, texts: list[str], *, kind: EmbeddingKind = "document") -> list[EmbeddingVector]:
        return get_embedding_engine().embed_batch_sync(texts, kind=kind)

    async def embed(self, texts: list[str], *, kind: EmbeddingKind = "document") -> list[EmbeddingVector]:
        return await get_embedding_engine().embed(texts, kind=kind)

    async def embed_single(self, text: str, *, kind: EmbeddingKind = "document") -> EmbeddingVector:
        return await get_embedding_engine().embed_single(text, kind=kind)


_provider = LocalEmbeddingProvider()


def get_embedding_provider() -> LocalEmbeddingProvider:
    return _provider


def match_threshold() -> float:
    """Cosine from which a query embedded by the active model matches a rule, skill or past turn."""
    engine = _engine
    return EMBEDDING_MODELS[engine._model_id if engine is not None else DEFAULT_MODEL].match_threshold


def model_cached() -> bool:
    """Whether the default model is on disk (a download only replaces it whole)."""
    return (_models_dir() / EMBEDDING_MODELS[DEFAULT_MODEL].file).is_file()


async def warm_up_embedding() -> None:
    if not embeddings_enabled() or not model_cached():
        return
    try:
        await asyncio.to_thread(get_embedding_engine().initialize)
    except Exception as exc:
        log.warning("embedding_warm_up_failed", error=type(exc).__name__)
