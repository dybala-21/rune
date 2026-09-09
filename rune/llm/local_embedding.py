"""Embedding provider with a supervised native worker."""

from __future__ import annotations

import asyncio
import atexit
import math
import os
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from filelock import FileLock

from rune.llm.embedding_models import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODELS,
    EmbeddingModelConfig,
    EmbeddingVector,
    _models_dir,
)
from rune.llm.embedding_worker import serve
from rune.utils.logger import get_logger
from rune.utils.process_worker import ProcessWorker, WorkerUnavailable

log = get_logger(__name__)
__all__ = ["DEFAULT_MODEL", "EMBEDDING_DIM", "EMBEDDING_MODELS", "EmbeddingModelConfig",
           "LocalEmbeddingEngine", "LocalEmbeddingProvider", "get_embedding_engine",
           "get_embedding_provider", "dispose_embedding_engine", "warm_up_embedding"]


class LocalEmbeddingEngine:
    def __init__(self, *, worker: ProcessWorker | None = None) -> None:
        self._worker = worker or ProcessWorker(serve)
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
        return self._fingerprint is not None and self._worker.status["state"] == "ready"

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
                with urllib.request.urlopen(url, timeout=10) as response, tempfile.NamedTemporaryFile(
                    dir=root, prefix=".download-", delete=False,
                ) as output:
                    temp = output.name
                    total = 0
                    while chunk := response.read(65536):
                        if self._closed.is_set() or time.monotonic() - started > 30:
                            raise WorkerUnavailable("Model download cancelled or timed out")
                        output.write(chunk)
                        total += len(chunk)
                    if total == 0 or (response.headers.get("Content-Length") and
                                      total != int(response.headers["Content-Length"])):
                        raise ValueError("Incomplete model download")
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
        if self._download is not None:
            if not self._download.done():
                return
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
        self._call([], timeout=30, download=True)

    def initialize(self, model_id: str = DEFAULT_MODEL) -> None:
        if model_id not in EMBEDDING_MODELS:
            log.warning("unknown_embedding_model", requested=model_id, fallback=DEFAULT_MODEL)
            model_id = DEFAULT_MODEL
        with self._sync_lock:
            if self._closed.is_set():
                raise WorkerUnavailable("Embedding engine is closed")
            self._activate_model(model_id)

    def _call(self, texts: list[str], *, timeout: float, download: bool = False) -> list[list[float]]:
        if self._closed.is_set():
            raise WorkerUnavailable("Embedding engine is closed")
        path = _models_dir() / EMBEDDING_MODELS[self._model_id].file
        if not path.is_file() or path.stat().st_size == 0:
            if download:
                path = self._download_model(self._model_id)
            else:
                self._schedule_download()
                self._worker.last_error = "model_not_cached"
                raise WorkerUnavailable("Embedding model is not cached; download pending or waiting to retry")
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamp != self._model_stamp:
            self._fingerprint = None
        result = self._worker.request({"model_id": self._model_id, "model_path": str(path),
                                       "model_stamp": list(stamp), "texts": texts}, timeout=timeout)
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

    def embed_sync(self, text: str) -> list[float]:
        return self.embed_batch_sync([text])[0]

    def embed_batch_sync(self, texts: list[str], *, deadline: float | None = None) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > 256 or sum(len(t.encode()) for t in texts) > 1_000_000:
            raise WorkerUnavailable("Embedding batch exceeds input limit")
        deadline = deadline if deadline is not None else time.monotonic() + 10
        if not self._sync_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise WorkerUnavailable("Embedding queue deadline exceeded")
        try:
            return self._call(texts, timeout=max(0, deadline - time.monotonic()))
        finally:
            self._sync_lock.release()

    async def embed_single(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._closed.is_set() or not self._slots.acquire(blocking=False):
            raise WorkerUnavailable("Embedding queue unavailable")
        try:
            future = self._executor.submit(self.embed_batch_sync, texts, deadline=time.monotonic() + 10)
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
    @property
    def fingerprint(self) -> str | None:
        return get_embedding_engine().fingerprint

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return get_embedding_engine().embed_batch_sync(texts)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await get_embedding_engine().embed(texts)

    async def embed_single(self, text: str) -> list[float]:
        return await get_embedding_engine().embed_single(text)


_provider = LocalEmbeddingProvider()


def get_embedding_provider() -> LocalEmbeddingProvider:
    return _provider


def _is_model_cached() -> bool:
    return (_models_dir() / EMBEDDING_MODELS[DEFAULT_MODEL].file).is_file()


async def warm_up_embedding() -> None:
    if not _is_model_cached():
        return
    try:
        await asyncio.to_thread(get_embedding_engine().initialize)
    except Exception as exc:
        log.warning("embedding_warm_up_failed", error=type(exc).__name__)
