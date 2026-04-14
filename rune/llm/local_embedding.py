"""Local GGUF Embedding engine for RUNE.

Ported from src/llm/local-embedding.ts - uses llama-cpp-python with GGUF
models for local embedding generation.  Matches the TS original:
  - node-llama-cpp -> llama-cpp-python
  - Default model: nomic-embed-text (768 dimensions)
  - Same-dimension failover between models
  - Model files stored in ~/.rune/models/embeddings/
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

# Supported embedding models (matches TS EMBEDDING_MODELS)


@dataclass(slots=True, frozen=True)
class EmbeddingModelConfig:
    name: str
    repo: str
    file: str
    size: str
    dimensions: int


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    "nomic-embed-text": EmbeddingModelConfig(
        name="Nomic Embed Text v1.5",
        repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
        file="nomic-embed-text-v1.5.Q8_0.gguf",
        size="~140MB",
        dimensions=768,
    ),
    "bge-small": EmbeddingModelConfig(
        name="BGE Small EN v1.5",
        repo="BAAI/bge-small-en-v1.5-gguf",
        file="bge-small-en-v1.5-q8_0.gguf",
        size="~45MB",
        dimensions=384,
    ),
}

DEFAULT_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # matches TS original (nomic-embed-text)


def _models_dir() -> Path:
    return rune_data() / "models" / "embeddings"


# Local Embedding Engine (ported from TS LocalEmbeddingEngine)


class LocalEmbeddingEngine:
    """Local GGUF embedding engine using llama-cpp-python.

    Mirrors the TS LocalEmbeddingEngine:
    - Lazy model download on first use
    - Same-dimension failover
    - Singleton pattern via get_embedding_engine()
    """

    def __init__(self) -> None:
        self._model: Any = None  # Llama instance
        self._model_id: str | None = None
        self._model_config: EmbeddingModelConfig | None = None
        self._initialized = False
        self._init_failovers = 0
        self._embed_failovers = 0
        self._last_error: str | None = None
        # Thread safety: Llama is not thread-safe. Serialize all access.
        # Sync lock protects direct embed_sync / initialize callers.
        self._sync_lock = threading.Lock()
        # Dedicated single-worker executor: caps concurrent native calls
        # to one thread regardless of caller count.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rune-embed",
        )
        # Async lock: lazily bound to the running loop. Protects concurrent
        # awaits on embed_single / embed. None until first use because the
        # loop may not exist at __init__ time (singleton constructed early).
        self._async_lock: asyncio.Lock | None = None

    @property
    def dimensions(self) -> int:
        return self._model_config.dimensions if self._model_config else EMBEDDING_DIM

    def is_ready(self) -> bool:
        return self._initialized

    # -- Model download (matches TS downloadModel) ---------------------------

    def _download_model(self, model_id: str) -> Path:
        """Download a GGUF model file if not already present."""
        config = EMBEDDING_MODELS.get(model_id)
        if config is None:
            raise ValueError(f"Unknown embedding model: {model_id}")

        models_dir = _models_dir()
        model_path = models_dir / config.file

        if model_path.exists():
            log.debug("embedding_model_exists", path=str(model_path))
            return model_path

        models_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://huggingface.co/{config.repo}/resolve/main/{config.file}"
        log.info(
            "downloading_embedding_model",
            model=config.name,
            size=config.size,
            url=url,
        )

        import urllib.request
        try:
            urllib.request.urlretrieve(url, str(model_path))
        except Exception as exc:
            # Clean up partial download
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download embedding model {config.name}: {exc}"
            ) from exc

        log.info("embedding_model_downloaded", path=str(model_path))
        return model_path

    # -- Model activation (matches TS activateModel) -------------------------

    def _activate_model(self, model_id: str) -> None:
        """Load a GGUF model into llama-cpp-python."""
        config = EMBEDDING_MODELS.get(model_id)
        if config is None:
            raise ValueError(f"Unknown embedding model: {model_id}")

        # Dispose previous model
        if self._model is not None:
            del self._model
            self._model = None

        model_path = self._download_model(model_id)

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python"
            ) from None

        # Suppress ggml_metal_init stderr spam (bf16 "not supported" lines)
        import contextlib
        import io

        stderr_trap = io.StringIO()
        with contextlib.redirect_stderr(stderr_trap):
            self._model = Llama(
                model_path=str(model_path),
                embedding=True,
                n_ctx=0,  # embedding only - no text generation context needed
                n_batch=512,
                verbose=False,
            )
        self._model_id = model_id
        self._model_config = config
        self._initialized = True
        self._last_error = None

        log.info(
            "embedding_engine_initialized",
            model=config.name,
            model_id=model_id,
            dimensions=config.dimensions,
        )

    # -- Same-dimension failover (matches TS getSameDimensionCandidates) ------

    def _same_dim_candidates(self, model_id: str) -> list[str]:
        """Get candidate model IDs with the same embedding dimension."""
        config = EMBEDDING_MODELS.get(model_id)
        if config is None:
            return [model_id]
        dim = config.dimensions
        return [model_id] + [
            mid for mid, cfg in EMBEDDING_MODELS.items()
            if mid != model_id and cfg.dimensions == dim
        ]

    # -- Initialize (matches TS initialize) ----------------------------------

    def initialize(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize the engine with failover across same-dimension models."""
        if model_id not in EMBEDDING_MODELS:
            log.warning(
                "unknown_embedding_model",
                requested=model_id,
                fallback=DEFAULT_MODEL,
            )
            model_id = DEFAULT_MODEL

        with self._sync_lock:
            # Re-check after acquiring lock — another caller may have won the race.
            if self._initialized and self._model_id == model_id:
                return

            last_error: Exception | None = None
            candidates = self._same_dim_candidates(model_id)
            for candidate in candidates:
                try:
                    self._activate_model(candidate)
                    if candidate != model_id:
                        self._init_failovers += 1
                        log.warning(
                            "embedding_model_fallback",
                            requested=model_id,
                            selected=candidate,
                        )
                    return
                except Exception as exc:
                    last_error = exc
                    self._last_error = str(exc)[:200]
                    log.warning(
                        "embedding_init_failed",
                        model_id=candidate,
                        error=self._last_error,
                    )
            raise last_error or RuntimeError("No embedding model could be initialized")

    # -- Embed (matches TS embed) -------------------------------------------

    def embed_sync(self, text: str) -> list[float]:
        """Generate embedding for a single text (synchronous, thread-safe)."""
        # initialize() acquires the lock internally; call before entering
        # our critical section to avoid a re-entrant acquire.
        if not self._initialized or self._model is None:
            self.initialize()

        with self._sync_lock:
            try:
                import contextlib
                import io
                stderr_trap = io.StringIO()
                with contextlib.redirect_stderr(stderr_trap):
                    result = self._model.embed(text)
                if result and isinstance(result[0], list):
                    return result[0]
                return result
            except Exception as exc:
                failed_model = self._model_id or DEFAULT_MODEL
                self._last_error = str(exc)[:200]
                log.warning(
                    "embedding_failed_trying_failover",
                    model_id=failed_model,
                    error=self._last_error,
                )

                candidates = self._same_dim_candidates(failed_model)
                for candidate in candidates:
                    if candidate == failed_model:
                        continue
                    try:
                        self._activate_model(candidate)
                        result = self._model.embed(text)
                        self._embed_failovers += 1
                        if result and isinstance(result[0], list):
                            return result[0]
                        return result
                    except Exception as fb_exc:
                        self._last_error = str(fb_exc)[:200]
                        log.warning(
                            "embedding_failover_failed",
                            from_model=failed_model,
                            to_model=candidate,
                            error=self._last_error,
                        )

                raise RuntimeError(f"All embedding models failed: {exc}") from exc

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (synchronous)."""
        # Iterate; each embed_sync call acquires the sync lock individually.
        # Lock is re-entrant-safe: we don't hold it across iterations so
        # other callers can interleave per-item.
        return [self.embed_sync(t) for t in texts]

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily bind the async lock to the current running loop."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding (async, serialized via async lock + dedicated executor)."""
        async with self._get_async_lock():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self.embed_sync, text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings (async, serialized via async lock + dedicated executor)."""
        async with self._get_async_lock():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self.embed_batch_sync, texts,
            )

    def dispose(self) -> None:
        """Release model resources and shut down the embedding executor."""
        with self._sync_lock:
            if self._model is not None:
                del self._model
                self._model = None
            self._initialized = False
            self._model_id = None
            self._model_config = None
        # Shut down without blocking — pending tasks drain naturally.
        self._executor.shutdown(wait=False)


# Singleton + compat API

_engine: LocalEmbeddingEngine | None = None


def get_embedding_engine() -> LocalEmbeddingEngine:
    """Get or create the singleton embedding engine."""
    global _engine
    if _engine is None:
        _engine = LocalEmbeddingEngine()
    return _engine


async def dispose_embedding_engine() -> None:
    """Dispose the singleton engine. Call before process exit."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


# Backward-compatible API used by memory/manager.py
# (same interface as the old LocalEmbeddingProvider)

class LocalEmbeddingProvider:
    """Compat wrapper that delegates to LocalEmbeddingEngine."""

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return get_embedding_engine().embed_batch_sync(texts)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await get_embedding_engine().embed(texts)

    async def embed_single(self, text: str) -> list[float]:
        return await get_embedding_engine().embed_single(text)


_provider: LocalEmbeddingProvider | None = None


def get_embedding_provider() -> LocalEmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = LocalEmbeddingProvider()
    return _provider


# Warm-up (only if model already downloaded)


def _is_model_cached() -> bool:
    """Check if the default GGUF model file exists locally."""
    config = EMBEDDING_MODELS.get(DEFAULT_MODEL)
    if config is None:
        return False
    return (_models_dir() / config.file).exists()


async def warm_up_embedding() -> None:
    """Pre-load embedding model if already downloaded locally.

    Skips warm-up if model hasn't been downloaded yet to avoid
    unexpected network downloads on startup.
    """
    try:
        if not _is_model_cached():
            log.debug("embedding_warm_up_skipped", reason="model not cached locally")
            return
        engine = get_embedding_engine()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, engine.initialize)
        log.info("embedding_warm_up_complete")
    except Exception as exc:
        log.debug("embedding_warm_up_failed", error=str(exc)[:100])
