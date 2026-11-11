"""Load llama.cpp only inside the embedding worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import socket
from pathlib import Path
from typing import Any

from rune.llm.embedding_models import EMBEDDING_MODELS
from rune.utils.logger import get_logger
from rune.utils.process_worker import receive_message, send_message, watch_parent

log = get_logger(__name__)


def _stamp(path: str) -> tuple[int, int, int]:
    stat = Path(path).stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def serve(sock: socket.socket) -> None:
    watch_parent()
    model: Any = None
    model_key: tuple[str, str, tuple[int, ...]] | None = None
    identity = ""
    try:
        while True:
            request = receive_message(sock, None)
            try:
                params = request["payload"]
                key = (params["model_id"], params["model_path"], tuple(params["model_stamp"]))
                config = EMBEDDING_MODELS[key[0]]
                if _stamp(key[1]) != key[2]:
                    raise ValueError("Embedding model changed before loading")
                if key != model_key:
                    from llama_cpp import Llama

                    if model is not None:
                        model.close()
                    model, model_key = None, None
                    with Path(key[1]).open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                    if config.sha256 and digest != config.sha256:
                        raise ValueError("Embedding model file does not match its pinned hash")
                    # All layers go to the GPU when llama.cpp was built with one (Metal on Apple
                    # Silicon: 17 ms a query against 187 ms on the CPU); otherwise this is a no-op.
                    model = Llama(
                        model_path=key[1], embedding=True, n_ctx=config.n_ctx,
                        n_batch=512, n_gpu_layers=0 if params.get("cpu_only") else -1, verbose=False,
                    )
                    if _stamp(key[1]) != key[2]:
                        raise ValueError("Embedding model changed while loading")
                    identity = hashlib.sha256(json.dumps([
                        key[0], digest, config.dimensions, config.prompt_version,
                        importlib.metadata.version("llama-cpp-python"),
                    ]).encode()).hexdigest()
                    model_key = key
                prefix = config.query_prefix if params.get("kind") == "query" else ""
                vectors = []
                for text in params.get("texts", []):
                    # Truncated to n_ctx tokens; the prefix comes first, so it always survives.
                    vector = model.embed(prefix + text, truncate=True)
                    vectors.append(vector[0] if vector and isinstance(vector[0], list) else vector)
                result = {"vectors": vectors, "fingerprint": identity, "dimensions": config.dimensions}
                send_message(sock, {"id": request["id"], "result": result}, None)
            except Exception as exc:
                send_message(sock, {"id": request["id"], "error": type(exc).__name__}, None)
    except (EOFError, OSError):
        log.debug("embedding_worker_connection_closed")
        return
    finally:
        sock.close()
        if model is not None:
            model.close()
