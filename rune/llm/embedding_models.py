"""Local embedding model catalog and cache identity."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rune.utils.paths import rune_data

# What a text is embedded as. Models trained with instructions (Qwen3-Embedding) only reach their
# measured recall when a query carries the instruction and a stored document does not.
EmbeddingKind = Literal["query", "document"]


class EmbeddingVector(list[float]):
    def __init__(self, values: list[float], fingerprint: str) -> None:
        super().__init__(values)
        self.fingerprint = fingerprint

@dataclass(slots=True, frozen=True)
class EmbeddingModelConfig:
    name: str
    repo: str
    file: str
    size: str
    dimensions: int
    # Context per sequence, in tokens; longer input is truncated. 0 lets llama.cpp use the model's
    # trained maximum, which for Qwen3 (32K) costs 4.2 GB instead of 0.76 GB.
    n_ctx: int = 0
    # Put in front of a query, never a document. Part of the fingerprint through prompt_version.
    query_prefix: str = ""
    # Names the input format in the fingerprint: change the prefix, change this, and every index
    # built with the old wording is rebuilt instead of mixed with the new one.
    prompt_version: str = "raw-text-v1"
    # SHA-256 of the GGUF file. A download or cached file that doesn't match is refused.
    sha256: str = ""


QWEN3_QUERY_V1 = ("Instruct: Given a user's question, retrieve the remembered fact or past task "
                  "that answers it\nQuery: ")

EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    # Ko-StrategyQA recall@10 0.825 against nomic's 0.05; 1024 dimensions; Apache-2.0.
    "qwen3-embedding-0.6b": EmbeddingModelConfig(
        name="Qwen3 Embedding 0.6B",
        repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
        file="Qwen3-Embedding-0.6B-Q8_0.gguf",
        size="~610MB",
        dimensions=1024,
        n_ctx=512,
        query_prefix=QWEN3_QUERY_V1,
        prompt_version="query-v1",
        sha256="06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
    ),
    # Kept so an existing nomic index stays readable; nothing selects it by default.
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

DEFAULT_MODEL = "qwen3-embedding-0.6b"
EMBEDDING_DIM = EMBEDDING_MODELS[DEFAULT_MODEL].dimensions


def _models_dir() -> Path:
    return rune_data() / "models" / "embeddings"
