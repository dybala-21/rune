"""Local embedding model catalog and cache identity."""

from dataclasses import dataclass
from pathlib import Path

from rune.utils.paths import rune_data


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
EMBEDDING_DIM = 768


def _models_dir() -> Path:
    return rune_data() / "models" / "embeddings"
