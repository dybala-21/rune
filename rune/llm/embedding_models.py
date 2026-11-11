"""Local embedding model catalog and cache identity."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rune.utils.paths import rune_data

# What a text is embedded as. Models trained with task prompts (EmbeddingGemma) only reach their
# measured recall when a query and a stored document each carry their own prompt.
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
    # trained maximum, which costs memory for every token of it.
    n_ctx: int = 0
    # Put in front of a query / a document. Part of the fingerprint through prompt_version.
    query_prefix: str = ""
    document_prefix: str = ""
    # Names the input format in the fingerprint: change the prefix, change this, and every index
    # built with the old wording is rebuilt instead of mixed with the new one.
    prompt_version: str = "raw-text-v1"
    # SHA-256 of the GGUF file. A download or cached file that doesn't match is refused.
    sha256: str = ""
    # Cosine from which a query counts as matching a rule or a skill. Each model spreads its scores
    # differently, so the cut is measured per model; RUNE_RULE_SIM_THRESHOLD and
    # RUNE_SKILL_SIM_THRESHOLD still override it.
    match_threshold: float = 0.55


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    # Measured 2026-11-12 against nomic, granite-107m, multilingual-e5-small and Qwen3-Embedding-0.6B.
    # Recall@10 alone: Ko-StrategyQA 0.868 (Qwen3 0.825, nomic 0.05), SciFact 0.906; memory set R@1
    # 0.97. On a CPU (4 threads, M4 Pro) a query takes 21 ms at p95 and the process holds ~430 MB,
    # 258 MB of it the file mapped read-only. 768 dimensions; Gemma Terms of Use.
    "embeddinggemma-300m": EmbeddingModelConfig(
        name="EmbeddingGemma 300M (QAT Q4_0)",
        repo="ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
        file="embeddinggemma-300M-qat-Q4_0.gguf",
        size="~265MB",
        dimensions=768,
        n_ctx=512,
        # The prompts from the model card for retrieval.
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
        prompt_version="gemma-retrieval-v1",
        sha256="50d28e22432a148f6f8a86eab3700f92add5d1f54baf7790675a2a4dadbccf26",
        # Memory set: keeps 97% of relevant pairs, passes 0.4% of 5,088 unrelated ones.
        # Ko-StrategyQA: keeps 82%, passes 0.12%. 0.55 would keep 73% / 60%.
        match_threshold=0.45,
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

DEFAULT_MODEL = "embeddinggemma-300m"
EMBEDDING_DIM = EMBEDDING_MODELS[DEFAULT_MODEL].dimensions


def _models_dir() -> Path:
    return rune_data() / "models" / "embeddings"
