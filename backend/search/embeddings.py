"""
embeddings.py — turn text into vectors with a free, local model.
Module 4: Embeddings + RAG

WHAT an embedding is:
    A list of numbers (here 384) that places a text in "meaning space".
    Texts about similar things land close together, even with different
    words: "ML engineer" sits near "machine learning developer". Closeness
    is measured with cosine similarity (1 = same direction, 0 = unrelated).

WHY all-MiniLM-L6-v2:
    Free, runs locally on CPU (no API key, no per-call cost), ~90 MB,
    and fast — a few hundred short texts per second on a laptop. The trade-
    off vs bigger models (all-mpnet-base-v2, 768 dims) is a bit less
    accuracy for ~5x speed and half the storage. The cross-encoder reranker
    (rerank.py) wins back much of that accuracy where it matters: the top 20.

WHY normalize vectors (length 1):
    For unit vectors, cosine similarity == dot product. That makes scoring a
    single matrix multiply, and matches pgvector's cosine operator (<=>).

WHY batch:
    The model processes a batch of texts in one forward pass. Embedding 500
    chunks one at a time is ~10x slower than 16 batches of 32.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that can embed texts. Tests use a fake; production uses the model."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dimensions) float32 array of L2-normalized vectors."""
        ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. The model loads lazily on first use."""

    def __init__(self, model_name: str, *, batch_size: int = 32, device: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        # WHY lazy: importing torch + loading weights takes seconds. Code that
        # never embeds (tests, the scraper) shouldn't pay that cost.
        if self._model is None:
            from sentence_transformers import (
                SentenceTransformer,  # heavy import, keep local
            )

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimensions(self) -> int:
        getter = (getattr(self.model, "get_embedding_dimension", None)
                  or self.model.get_sentence_embedding_dimension)
        return int(getter())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 (rows of zeros stay zero)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    """
    Combine several chunk vectors into one document vector.

    WHY: the jobs table has ONE vector column per job (one HNSW index entry).
    Averaging the normalized chunk vectors gives a vector pointing at the
    "centre of meaning" of the whole document; re-normalizing keeps it a
    valid unit vector for cosine search.
    """
    if len(vectors) == 0:
        raise ValueError("cannot pool zero vectors")
    return normalize(vectors.mean(axis=0))


@lru_cache(maxsize=4)
def get_embedder(model_name: str, batch_size: int = 32) -> SentenceTransformerEmbedder:
    """One shared model per process — loading it twice wastes ~100 MB of RAM."""
    return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
