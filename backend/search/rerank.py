"""
rerank.py — cross-encoder reranking of the top candidates.
Module 4: Embeddings + RAG

BI-ENCODER vs CROSS-ENCODER:
    The embedder (bi-encoder) reads the query and each job SEPARATELY and
    compares two vectors. That's why it's fast — job vectors are computed
    once, ahead of time — but it never sees query and job side by side.

    A cross-encoder reads "query [SEP] job" TOGETHER through the whole
    transformer, so every query word can attend to every job word. It
    understands "remote Python role, NOT data science" far better.
    The cost: nothing can be precomputed — one full model pass per pair.

THE STANDARD PATTERN — retrieve wide, rerank narrow:
    cheap retrievers pick ~20 candidates from thousands → the expensive
    cross-encoder orders just those 20 → return the top 5.
    20 pairs take ~100–300 ms on CPU; reranking all 5,000 jobs would take minutes.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 — free, ~90 MB, trained on
Bing search queries (MS MARCO), so it's good at "query → relevant passage".
"""
from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query (higher = more relevant)."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:  # lazy, like the embedder
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
