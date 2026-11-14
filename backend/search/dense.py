"""
dense.py — vector (dense) search.
Module 4: Embeddings + RAG

Two interchangeable retrievers behind one interface:

    PgVectorRetriever      production: asks Postgres. The HNSW index finds
                           nearest neighbours without comparing against every
                           row — fast even with millions of jobs.
    InMemoryDenseRetriever offline/tests: one numpy matrix multiply over all
                           vectors. Exact, simple, fine for a few thousand jobs.

WHY cosine similarity (and not L2 or dot product):
    Cosine compares DIRECTION — what a text is about — and ignores length,
    so a 50-word and a 900-word JD about the same role score alike.
    Our vectors are normalized, so cosine == dot product (cheapest to compute),
    and L2 distance would give the same ranking anyway. pgvector's cosine
    operator is <=> (it returns DISTANCE = 1 − similarity).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from backend.db.jobs_repo import JobsRepo
from backend.search.embeddings import normalize
from backend.search.types import SearchDoc, SearchFilters


class DenseRetriever(Protocol):
    async def search(
        self, query_vector: np.ndarray, k: int, filters: SearchFilters | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k (doc_id, cosine similarity), best first."""
        ...


class InMemoryDenseRetriever:
    def __init__(self, docs: list[SearchDoc], vectors: np.ndarray) -> None:
        if len(docs) != len(vectors):
            raise ValueError("need exactly one vector per doc")
        self.docs = docs
        self.matrix = normalize(np.asarray(vectors, dtype=np.float32)) if len(docs) else vectors

    async def search(
        self, query_vector: np.ndarray, k: int, filters: SearchFilters | None = None,
    ) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        scores = self.matrix @ normalize(np.asarray(query_vector, dtype=np.float32))
        hits: list[tuple[str, float]] = []
        for i in np.argsort(-scores):
            if filters and not filters.matches(self.docs[i]):
                continue
            hits.append((self.docs[i].id, float(scores[i])))
            if len(hits) == k:
                break
        return hits


class PgVectorRetriever:
    """Delegates to the match_jobs() SQL function (see setup_db.sql)."""

    def __init__(self, jobs_repo: JobsRepo, *, threshold: float = 0.0) -> None:
        self.jobs_repo = jobs_repo
        self.threshold = threshold

    async def search(
        self, query_vector: np.ndarray, k: int, filters: SearchFilters | None = None,
    ) -> list[tuple[str, float]]:
        filters = filters or SearchFilters()
        rows = await self.jobs_repo.semantic_search(
            np.asarray(query_vector, dtype=float).tolist(),
            threshold=self.threshold,
            limit=k,
            remote_only=filters.remote_only,
            min_salary=filters.min_salary,
            posted_after=filters.posted_after,
            sources=filters.sources or None,
        )
        hits = [(str(r["id"]), float(r["similarity"])) for r in rows]
        # match_score isn't a SQL filter — apply it here so both retrievers agree.
        if filters.min_match_score is not None:
            keep = {str(r["id"]) for r in rows if (r.get("match_score") or 0) >= filters.min_match_score}
            hits = [h for h in hits if h[0] in keep]
        return hits
