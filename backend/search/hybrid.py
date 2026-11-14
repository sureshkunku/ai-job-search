"""
hybrid.py — the full search pipeline in one class.
Module 4: Embeddings + RAG

    query ──┬─▶ embed ─▶ dense retriever (meaning)   ─▶ top 20 ─┐
            │                                                    ├─▶ RRF merge ─▶ top 20 ─▶ cross-encoder ─▶ top 5
            └─────────▶ BM25 (exact keywords)         ─▶ top 20 ─┘
                 (metadata filters applied inside both retrievers)

WHY each stage exists:
    dense   catches paraphrases ("ML" ≈ "machine learning")
    BM25    catches exact must-have terms ("FastAPI", "pgvector")
    RRF     merges two incompatible score scales by rank
    rerank  reads query + job together for the final, most accurate order

Dense and BM25 run CONCURRENTLY (asyncio.gather): the vector query waits on
Postgres while BM25 scores on a worker thread.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from backend.db.jobs_repo import JobsRepo
from backend.search.bm25 import BM25Index
from backend.search.dense import (
    DenseRetriever,
    InMemoryDenseRetriever,
    PgVectorRetriever,
)
from backend.search.embeddings import Embedder
from backend.search.fusion import reciprocal_rank_fusion
from backend.search.pipeline import embed_job_documents
from backend.search.rerank import Reranker
from backend.search.types import SearchDoc, SearchFilters, SearchHit, SearchResponse


class HybridSearcher:
    def __init__(
        self,
        docs: list[SearchDoc],
        embedder: Embedder,
        dense: DenseRetriever,
        reranker: Reranker | None = None,
        *,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.docs_by_id = {d.id: d for d in docs}
        self.embedder = embedder
        self.dense = dense
        self.bm25 = BM25Index(docs)
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    async def from_database(
        cls,
        jobs_repo: JobsRepo,
        embedder: Embedder,
        reranker: Reranker | None = None,
        **kwargs: int,
    ) -> HybridSearcher:
        """
        Production setup: dense search runs in Postgres (pgvector/HNSW);
        BM25 needs the texts in memory, so we load the corpus once.
        """
        rows = await jobs_repo.get_search_corpus()
        docs = [SearchDoc.from_row(r) for r in rows]
        return cls(docs, embedder, PgVectorRetriever(jobs_repo), reranker, **kwargs)

    @classmethod
    def in_memory(
        cls,
        docs: list[SearchDoc],
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        max_words: int = 200,
        overlap: int = 30,
        **kwargs: int,
    ) -> HybridSearcher:
        """No database: embed the docs now and search them with numpy."""
        vectors = embed_job_documents(
            embedder, [d.model_dump() for d in docs], max_words=max_words, overlap=overlap)
        return cls(docs, embedder, InMemoryDenseRetriever(docs, vectors), reranker, **kwargs)

    # ── Search ──────────────────────────────────────────────────────────────

    async def search(
        self, query: str, filters: SearchFilters | None = None, top_k: int = 5,
    ) -> SearchResponse:
        started = time.perf_counter()
        query_vector = (await asyncio.to_thread(self.embedder.embed, [query]))[0]

        dense_hits, bm25_hits = await asyncio.gather(
            self.dense.search(query_vector, self.candidates, filters),
            asyncio.to_thread(self.bm25.search, query, self.candidates, filters),
        )
        return await self._fuse_and_rank(query, dense_hits, bm25_hits, top_k, started)

    async def search_by_vector(
        self, vector: np.ndarray, filters: SearchFilters | None = None, top_k: int = 5,
        label: str = "(vector)",
    ) -> SearchResponse:
        """Dense-only search — e.g. 'jobs most similar to my resume' (resume.py)."""
        started = time.perf_counter()
        dense_hits = await self.dense.search(vector, self.candidates, filters)
        hits = [self._hit(rank, doc_id, dense=(rank, score))
                for rank, (doc_id, score) in enumerate(dense_hits[:top_k], start=1)
                if doc_id in self.docs_by_id]
        return SearchResponse(query=label, hits=hits, candidates=len(dense_hits), reranked=False,
                              latency_ms=round((time.perf_counter() - started) * 1000, 1))

    async def _fuse_and_rank(
        self,
        query: str,
        dense_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
        top_k: int,
        started: float,
    ) -> SearchResponse:
        dense_info = {d: (r, s) for r, (d, s) in enumerate(dense_hits, start=1)}
        bm25_info = {d: (r, s) for r, (d, s) in enumerate(bm25_hits, start=1)}

        fused = reciprocal_rank_fusion(
            [[d for d, _ in dense_hits], [d for d, _ in bm25_hits]], k=self.rrf_k)
        # The DB may return a job added after we loaded the corpus — skip it.
        fused = [(d, s) for d, s in fused if d in self.docs_by_id][: self.candidates]

        rerank_scores: dict[str, float] = {}
        if self.reranker and fused:
            texts = [self.docs_by_id[d].rerank_text() for d, _ in fused]
            scores = await asyncio.to_thread(self.reranker.score, query, texts)
            rerank_scores = {d: s for (d, _), s in zip(fused, scores, strict=True)}
            fused.sort(key=lambda item: rerank_scores[item[0]], reverse=True)

        hits = [
            self._hit(rank, doc_id, dense=dense_info.get(doc_id), bm25=bm25_info.get(doc_id),
                      rrf=rrf, rerank=rerank_scores.get(doc_id))
            for rank, (doc_id, rrf) in enumerate(self._collapse_duplicates(fused)[:top_k], start=1)
        ]
        return SearchResponse(
            query=query, hits=hits, candidates=len(fused), reranked=bool(rerank_scores),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _collapse_duplicates(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Keep only the best-ranked job per (title, company).

        WHY: some employers post one job per country ("ML Engineer — Remote"
        × UK, US, India, Canada...). dedup_hash includes location, so they're
        stored as different jobs — correct for the database, but in search
        results 7 near-identical rows would bury everything else.
        """
        seen: set[tuple[str, str]] = set()
        kept: list[tuple[str, float]] = []
        for doc_id, score in ranked:
            doc = self.docs_by_id[doc_id]
            key = (" ".join(doc.title.lower().split()), doc.company.lower().strip())
            if key not in seen:
                seen.add(key)
                kept.append((doc_id, score))
        return kept

    def _hit(
        self,
        rank: int,
        doc_id: str,
        *,
        dense: tuple[int, float] | None = None,
        bm25: tuple[int, float] | None = None,
        rrf: float = 0.0,
        rerank: float | None = None,
    ) -> SearchHit:
        return SearchHit(
            rank=rank, doc=self.docs_by_id[doc_id],
            dense_rank=dense[0] if dense else None, dense_score=dense[1] if dense else None,
            bm25_rank=bm25[0] if bm25 else None, bm25_score=bm25[1] if bm25 else None,
            rrf_score=rrf, rerank_score=rerank,
        )
