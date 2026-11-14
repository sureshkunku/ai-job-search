"""
pipeline.py — embed jobs and store their vectors.
Module 4: Embeddings + RAG

    jobs (embedding IS NULL) ─▶ chunk each job ─▶ embed ALL chunks in one batch
        ─▶ mean-pool chunks per job ─▶ save_embedding (pgvector, HNSW-indexed)

WHY embed right after scraping (not after enrichment):
    The vector only needs the scraped text, so search works as soon as jobs
    land — before the LLM agents (Module 6) have run. It also means a job is
    embedded once; enrichment never forces a re-embed.

WHY asyncio.to_thread:
    model.encode() is CPU-heavy and synchronous. Calling it directly inside
    async code would freeze the event loop (FastAPI couldn't answer requests
    while a batch embeds). to_thread runs it on a worker thread instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.db.jobs_repo import JobsRepo
from backend.search.chunking import chunk_job
from backend.search.embeddings import Embedder, mean_pool

logger = logging.getLogger(__name__)


def embed_job_documents(
    embedder: Embedder,
    jobs: list[dict[str, Any]],
    *,
    max_words: int = 200,
    overlap: int = 30,
) -> np.ndarray:
    """
    Return one unit vector per job, shape (len(jobs), dims).

    All chunks of ALL jobs go to the model in a single embed() call (it
    batches internally); `owners` remembers which job each chunk came from
    so we can pool them back per job afterwards.
    """
    texts: list[str] = []
    owners: list[int] = []
    for i, job in enumerate(jobs):
        for chunk in chunk_job(
            job.get("title") or "", job.get("company") or "", job.get("description") or "",
            location=job.get("location"), skills=job.get("required_skills") or None,
            max_words=max_words, overlap=overlap,
        ):
            texts.append(chunk.text)
            owners.append(i)

    if not jobs:
        return np.zeros((0, embedder.dimensions), dtype=np.float32)
    vectors = embedder.embed(texts)
    owner_ids = np.asarray(owners)
    return np.stack([mean_pool(vectors[owner_ids == i]) for i in range(len(jobs))])


@dataclass
class PipelineReport:
    embedded: int = 0
    failed: int = 0
    seconds: float = 0.0


class EmbeddingPipeline:
    def __init__(
        self,
        jobs_repo: JobsRepo,
        embedder: Embedder,
        *,
        expected_dimensions: int = 384,
        batch_size: int = 64,
        max_words: int = 200,
        overlap: int = 30,
    ) -> None:
        self.jobs_repo = jobs_repo
        self.embedder = embedder
        self.expected_dimensions = expected_dimensions
        self.batch_size = batch_size     # jobs per batch (each job → several chunks)
        self.max_words = max_words
        self.overlap = overlap

    async def run(self, limit: int = 500) -> PipelineReport:
        """Embed up to `limit` jobs that don't have a vector yet."""
        # Fail fast with a clear message: pgvector would reject a 768-dim
        # vector for a vector(384) column, but only after we'd embedded everything.
        dims = self.embedder.dimensions
        if dims != self.expected_dimensions:
            raise ValueError(
                f"Embedding model outputs {dims} dims but the database column is "
                f"vector({self.expected_dimensions}). Change setup_db.sql or the model."
            )

        started = time.perf_counter()
        report = PipelineReport()
        rows = await self.jobs_repo.get_unembedded(limit=limit)

        for start in range(0, len(rows), self.batch_size):
            batch = rows[start:start + self.batch_size]
            vectors = await asyncio.to_thread(
                embed_job_documents, self.embedder, batch,
                max_words=self.max_words, overlap=self.overlap,
            )
            for row, vector in zip(batch, vectors, strict=True):
                try:
                    await self.jobs_repo.save_embedding(str(row["id"]), vector.tolist())
                    report.embedded += 1
                except Exception as exc:  # noqa: BLE001 — one bad row shouldn't stop the batch
                    report.failed += 1
                    logger.warning("could not save embedding for %s: %s", row.get("id"), exc)

        report.seconds = round(time.perf_counter() - started, 2)
        logger.info("embedded=%d failed=%d in %.1fs", report.embedded, report.failed, report.seconds)
        return report
