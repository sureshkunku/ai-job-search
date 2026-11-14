"""
resume.py — embed your resume once and keep it as a reference.
Module 4: Embeddings + RAG

Two things come out of your resume:
    1. A REFERENCE VECTOR (mean of all chunks) → "find jobs most like my
       resume" is just a vector search with this as the query.
    2. CHUNK VECTORS (one per section) → retrieval for the Resume Tailor
       Agent in Module 6: given a job, pull the 2–3 resume sections that are
       most relevant and hand ONLY those to the LLM. That's RAG — fewer
       tokens, less noise, and the LLM can't "forget" the relevant bit.

WHY cache to disk:
    Your resume rarely changes, so re-embedding it on every run is waste.
    The cache key is a hash of (model name + resume text): edit the resume
    or switch models and the cache is rebuilt automatically.
    The cache file contains your resume text — it lives in data/ and is
    gitignored, so it never reaches GitHub.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from backend.search.chunking import Chunk, chunk_resume
from backend.search.embeddings import Embedder, mean_pool, normalize

logger = logging.getLogger(__name__)


class ResumeIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("need one vector per chunk, and at least one chunk")
        self.chunks = chunks
        self.vectors = normalize(vectors)
        self.vector = mean_pool(self.vectors)   # the reference vector

    @staticmethod
    def cache_key(text: str, model_name: str) -> str:
        return hashlib.sha256(f"{model_name}\n{text}".encode()).hexdigest()

    @classmethod
    def build(cls, text: str, embedder: Embedder, *, max_words: int = 200, overlap: int = 30) -> ResumeIndex:
        chunks = chunk_resume(text, max_words=max_words, overlap=overlap)
        if not chunks:
            raise ValueError("resume text is empty")
        return cls(chunks, embedder.embed([c.text for c in chunks]))

    @classmethod
    def load_or_build(
        cls,
        text: str,
        embedder: Embedder,
        *,
        model_name: str,
        cache_path: Path | None,
        max_words: int = 200,
        overlap: int = 30,
    ) -> ResumeIndex:
        key = cls.cache_key(text, model_name)
        if cache_path and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("key") == key:
                    chunks = [Chunk(**c) for c in cached["chunks"]]
                    return cls(chunks, np.asarray(cached["vectors"], dtype=np.float32))
                logger.info("resume or model changed — rebuilding resume index")
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("ignoring unreadable resume cache: %s", exc)

        index = cls.build(text, embedder, max_words=max_words, overlap=overlap)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "key": key,
                "chunks": [c.__dict__ for c in index.chunks],
                "vectors": index.vectors.round(6).tolist(),
            }), encoding="utf-8")
        return index

    def top_chunks(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        """The k resume chunks most similar to a query/job vector (cosine)."""
        scores = self.vectors @ normalize(np.asarray(query_vector, dtype=np.float32))
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]
