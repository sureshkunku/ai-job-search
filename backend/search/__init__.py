"""
Module 4: Embeddings, RAG and hybrid search.

    chunking.py    section-aware chunking for job descriptions and resumes
    embeddings.py  local sentence-transformers embedder (all-MiniLM-L6-v2, 384 dims)
    pipeline.py    embed unembedded jobs → pgvector
    resume.py      resume reference vector + chunk retrieval (RAG for Module 6)
    bm25.py        keyword search
    dense.py       vector search: pgvector (prod) or numpy (offline)
    fusion.py      Reciprocal Rank Fusion
    rerank.py      cross-encoder reranker
    hybrid.py      the whole pipeline: dense + BM25 → RRF → rerank → top-k
"""
from backend.search.bm25 import BM25Index, tokenize
from backend.search.chunking import Chunk, chunk_job, chunk_resume
from backend.search.dense import InMemoryDenseRetriever, PgVectorRetriever
from backend.search.embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from backend.search.fusion import reciprocal_rank_fusion
from backend.search.hybrid import HybridSearcher
from backend.search.pipeline import EmbeddingPipeline, embed_job_documents
from backend.search.rerank import CrossEncoderReranker, Reranker
from backend.search.resume import ResumeIndex
from backend.search.types import SearchDoc, SearchFilters, SearchHit, SearchResponse

__all__ = [
    "BM25Index", "tokenize", "Chunk", "chunk_job", "chunk_resume",
    "InMemoryDenseRetriever", "PgVectorRetriever",
    "Embedder", "SentenceTransformerEmbedder", "get_embedder",
    "reciprocal_rank_fusion", "HybridSearcher", "EmbeddingPipeline", "embed_job_documents",
    "CrossEncoderReranker", "Reranker", "ResumeIndex",
    "SearchDoc", "SearchFilters", "SearchHit", "SearchResponse",
]
