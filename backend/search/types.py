"""
types.py — the data shapes search works with.
Module 4: Embeddings + RAG
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class SearchDoc(BaseModel):
    """One job as the search layer sees it (a row of the jobs table, minus the vector)."""
    id: str
    title: str
    company: str
    description: str = ""
    location: str | None = None
    source: str = "unknown"
    url: str | None = None
    remote_type: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    match_score: float | None = None
    posted_at: datetime | None = None
    required_skills: list[str] = Field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SearchDoc:
        """Build from a Supabase row (ignores extra columns, tolerates NULL lists)."""
        data = {k: row.get(k) for k in cls.model_fields if k in row}
        data["id"] = str(row["id"])
        data["required_skills"] = row.get("required_skills") or []
        data["description"] = row.get("description") or ""
        return cls(**data)

    @property
    def is_remote(self) -> bool:
        # remote_type is set by the Enrichment Agent (Module 6). Until then,
        # scrapers put "Remote" in the location, so check both.
        return self.remote_type == "remote" or "remote" in (self.location or "").lower()

    def bm25_text(self) -> str:
        """
        Text indexed by BM25. Title and skills are repeated so a keyword in
        the title counts more than the same word buried in the description.
        """
        skills = " ".join(self.required_skills)
        return " ".join([self.title, self.title, self.company, skills, skills,
                         self.location or "", self.description[:3000]])

    def rerank_text(self) -> str:
        """Text shown to the cross-encoder. It reads ~512 tokens max, so keep the start."""
        return f"{self.title} at {self.company} ({self.location or 'n/a'}). {self.description[:1500]}"


class SearchFilters(BaseModel):
    """
    Metadata filters applied BEFORE ranking — "remote jobs over $150k from
    the last week". The same rules run in Python (BM25 / in-memory) and in
    SQL (match_jobs), so both retrievers agree on what's eligible.

    Missing data is EXCLUDED when you filter on it: a job with no salary
    can't prove it pays ≥ $150k. (Salaries are mostly filled by Module 6.)
    """
    remote_only: bool = False
    min_salary: int | None = None
    posted_within_days: int | None = None
    sources: list[str] = Field(default_factory=list)
    min_match_score: float | None = None

    @property
    def posted_after(self) -> datetime | None:
        if self.posted_within_days is None:
            return None
        return datetime.now(UTC) - timedelta(days=self.posted_within_days)

    def matches(self, doc: SearchDoc) -> bool:
        if self.remote_only and not doc.is_remote:
            return False
        if self.min_salary is not None:
            best = max(doc.salary_max or 0, doc.salary_min or 0)
            if best < self.min_salary:
                return False
        if self.posted_after is not None:
            if doc.posted_at is None:
                return False
            posted = doc.posted_at if doc.posted_at.tzinfo else doc.posted_at.replace(tzinfo=UTC)
            if posted < self.posted_after:
                return False
        if self.sources and doc.source not in self.sources:
            return False
        if self.min_match_score is not None and (doc.match_score or 0) < self.min_match_score:  # noqa: SIM103 — one rule per `if` reads clearer
            return False
        return True


class SearchHit(BaseModel):
    """
    One result, with the score from EVERY stage so you can see why it ranked
    where it did (great for learning, and for debugging bad results).
    """
    rank: int
    doc: SearchDoc
    dense_rank: int | None = None      # position in vector search (1 = best), None = not found
    dense_score: float | None = None   # cosine similarity
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None  # cross-encoder relevance (higher = better)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    candidates: int        # how many distinct jobs the retrievers proposed
    reranked: bool
    latency_ms: float
