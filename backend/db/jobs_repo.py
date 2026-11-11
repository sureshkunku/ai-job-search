"""
jobs_repo.py — Jobs Repository
Module 2: Data Layer

All database operations for the jobs table.
Agents, scrapers, and API routes use this — never raw DB calls.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from postgrest.exceptions import APIError

from backend.db.base_repo import BaseRepo
from backend.models.job import EnrichedJob, RawJob

logger = logging.getLogger(__name__)

UNIQUE_VIOLATION = "23505"  # Postgres error code for a UNIQUE constraint clash


class JobsRepo(BaseRepo):
    TABLE = "jobs"

    # ------------------------------------------------------------------
    # WRITE OPERATIONS
    # ------------------------------------------------------------------

    async def upsert_raw(self, job: RawJob) -> dict:
        """
        Insert a new job or skip if dedup_hash already exists.

        Returns:
            dict with keys: {"action": "inserted"|"skipped", "data": {...}}

        Why upsert instead of insert?
            Scrapers run every 6 hours. Same job appears on multiple runs.
            dedup_hash = SHA256(title+company+url) catches duplicates.
            ON CONFLICT DO NOTHING = idempotent scraping (safe to replay).
        """
        client = await self._client()
        row = {
            "dedup_hash": job.dedup_hash,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "description": job.description,
            "url": str(job.url) if job.url else None,
            "source": job.source.value if job.source else "unknown",
            "posted_at": job.posted_at.isoformat() if job.posted_at else None,
            # Unparsed salary text ("USD 120,000–150,000 /year") — the
            # Enrichment Agent turns it into salary_min/max in Module 6.
            "salary_raw": job.salary_raw,
            "scraped_at": datetime.now(UTC).isoformat(),
        }

        # Check if already exists (dedup)
        existing = await client.table(self.TABLE)\
            .select("id")\
            .eq("dedup_hash", job.dedup_hash)\
            .execute()

        if existing.data:
            logger.debug(f"Duplicate job skipped: {job.title} @ {job.company}")
            return {"action": "skipped", "data": existing.data[0]}

        try:
            result = await client.table(self.TABLE).insert(row).execute()
        except APIError as exc:
            # Race condition: scrapers run in parallel, so two sources can
            # both check "not there yet" and then both insert the same job.
            # The UNIQUE(dedup_hash) constraint rejects the second insert with
            # Postgres error 23505 — that's a duplicate, not a failure.
            if exc.code != UNIQUE_VIOLATION:
                raise
            logger.debug(f"Duplicate job (concurrent insert): {job.title} @ {job.company}")
            winner = await client.table(self.TABLE)\
                .select("id")\
                .eq("dedup_hash", job.dedup_hash)\
                .execute()
            return {"action": "skipped", "data": winner.data[0] if winner.data else {}}

        logger.info(f"New job inserted: {job.title} @ {job.company}")
        return {"action": "inserted", "data": result.data[0]}

    async def save_enrichment(self, job_id: str, enriched: EnrichedJob) -> dict:
        """
        Save LangGraph agent enrichment results back to jobs table.
        Called after the enrichment agent completes.

        Updates:
            - match_score, skills_match, skills_gap
            - remote_type, seniority, salary range
            - required_skills, nice_to_have_skills
            - embedding (vector, filled in Module 4)
            - MLOps tracking: model, prompt_version, tokens, cost
        """
        client = await self._client()
        updates = {
            # Enriched fields
            "remote_type": enriched.remote_type.value if enriched.remote_type else None,
            "seniority": enriched.seniority.value if enriched.seniority else None,
            "salary_min": enriched.salary_min,
            "salary_max": enriched.salary_max,
            "required_skills": enriched.required_skills,
            "nice_to_have_skills": enriched.nice_to_have_skills,
            # Scoring
            "match_score": enriched.match_score,
            "skills_match": enriched.skills_match,
            "skills_gap": enriched.skills_gap,
            # Resume tailoring
            # Model field is resume_bullet_suggestions; DB column is bullet_suggestions
            "bullet_suggestions": enriched.resume_bullet_suggestions,
            "keywords_to_add": enriched.keywords_to_add,
            # MLOps
            "enrichment_model": enriched.enrichment_model,
            "prompt_version": enriched.prompt_version,
            "enriched_at": datetime.now(UTC).isoformat(),
        }
        # Remove None values so we don't overwrite existing data with null
        updates = {k: v for k, v in updates.items() if v is not None}

        result = await client.table(self.TABLE)\
            .update(updates)\
            .eq("id", str(job_id))\
            .execute()

        return result.data[0] if result.data else {}

    async def save_embedding(self, job_id: str, embedding: list[float]) -> None:
        """
        Save the vector embedding for a job.
        Called by the embedding pipeline in Module 4.

        The embedding is a 384-dim float vector from all-MiniLM-L6-v2.
        Stored in the pgvector column, indexed with HNSW for fast cosine search.
        """
        client = await self._client()
        await client.table(self.TABLE)\
            .update({"embedding": embedding})\
            .eq("id", str(job_id))\
            .execute()

    # ------------------------------------------------------------------
    # READ OPERATIONS
    # ------------------------------------------------------------------

    async def get_by_id(self, job_id: str) -> dict | None:
        """Fetch one job by UUID."""
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("*")\
            .eq("id", job_id)\
            .execute()
        return result.data[0] if result.data else None

    async def get_top_matches(
        self,
        limit: int = 20,
        min_score: float = 50.0,
        source: str | None = None,
    ) -> list[dict]:
        """
        Fetch top matching jobs by AI score.
        Used by the Streamlit dashboard (Module 12).

        Args:
            limit: Max results to return
            min_score: Minimum match_score threshold (0-100)
            source: Filter by scrape source (linkedin, indeed, etc.)
        """
        client = await self._client()
        q = client.table(self.TABLE)\
            .select("id,title,company,location,url,source,match_score,skills_gap,remote_type,seniority,salary_min,salary_max,scraped_at")\
            .gte("match_score", min_score)\
            .order("match_score", desc=True)\
            .limit(limit)

        if source:
            q = q.eq("source", source)

        result = await q.execute()
        return result.data or []

    async def get_unenriched(self, limit: int = 50) -> list[dict]:
        """
        Fetch jobs that haven't been enriched yet.
        Used by the enrichment agent to find work to do.

        Why is_null check?
            enriched_at is NULL until the agent processes the job.
            This is the queue — agents poll this to find jobs to work on.
        """
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("id,title,company,description,url,source")\
            .is_("enriched_at", "null")\
            .order("scraped_at", desc=False)\
            .limit(limit)\
            .execute()
        return result.data or []

    async def get_unembedded(self, limit: int = 100) -> list[dict]:
        """
        Fetch jobs that have been enriched but not yet embedded.
        Used by the embedding pipeline (Module 4).
        """
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("id,title,company,description,required_skills,seniority,remote_type")\
            .not_.is_("enriched_at", "null")\
            .is_("embedding", "null")\
            .limit(limit)\
            .execute()
        return result.data or []

    async def search_by_text(self, query: str, limit: int = 20) -> list[dict]:
        """
        Full-text search using Postgres built-in search.
        Fast for keyword matching (BM25 is handled in Python, Module 4).

        Note: For production-grade search, Module 4 adds:
            - BM25 (rank-bm25 library)
            - Vector similarity (pgvector)
            - Hybrid RRF merge
            - Cross-encoder reranking
        """
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("id,title,company,location,url,match_score,remote_type")\
            .text_search("title", query)\
            .order("match_score", desc=True)\
            .limit(limit)\
            .execute()
        return result.data or []

    async def semantic_search(
        self,
        query_embedding: list[float],
        threshold: float = 0.5,
        limit: int = 10,
    ) -> list[dict]:
        """
        Vector similarity search using pgvector.
        Calls the match_jobs SQL function defined in setup_db.sql.

        Used by the RAG pipeline in Module 4.
        Returns jobs sorted by cosine similarity to the query embedding.
        """
        client = await self._client()
        result = await client.rpc(
            "match_jobs",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit,
            }
        ).execute()
        return result.data or []

    async def get_stats(self) -> dict:
        """
        Aggregate stats for the dashboard.
        Returns counts by status, source, and score distribution.
        """
        client = await self._client()

        total = await self.count()
        # eq("enriched_at", "not null") would compare a timestamp to the string
        # "not null" and fail in Postgres — use a real IS NOT NULL filter instead.
        enriched_result = await client.table(self.TABLE)\
            .select("id", count="exact")\
            .not_.is_("enriched_at", "null")\
            .execute()
        enriched = enriched_result.count or 0

        # Count by source
        result = await client.table(self.TABLE)\
            .select("source")\
            .execute()

        source_counts: dict[str, int] = {}
        for row in (result.data or []):
            s = row.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "total_jobs": total,
            "enriched_jobs": enriched,
            "unenriched_jobs": total - enriched,
            "by_source": source_counts,
        }
