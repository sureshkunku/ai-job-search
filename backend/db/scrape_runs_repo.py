"""
scrape_runs_repo.py — Scrape Runs Repository
Module 3: Scrapers (audit log)

One row per source per scrape: when it started/finished, how many jobs
were found / new / duplicate, and the error if it failed.

WHY keep this log:
    Scrapers fail silently in the real world — an API changes shape, starts
    returning 403, or returns 0 jobs. Without a log you only notice weeks
    later when the dashboard looks stale. This table answers "is every
    source still healthy?" at a glance, and feeds the Module 9 dashboards.
"""
from __future__ import annotations

from datetime import UTC, datetime

from backend.db.base_repo import BaseRepo


class ScrapeRunsRepo(BaseRepo):
    TABLE = "scrape_runs"

    async def start_run(self, source: str) -> str:
        """Insert a 'running' row and return its id. started_at defaults to now() in SQL."""
        client = await self._client()
        result = await client.table(self.TABLE)\
            .insert({"source": source, "status": "running"})\
            .execute()
        return str(result.data[0]["id"])

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        jobs_found: int = 0,
        jobs_new: int = 0,
        jobs_duplicate: int = 0,
        error_message: str | None = None,
    ) -> dict:
        """Close a run with its final counts. status: 'success' | 'failed'."""
        client = await self._client()
        result = await client.table(self.TABLE)\
            .update({
                "status": status,
                "finished_at": datetime.now(UTC).isoformat(),
                "jobs_found": jobs_found,
                "jobs_new": jobs_new,
                "jobs_duplicate": jobs_duplicate,
                "error_message": error_message,
            })\
            .eq("id", run_id)\
            .execute()
        return result.data[0] if result.data else {}

    async def recent_runs(self, limit: int = 20) -> list[dict]:
        """Newest runs first — for a 'scraper health' view."""
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("*")\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data or []
