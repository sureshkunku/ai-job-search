"""
applications_repo.py — Applications Repository
Module 2: Data Layer

Tracks your personal job application pipeline:
saved → applied → screening → interview → offer → rejected/withdrawn
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from backend.db.base_repo import BaseRepo
from backend.models.job import ApplicationStatus

logger = logging.getLogger(__name__)


class ApplicationsRepo(BaseRepo):
    TABLE = "applications"

    async def save_job(self, job_id: str) -> dict:
        """
        Save a job to your pipeline (status=saved).
        First step when you find a job you want to track.
        """
        client = await self._client()

        # Check if already saved
        existing = await client.table(self.TABLE)\
            .select("id,status")\
            .eq("job_id", job_id)\
            .execute()
        if existing.data:
            return {"action": "already_saved", "data": existing.data[0]}

        result = await client.table(self.TABLE)\
            .insert({"job_id": job_id, "status": ApplicationStatus.SAVED.value})\
            .execute()
        return {"action": "saved", "data": result.data[0]}

    async def update_status(
        self,
        application_id: str,
        new_status: ApplicationStatus,
        **extra_fields,
    ) -> dict:
        """
        Move an application through the pipeline.

        Examples:
            await repo.update_status(id, ApplicationStatus.APPLIED, applied_at=datetime.now(UTC))
            await repo.update_status(id, ApplicationStatus.INTERVIEW, interview_date=dt)
            await repo.update_status(id, ApplicationStatus.OFFER, offer_amount=145000)
            await repo.update_status(id, ApplicationStatus.REJECTED)
        """
        client = await self._client()
        updates = {"status": new_status.value, **extra_fields}

        # Auto-set applied_at timestamp
        if new_status == ApplicationStatus.APPLIED and "applied_at" not in updates:
            updates["applied_at"] = datetime.now(UTC).isoformat()

        result = await client.table(self.TABLE)\
            .update(updates)\
            .eq("id", application_id)\
            .execute()
        return result.data[0] if result.data else {}

    async def get_pipeline(self) -> dict[str, list[dict]]:
        """
        Return all applications grouped by status.
        Used by the Streamlit dashboard Kanban view (Module 12).

        Returns:
            {
                "saved": [...],
                "applied": [...],
                "screening": [...],
                "interview": [...],
                "offer": [...],
                "rejected": [...],
            }
        """
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("*, jobs(title, company, url, match_score, location)")\
            .order("created_at", desc=True)\
            .execute()

        pipeline: dict[str, list] = {s.value: [] for s in ApplicationStatus}
        for row in (result.data or []):
            status = row.get("status", ApplicationStatus.SAVED.value)
            if status in pipeline:
                pipeline[status].append(row)
        return pipeline

    async def get_by_job(self, job_id: str) -> dict | None:
        """Check if a job is already in your pipeline."""
        client = await self._client()
        result = await client.table(self.TABLE)\
            .select("*")\
            .eq("job_id", job_id)\
            .execute()
        return result.data[0] if result.data else None

    async def add_note(self, application_id: str, note: str) -> dict:
        """Append a note to an application."""
        client = await self._client()
        result = await client.table(self.TABLE)\
            .update({"notes": note})\
            .eq("id", application_id)\
            .execute()
        return result.data[0] if result.data else {}

    async def get_pipeline_stats(self) -> dict:
        """Count applications by status for the dashboard metrics row."""
        client = await self._client()
        result = await client.table(self.TABLE).select("status").execute()

        stats = {s.value: 0 for s in ApplicationStatus}
        for row in (result.data or []):
            s = row.get("status")
            if s in stats:
                stats[s] += 1

        stats["total"] = sum(stats.values())
        return stats
