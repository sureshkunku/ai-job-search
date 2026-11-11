"""
remoteok.py — RemoteOK scraper (free, no API key).
Module 3: Scrapers

API: GET https://remoteok.com/api → JSON list of the ~100 newest remote jobs.
    - Item 0 is NOT a job: it's {"last_updated", "legal"} — we skip it.
    - No paging and no search: one call gets everything, we filter locally.
    - Text is mojibake ("â€”"), fixed by clean_text().
    - salary_min/max are 0 when unknown.

Terms of use (from the API's own "legal" field): link back to the job's
RemoteOK URL and mention RemoteOK as the source. We store each job's
RemoteOK `url` and source="remoteok", so the dashboard can do both.
"""
from __future__ import annotations

from typing import Any

from backend.models.job import JobSource, RawJob
from backend.scrapers.base import BaseScraper
from backend.scrapers.normalize import clean_text, from_epoch, from_iso, salary_text


class RemoteOKScraper(BaseScraper):
    source = JobSource.REMOTEOK
    URL = "https://remoteok.com/api"

    async def fetch(self) -> list[dict[str, Any]]:
        data = await self._get_json(self.URL)
        # Keep only real job items (the legal notice has no "position").
        return [d for d in data if isinstance(d, dict) and d.get("position")]

    def parse(self, item: dict[str, Any]) -> RawJob | None:
        title = clean_text(item["position"])
        return RawJob(
            title=title,
            company=clean_text(item["company"]),
            # Every RemoteOK job is remote; location is often "" or a region hint.
            location=clean_text(item.get("location")) or "Remote",
            description=clean_text(item.get("description")) or title,
            url=item.get("url") or item["apply_url"],
            source=self.source,
            posted_at=from_epoch(item.get("epoch")) or from_iso(item.get("date")),
            salary_raw=salary_text(item.get("salary_min"), item.get("salary_max"), "USD", "annual"),
        )

    def match_text(self, item: dict[str, Any]) -> str:
        # Tags like ["python", "ai", "senior"] catch jobs with generic titles.
        return " ".join(item.get("tags") or [])
