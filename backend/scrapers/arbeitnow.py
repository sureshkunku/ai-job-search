"""
arbeitnow.py — Arbeitnow scraper (free, no API key).
Module 3: Scrapers

API: GET https://www.arbeitnow.com/api/job-board-api?page=N
    → {"data": [...jobs], "links": {"next": url | null}, "meta": {...}}
    - Mostly European (often German) jobs; `remote` is a boolean.
    - `created_at` is a Unix epoch.
    - Some descriptions are double-escaped HTML ("&lt;div&gt;") — clean_text handles it.
    - No search parameter, so we page through the newest jobs and filter locally.
"""
from __future__ import annotations

from typing import Any

from backend.models.job import JobSource, RawJob
from backend.scrapers.base import BaseScraper
from backend.scrapers.normalize import clean_text, from_epoch


class ArbeitnowScraper(BaseScraper):
    source = JobSource.ARBEITNOW
    URL = "https://www.arbeitnow.com/api/job-board-api"

    async def fetch(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for page in range(1, self.max_pages + 1):
            data = await self._get_json(self.URL, params={"page": page})
            items.extend(data.get("data") or [])
            # WHY check links.next: stop as soon as the API says there's no
            # more, instead of requesting empty pages up to max_pages.
            if not (data.get("links") or {}).get("next"):
                break
        return items

    def parse(self, item: dict[str, Any]) -> RawJob | None:
        title = clean_text(item["title"])
        location = clean_text(item.get("location"))
        if item.get("remote"):
            # Keep the remote flag visible for the Enrichment Agent (remote_type).
            location = f"{location} (Remote)" if location else "Remote"
        return RawJob(
            title=title,
            company=clean_text(item["company_name"]),
            location=location or None,
            description=clean_text(item.get("description")) or title,
            url=item["url"],
            source=self.source,
            posted_at=from_epoch(item.get("created_at")),
        )

    def match_text(self, item: dict[str, Any]) -> str:
        return " ".join((item.get("tags") or []) + (item.get("job_types") or []))
