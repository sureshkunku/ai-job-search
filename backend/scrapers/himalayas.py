"""
himalayas.py — Himalayas scraper (free, no API key).
Module 3: Scrapers

API: GET https://himalayas.app/jobs/api/search?q=TERM&page=N
    → {"jobs": [...], "totalCount": N, ...}, 20 jobs per page.
    - Unlike RemoteOK/Arbeitnow it has SERVER-SIDE search, so we send our
      search terms (settings.scraper_search_terms) and get relevant jobs
      instead of paging through everything.
    - Salaries come with currency AND period (monthly PHP, hourly USD...).
    - `pubDate` is a Unix epoch.
    - The same job can match several terms ("python" and "llm") — the
      base class drops those in-batch duplicates by dedup_hash.
"""
from __future__ import annotations

from typing import Any

import httpx

from backend.models.job import JobSource, RawJob
from backend.scrapers.base import BaseScraper
from backend.scrapers.normalize import clean_text, from_epoch, salary_text

PAGE_SIZE = 20


class HimalayasScraper(BaseScraper):
    source = JobSource.HIMALAYAS
    URL = "https://himalayas.app/jobs/api/search"

    def __init__(self, client: httpx.AsyncClient, *, search_terms: list[str], **kwargs: Any) -> None:
        super().__init__(client, **kwargs)
        self.search_terms = search_terms

    async def fetch(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for term in self.search_terms:
            for page in range(1, self.max_pages + 1):
                data = await self._get_json(self.URL, params={"q": term, "page": page})
                jobs = data.get("jobs") or []
                items.extend(jobs)
                if len(jobs) < PAGE_SIZE:  # short page = last page
                    break
        return items

    def parse(self, item: dict[str, Any]) -> RawJob | None:
        title = clean_text(item["title"])
        regions = item.get("locationRestrictions") or []
        return RawJob(
            title=title,
            company=clean_text(item["companyName"]),
            # Empty restrictions = hire from anywhere.
            location=", ".join(regions) + " (Remote)" if regions else "Remote (worldwide)",
            description=clean_text(item.get("description")) or clean_text(item.get("excerpt")) or title,
            # guid is the canonical job page; applicationLink adds utm tracking.
            url=item.get("guid") or item["applicationLink"],
            source=self.source,
            posted_at=from_epoch(item.get("pubDate")),
            salary_raw=salary_text(
                item.get("minSalary"), item.get("maxSalary"),
                item.get("currency"), item.get("salaryPeriod"),
            ),
        )

    def match_text(self, item: dict[str, Any]) -> str:
        # categories look like "Machine-Learning-Engineer" → make them word-like
        cats = (item.get("categories") or []) + (item.get("parentCategories") or [])
        return " ".join(c.replace("-", " ") for c in cats)
