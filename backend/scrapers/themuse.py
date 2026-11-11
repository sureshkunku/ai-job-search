"""
themuse.py — The Muse scraper (free API key required in practice).
Module 3: Scrapers

API: GET https://www.themuse.com/api/public/jobs?page=N&category=...&api_key=KEY
    → {"page", "page_count", "results": [...]}
    Each result: name (title), company.name, locations[].name, contents (HTML),
    refs.landing_page (URL), publication_date (ISO), levels[].name, categories[].name.

STATUS — disabled unless THEMUSE_API_KEY is set:
    When this module was built (Nov 2026) The Muse answered anonymous
    automated requests with 403 Forbidden. We do NOT try to get around
    that (e.g. by faking a browser User-Agent) — a site blocking bots is
    a clear "no". Registering a free key is the legitimate path:
    https://www.themuse.com/developers/api/v2
    The field mapping follows The Muse's documented schema; it hasn't been
    verified against a live response yet.
"""
from __future__ import annotations

from typing import Any

import httpx

from backend.models.job import JobSource, RawJob
from backend.scrapers.base import BaseScraper
from backend.scrapers.normalize import clean_text, from_iso

# The Muse's own category names relevant to our target roles.
CATEGORIES = ["Software Engineering", "Data and Analytics", "Data Science"]


class TheMuseScraper(BaseScraper):
    source = JobSource.THEMUSE
    URL = "https://www.themuse.com/api/public/jobs"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(client, **kwargs)
        self.api_key = api_key

    async def fetch(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for page in range(self.max_pages):  # The Muse pages start at 0
            # A list of tuples lets us repeat ?category=...&category=...
            params: list[tuple[str, str | int]] = [("page", page), ("api_key", self.api_key)]
            params += [("category", c) for c in CATEGORIES]
            data = await self._get_json(self.URL, params=params)
            items.extend(data.get("results") or [])
            if page + 1 >= int(data.get("page_count") or 0):
                break
        return items

    def parse(self, item: dict[str, Any]) -> RawJob | None:
        title = clean_text(item["name"])
        locations = [loc.get("name", "") for loc in item.get("locations") or []]
        return RawJob(
            title=title,
            company=clean_text(item["company"]["name"]),
            location=", ".join(filter(None, locations)) or None,
            description=clean_text(item.get("contents")) or title,
            url=item["refs"]["landing_page"],
            source=self.source,
            posted_at=from_iso(item.get("publication_date")),
        )

    def match_text(self, item: dict[str, Any]) -> str:
        names = [c.get("name", "") for c in item.get("categories") or []]
        names += [lvl.get("name", "") for lvl in item.get("levels") or []]
        return " ".join(names)
