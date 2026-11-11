"""
base.py — BaseScraper: everything all scrapers share.
Module 3: Scrapers

Design pattern: Template Method
    BaseScraper.scrape() defines the fixed steps:
        fetch() → parse() each item → keyword filter → in-batch dedup
    Each source only fills in the two source-specific steps:
        fetch()  — which URLs/pages to call          (async, network)
        parse()  — how one raw item becomes a RawJob (pure, easy to test)
    Retry, backoff, rate limiting, filtering and counting are written ONCE here.

WHY these reliability features:
    - Retry with exponential backoff: free public APIs blip (timeouts,
      502s, 429 "slow down"). Waiting 1s, 2s, 4s... and trying again turns
      most blips into successes without hammering the server.
    - Only retry what can succeed later: a 404 or 403 will be the same
      next time, so fail fast on those instead of wasting 4 attempts.
    - Rate limiting (min gap between requests): these APIs are free —
      being polite is how we keep them free and avoid getting blocked.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from backend.models.job import JobSource, RawJob
from backend.scrapers.normalize import matches_keywords

logger = logging.getLogger(__name__)

# 429 = rate limited, 5xx = server trouble → worth retrying.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    """Decide if a failed request might succeed if we try again."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    # Timeouts, DNS failures, dropped connections — all transient.
    return isinstance(exc, httpx.TransportError)


@dataclass
class ScrapeResult:
    """What one scraper produced, plus counts so we can see what was dropped and why."""
    source: JobSource
    jobs: list[RawJob] = field(default_factory=list)
    fetched: int = 0              # raw items the API returned
    unparseable: int = 0          # items missing required fields
    filtered_out: int = 0         # didn't match our keywords
    duplicates_in_batch: int = 0  # same job twice in this run (e.g. two search terms)


class BaseScraper(ABC):
    source: ClassVar[JobSource]
    max_attempts: ClassVar[int] = 4      # 1 try + 3 retries

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        keywords: list[str] | None = None,
        max_pages: int = 3,
        min_interval: float = 1.0,
        backoff_base: float = 1.0,
    ) -> None:
        # WHY inject the client: one shared AsyncClient reuses connections
        # across all scrapers, and tests can pass a client that respx mocks.
        self.client = client
        self.keywords = keywords or []
        self.max_pages = max_pages
        self.min_interval = min_interval
        self.backoff_base = backoff_base
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    # ── The two steps each source implements ────────────────────────────────

    @abstractmethod
    async def fetch(self) -> list[dict[str, Any]]:
        """Call the API (all pages) and return the raw job items."""

    @abstractmethod
    def parse(self, item: dict[str, Any]) -> RawJob | None:
        """Map one raw item to a RawJob. Return None (or raise) if unusable."""

    def match_text(self, item: dict[str, Any]) -> str:
        """Extra text (tags, categories) used by the keyword filter. Optional."""
        return ""

    # ── The fixed algorithm ────────────────────────────────────────────────

    async def scrape(self) -> ScrapeResult:
        items = await self.fetch()
        result = ScrapeResult(source=self.source, fetched=len(items))
        seen_hashes: set[str] = set()

        for item in items:
            try:
                job = self.parse(item)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                # ValueError also covers Pydantic's ValidationError.
                # One malformed item must never sink the whole batch.
                logger.debug("[%s] unparseable item skipped: %s", self.source.value, exc)
                job = None
            if job is None:
                result.unparseable += 1
                continue

            if not matches_keywords(f"{job.title} {self.match_text(item)}", self.keywords):
                result.filtered_out += 1
                continue

            if job.dedup_hash in seen_hashes:
                result.duplicates_in_batch += 1
                continue
            seen_hashes.add(job.dedup_hash)
            result.jobs.append(job)

        logger.info(
            "[%s] fetched=%d kept=%d filtered=%d unparseable=%d",
            self.source.value, result.fetched, len(result.jobs),
            result.filtered_out, result.unparseable,
        )
        return result

    # ── HTTP helpers (shared retry + rate limit) ───────────────────────────

    async def _get_json(self, url: str, params: Any = None) -> Any:
        """GET a URL and return parsed JSON, with retry/backoff on transient errors."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            # waits backoff_base × 1, 2, 4, 8... seconds (capped at 30s)
            wait=wait_exponential(multiplier=self.backoff_base, max=30),
            retry=retry_if_exception(is_retryable),
            reraise=True,  # after the last attempt, raise the real error, not RetryError
        ):
            with attempt:
                await self._throttle()
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        raise RuntimeError("unreachable")  # tenacity always returns or raises above

    async def _throttle(self) -> None:
        """
        Enforce at least `min_interval` seconds between this scraper's requests.

        WHY a lock: if two coroutines of the same scraper ever request at
        once, both would read the same _last_request and skip the wait.
        """
        async with self._lock:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()
