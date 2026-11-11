"""
orchestrator.py — run every scraper in parallel and store the results.
Module 3: Scrapers

Flow for ONE scrape run (later triggered every 6h by APScheduler, Module 11):

    build_scrapers()                    which sources are enabled
         │
    asyncio.gather(  ── fan-out ──  )   all sources at the same time
         │      │      │
     RemoteOK Arbeitnow Himalayas ...   each: start_run → scrape → upsert → finish_run
         │      │      │
    list[SourceReport]  ── fan-in ──    one summary per source

WHY parallel (asyncio fan-out):
    Scraping is almost all waiting on the network. Running 4 sources one
    after another takes the SUM of their times; asyncio.gather takes about
    the time of the SLOWEST one — with no threads, because each scraper
    yields control while it awaits a response.

WHY each source is isolated:
    _run_one catches every exception and turns it into a 'failed' report.
    So if Arbeitnow is down, RemoteOK and Himalayas still save their jobs,
    and the failure is recorded in scrape_runs instead of crashing the run.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from backend.config import Settings, get_settings
from backend.db.jobs_repo import JobsRepo
from backend.db.scrape_runs_repo import ScrapeRunsRepo
from backend.models.job import RawJob
from backend.scrapers.arbeitnow import ArbeitnowScraper
from backend.scrapers.base import BaseScraper
from backend.scrapers.himalayas import HimalayasScraper
from backend.scrapers.remoteok import RemoteOKScraper
from backend.scrapers.themuse import TheMuseScraper

logger = logging.getLogger(__name__)


@dataclass
class SourceReport:
    """Outcome of one source in one run — what you see after `run_scrapers.py`."""
    source: str
    status: str = "success"        # success | failed
    fetched: int = 0               # raw items from the API
    matched: int = 0               # passed the keyword filter (after per-source cap)
    new: int = 0                   # inserted into jobs
    duplicates: int = 0            # already in the DB (seen on an earlier run / other source)
    save_errors: int = 0           # rows the DB rejected
    error: str | None = None
    seconds: float = 0.0
    jobs: list[RawJob] = field(default_factory=list)  # kept for --dry-run display


def make_http_client(settings: Settings) -> httpx.AsyncClient:
    """One shared HTTP client: connection reuse, a timeout, and an honest User-Agent."""
    return httpx.AsyncClient(
        timeout=settings.scraper_timeout_seconds,
        headers={"User-Agent": settings.scraper_user_agent, "Accept": "application/json"},
        follow_redirects=True,
    )


def build_scrapers(
    client: httpx.AsyncClient,
    settings: Settings,
    only: Sequence[str] | None = None,
) -> list[BaseScraper]:
    """
    Create the enabled scrapers with settings applied.

    Args:
        only: restrict to these source names, e.g. ["remoteok", "himalayas"].
    """
    common = {
        "keywords": settings.scraper_title_keywords,
        "max_pages": settings.scraper_max_pages,
        "min_interval": settings.scraper_min_interval_seconds,
    }
    scrapers: list[BaseScraper] = [
        RemoteOKScraper(client, **common),
        ArbeitnowScraper(client, **common),
        HimalayasScraper(client, search_terms=settings.scraper_search_terms, **common),
    ]
    if settings.themuse_api_key:  # see themuse.py for why this needs a key
        scrapers.append(TheMuseScraper(client, api_key=settings.themuse_api_key, **common))

    if only:
        wanted = {name.strip().lower() for name in only}
        scrapers = [s for s in scrapers if s.source.value in wanted]
    return scrapers


class ScrapeOrchestrator:
    def __init__(
        self,
        scrapers: Sequence[BaseScraper],
        jobs_repo: JobsRepo | None = None,
        runs_repo: ScrapeRunsRepo | None = None,
        *,
        max_jobs_per_source: int = 200,
        dry_run: bool = False,
    ) -> None:
        if not dry_run and (jobs_repo is None or runs_repo is None):
            raise ValueError("jobs_repo and runs_repo are required unless dry_run=True")
        self.scrapers = list(scrapers)
        self.jobs_repo = jobs_repo
        self.runs_repo = runs_repo
        # Cost guardrail: every stored job is later sent to the LLM (Module 6),
        # so cap how many one source can add per run.
        self.max_jobs_per_source = max_jobs_per_source
        self.dry_run = dry_run

    async def run(self) -> list[SourceReport]:
        """Scrape all sources concurrently; never raises for a single source's failure."""
        return list(await asyncio.gather(*(self._run_one(s) for s in self.scrapers)))

    async def _run_one(self, scraper: BaseScraper) -> SourceReport:
        name = scraper.source.value
        report = SourceReport(source=name)
        started = time.perf_counter()
        run_id: str | None = None

        try:
            if not self.dry_run:
                assert self.runs_repo is not None
                run_id = await self.runs_repo.start_run(name)

            result = await scraper.scrape()
            report.fetched = result.fetched
            report.jobs = result.jobs[: self.max_jobs_per_source]
            report.matched = len(report.jobs)

            if not self.dry_run:
                await self._save(report)
        except Exception as exc:  # noqa: BLE001 — isolate each source, by design
            report.status = "failed"
            report.error = f"{type(exc).__name__}: {exc}"[:500]
            logger.exception("[%s] scrape failed", name)
        finally:
            report.seconds = round(time.perf_counter() - started, 2)
            if run_id is not None:
                await self._finish(run_id, report)
        return report

    async def _save(self, report: SourceReport) -> None:
        """Upsert each job; one bad row is counted, not fatal."""
        assert self.jobs_repo is not None
        for job in report.jobs:
            try:
                outcome = await self.jobs_repo.upsert_raw(job)
            except Exception as exc:  # noqa: BLE001
                report.save_errors += 1
                logger.warning("[%s] could not save %r: %s", report.source, job.title, exc)
                continue
            if outcome["action"] == "inserted":
                report.new += 1
            else:
                report.duplicates += 1

    async def _finish(self, run_id: str, report: SourceReport) -> None:
        """Record the outcome in scrape_runs; logging failures must not hide the real result."""
        assert self.runs_repo is not None
        try:
            await self.runs_repo.finish_run(
                run_id,
                status=report.status,
                jobs_found=report.matched,
                jobs_new=report.new,
                jobs_duplicate=report.duplicates,
                error_message=report.error,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[%s] could not record scrape_run %s", report.source, run_id)


async def run_scrape(
    *,
    dry_run: bool = False,
    only: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> list[SourceReport]:
    """
    One-call entry point: used by scripts/run_scrapers.py now and by the
    APScheduler job in Module 11.
    """
    settings = settings or get_settings()
    async with make_http_client(settings) as client:
        orchestrator = ScrapeOrchestrator(
            build_scrapers(client, settings, only),
            jobs_repo=None if dry_run else JobsRepo(),
            runs_repo=None if dry_run else ScrapeRunsRepo(),
            max_jobs_per_source=settings.max_jobs_per_run,
            dry_run=dry_run,
        )
        return await orchestrator.run()
