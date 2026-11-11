"""
Module 3: Scrapers — pull jobs from free job APIs and store them as RawJob rows.

    normalize.py     clean text, dates, salaries; keyword matching
    base.py          BaseScraper: retry, backoff, rate limit, filter, dedup
    remoteok.py      RemoteOK      (no key)
    arbeitnow.py     Arbeitnow     (no key)
    himalayas.py     Himalayas     (no key, server-side search)
    themuse.py       The Muse      (needs a free key — disabled otherwise)
    orchestrator.py  run all sources in parallel → JobsRepo + ScrapeRunsRepo
"""
from backend.scrapers.arbeitnow import ArbeitnowScraper
from backend.scrapers.base import BaseScraper, ScrapeResult
from backend.scrapers.himalayas import HimalayasScraper
from backend.scrapers.orchestrator import (
    ScrapeOrchestrator,
    SourceReport,
    build_scrapers,
    run_scrape,
)
from backend.scrapers.remoteok import RemoteOKScraper
from backend.scrapers.themuse import TheMuseScraper

__all__ = [
    "BaseScraper", "ScrapeResult",
    "RemoteOKScraper", "ArbeitnowScraper", "HimalayasScraper", "TheMuseScraper",
    "ScrapeOrchestrator", "SourceReport", "build_scrapers", "run_scrape",
]
