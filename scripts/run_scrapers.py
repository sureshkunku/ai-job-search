"""
run_scrapers.py — run Module 3 by hand.

    uv run python scripts/run_scrapers.py --dry-run             # call the real APIs, save nothing
    uv run python scripts/run_scrapers.py --dry-run --show 10   # ...and list 10 jobs per source
    uv run python scripts/run_scrapers.py                       # scrape AND save to Supabase
    uv run python scripts/run_scrapers.py --only remoteok,himalayas

Saving needs .env (SUPABASE_URL + SUPABASE_SECRET_KEY) and scripts/setup_db.sql
run in Supabase. After a real run, look at Table Editor → jobs and scrape_runs.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from backend.scrapers import run_scrape

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the free job APIs.")
    parser.add_argument("--dry-run", action="store_true", help="don't write to Supabase")
    parser.add_argument("--only", help="comma-separated sources, e.g. remoteok,himalayas")
    parser.add_argument("--show", type=int, default=0, help="list N kept jobs per source")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    only = args.only.split(",") if args.only else None
    mode = "[yellow]DRY RUN — nothing saved[/]" if args.dry_run else "[green]saving to Supabase[/]"
    console.print(f"Scraping… {mode}")
    reports = asyncio.run(run_scrape(dry_run=args.dry_run, only=only))

    table = Table("source", "status", "fetched", "kept", "new", "duplicate", "errors", "secs")
    for r in reports:
        status = "[green]ok[/]" if r.status == "success" else f"[red]{r.status}[/]"
        saved = ("—", "—") if args.dry_run else (str(r.new), str(r.duplicates))
        table.add_row(r.source, status, str(r.fetched), str(r.matched), *saved,
                      str(r.save_errors), str(r.seconds))
    console.print(table)
    for r in reports:
        if r.error:
            console.print(f"[red]{r.source}:[/] {r.error}")

    if args.show:
        for r in reports:
            if not r.jobs:
                continue
            jobs = Table("title", "company", "location", "salary", "posted", title=r.source)
            for job in r.jobs[: args.show]:
                jobs.add_row(job.title, job.company, job.location or "—", job.salary_raw or "—",
                             job.posted_at.strftime("%Y-%m-%d") if job.posted_at else "—")
            console.print(jobs)


if __name__ == "__main__":
    main()
