"""
search_jobs.py — try hybrid search (Module 4).

    # No database needed: scrape the live APIs now and search the results in memory
    uv run python scripts/search_jobs.py "remote python llm engineer" --live-scrape

    # Search your Supabase jobs (after run_scrapers.py + embed_jobs.py)
    uv run python scripts/search_jobs.py "fastapi backend" --remote --days 14

    # Jobs most similar to YOUR resume (RESUME_FILE_PATH in .env)
    uv run python scripts/search_jobs.py --resume --live-scrape

Every result shows its rank in each stage — dense, BM25, RRF, rerank — so
you can see WHY it ended up where it did.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Quiet the model libraries' progress bars and Windows symlink warnings
# (must be set before they're imported).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hides the "unauthenticated requests" notice

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from backend.config import get_settings
from backend.db import JobsRepo
from backend.scrapers import run_scrape
from backend.search import (
    CrossEncoderReranker,
    HybridSearcher,
    ResumeIndex,
    SearchDoc,
    SearchFilters,
    get_embedder,
)

console = Console()


async def build_searcher(live_scrape: bool, rerank: bool) -> HybridSearcher:
    s = get_settings()
    embedder = get_embedder(s.embedding_model, s.embed_batch_size)
    reranker = CrossEncoderReranker(s.reranker_model) if rerank else None
    kwargs = {"candidates": s.search_candidates, "rrf_k": s.rrf_k}

    if not live_scrape:
        return await HybridSearcher.from_database(JobsRepo(), embedder, reranker, **kwargs)

    console.print("[dim]Scraping live APIs (nothing is saved)…[/]")
    reports = await run_scrape(dry_run=True)
    docs = [
        SearchDoc(id=f"{r.source}-{i}", title=j.title, company=j.company, location=j.location,
                  description=j.description, source=r.source, url=j.url, posted_at=j.posted_at)
        for r in reports for i, j in enumerate(r.jobs)
    ]
    started = time.perf_counter()
    console.print(f"[dim]Embedding {len(docs)} jobs locally…[/]")
    searcher = HybridSearcher.in_memory(docs, embedder, reranker, max_words=s.chunk_max_words,
                                        overlap=s.chunk_overlap_words, **kwargs)
    console.print(f"[dim]…done in {time.perf_counter() - started:.1f}s[/]")
    return searcher


def show(response, title: str) -> None:
    table = Table("#", "title", "company", "location", "dense", "bm25", "rrf", "rerank", title=title)
    for h in response.hits:
        table.add_row(
            str(h.rank), h.doc.title[:60], h.doc.company[:24], (h.doc.location or "—")[:28],
            f"#{h.dense_rank}" if h.dense_rank else "—",
            f"#{h.bm25_rank}" if h.bm25_rank else "—",
            f"{h.rrf_score:.4f}" if h.rrf_score else "—",
            f"{h.rerank_score:.2f}" if h.rerank_score is not None else "—",
        )
    console.print(table)
    console.print(f"[dim]{response.candidates} candidates · reranked={response.reranked} · "
                  f"{response.latency_ms} ms[/]")


async def main(args: argparse.Namespace) -> None:
    filters = SearchFilters(
        remote_only=args.remote, min_salary=args.min_salary,
        posted_within_days=args.days, sources=args.source or [],
    )
    searcher = await build_searcher(args.live_scrape, rerank=not args.no_rerank)

    if args.resume:
        s = get_settings()
        resume = ResumeIndex.load_or_build(
            s.resume_text, searcher.embedder, model_name=s.embedding_model,
            cache_path=Path(s.resume_index_path),
            max_words=s.chunk_max_words, overlap=s.chunk_overlap_words)
        response = await searcher.search_by_vector(resume.vector, filters, args.top, "my resume")
        show(response, "Jobs most similar to your resume")
    else:
        response = await searcher.search(args.query, filters, args.top)
        show(response, f'"{args.query}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid job search (BM25 + vectors + rerank).")
    p.add_argument("query", nargs="?", default="python ai engineer")
    p.add_argument("--live-scrape", action="store_true", help="search fresh API results, no DB")
    p.add_argument("--resume", action="store_true", help="rank jobs by similarity to your resume")
    p.add_argument("--remote", action="store_true", help="remote jobs only")
    p.add_argument("--min-salary", type=int, help="e.g. 150000 (jobs without salary are excluded)")
    p.add_argument("--days", type=int, help="posted within N days")
    p.add_argument("--source", action="append", help="remoteok / arbeitnow / himalayas (repeatable)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(p.parse_args()))
