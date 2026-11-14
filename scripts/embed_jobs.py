"""
embed_jobs.py — give every job in Supabase a vector (Module 4).

    uv run python scripts/embed_jobs.py              # embed up to 500 unembedded jobs
    uv run python scripts/embed_jobs.py --limit 50

Run after scripts/run_scrapers.py. Needs .env + setup_db.sql run in Supabase.
The first run downloads the embedding model (~90 MB, cached afterwards).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hides the "unauthenticated requests" notice

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.db import JobsRepo
from backend.search import EmbeddingPipeline, get_embedder


async def main(limit: int) -> None:
    s = get_settings()
    pipeline = EmbeddingPipeline(
        JobsRepo(), get_embedder(s.embedding_model, s.embed_batch_size),
        expected_dimensions=s.embedding_dimensions,
        max_words=s.chunk_max_words, overlap=s.chunk_overlap_words,
    )
    report = await pipeline.run(limit=limit)
    print(f"embedded {report.embedded} jobs, {report.failed} failed, in {report.seconds}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed unembedded jobs into pgvector.")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(args.limit))
