"""
demo_flow.py — Walk through the job lifecycle step by step.

Run it:
    uv run python scripts/demo_flow.py          # offline: in-memory database
    uv run python scripts/demo_flow.py --live   # real Supabase (needs .env + setup_db.sql)
    uv run python scripts/demo_flow.py --live --keep   # ...and leave rows to view in Supabase

WHY this script exists:
    Modules 1–2 are libraries (models + repositories) — there's no app to "run"
    yet. This script drives the REAL code (RawJob, JobsRepo, ApplicationsRepo,
    the enrichment output models) through the same steps the finished app will
    take, so you can see what each piece does and what data it produces.

    Steps that belong to future modules (scraping, LLM calls, embeddings) are
    SIMULATED with hard-coded data and clearly labelled — they'll be replaced
    by the real thing as each module is built.

WHY an in-memory database by default:
    The repositories only talk to Supabase through `client.table(...)...execute()`.
    InMemorySupabase below implements just enough of that API to run the real
    repo code, so the demo works with no keys, no network and no DB setup —
    the same idea as the mocks in test_db.py, but with real state you can watch.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

# Make `backend` importable when run as `python scripts/demo_flow.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.config import get_settings
from backend.db import ApplicationsRepo, JobsRepo
from backend.models import (
    ApplicationStatus,
    EnrichedJob,
    EnrichmentOutput,
    JobSource,
    RawJob,
    RemoteType,
    ResumeTailorOutput,
    ScoringOutput,
    Seniority,
)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# In-memory stand-in for the Supabase async client
# ═════════════════════════════════════════════════════════════════════════════

class _Query:
    """Supports the subset of the PostgREST query builder our repos use."""

    def __init__(self, db: dict[str, list[dict]], table: str) -> None:
        self._db, self._table = db, table
        self._op, self._payload = "select", None
        self._cols, self._count = "*", None
        self._filters: list[Any] = []
        self._order: tuple[str, bool] | None = None
        self._limit: int | None = None
        self._negate_next = False

    # -- operations ---------------------------------------------------------
    def select(self, cols: str = "*", count: str | None = None) -> _Query:
        self._cols, self._count = cols, count
        return self

    def insert(self, row: dict) -> _Query:
        self._op, self._payload = "insert", row
        return self

    def update(self, values: dict) -> _Query:
        self._op, self._payload = "update", values
        return self

    # -- filters ------------------------------------------------------------
    def eq(self, col: str, val: Any) -> _Query:
        self._filters.append(lambda r: r.get(col) == val)
        return self

    def gte(self, col: str, val: Any) -> _Query:
        self._filters.append(lambda r: r.get(col) is not None and r[col] >= val)
        return self

    def is_(self, col: str, val: str) -> _Query:
        negate, self._negate_next = self._negate_next, False
        self._filters.append(lambda r: (r.get(col) is None) != negate)
        return self

    @property
    def not_(self) -> _Query:
        self._negate_next = True
        return self

    def order(self, col: str, desc: bool = False) -> _Query:
        self._order = (col, desc)
        return self

    def limit(self, n: int) -> _Query:
        self._limit = n
        return self

    # -- execution ----------------------------------------------------------
    async def execute(self) -> SimpleNamespace:
        rows = self._db.setdefault(self._table, [])
        now = datetime.now(UTC).isoformat()

        if self._op == "insert":
            row = {"id": str(uuid4()), "created_at": now, **self._payload}
            rows.append(row)
            return SimpleNamespace(data=[copy.deepcopy(row)], count=None)

        matched = [r for r in rows if all(f(r) for f in self._filters)]

        if self._op == "update":
            for r in matched:
                r.update(self._payload)
            return SimpleNamespace(data=copy.deepcopy(matched), count=None)

        total = len(matched)
        if self._order:
            col, desc = self._order
            present = sorted((r for r in matched if r.get(col) is not None),
                             key=lambda r: r[col], reverse=desc)
            matched = present + [r for r in matched if r.get(col) is None]  # NULLS LAST
        if self._limit is not None:
            matched = matched[: self._limit]
        return SimpleNamespace(data=[self._project(r) for r in matched], count=total)

    def _project(self, row: dict) -> dict:
        """Apply the select column list, including a `jobs(...)` join."""
        parts = [p.strip() for p in self._cols.split(",")]
        join_cols = None
        if "jobs(" in self._cols:
            inner = self._cols[self._cols.index("jobs(") + 5: self._cols.index(")")]
            join_cols = [c.strip() for c in inner.split(",")]
            parts = [p for p in parts if not p.startswith("jobs(") and p not in join_cols
                     and not p.endswith(")")]
        out = dict(row) if "*" in parts else {c: row.get(c) for c in parts}
        if join_cols is not None:
            job = next((j for j in self._db.get("jobs", []) if j["id"] == row.get("job_id")), {})
            out["jobs"] = {c: job.get(c) for c in join_cols}
        return copy.deepcopy(out)


class InMemorySupabase:
    def __init__(self) -> None:
        self.db: dict[str, list[dict]] = {}

    def table(self, name: str) -> _Query:
        return _Query(self.db, name)


# ═════════════════════════════════════════════════════════════════════════════
# Sample data (stands in for Module 3 scrapers and Module 6 LLM agents)
# ═════════════════════════════════════════════════════════════════════════════

SCRAPED = [
    RawJob(title="  Senior AI Engineer ", company="Acme AI", location="Remote",
           description="Build LangGraph agents in Python on AWS. FastAPI, RAG, pgvector.",
           url="https://remoteok.com/jobs/1", source=JobSource.REMOTEOK),
    RawJob(title="Senior Python Engineer", company="DataCo", location="Dallas TX",
           description="FastAPI microservices, Databricks, Spark, AWS. Hybrid 3 days.",
           url="https://themuse.com/jobs/2", source=JobSource.THEMUSE),
    RawJob(title="Frontend Developer", company="PixelWorks", location="Austin TX",
           description="React, TypeScript, design systems. Onsite.",
           url="https://arbeitnow.com/jobs/3", source=JobSource.ARBEITNOW),
    # Same job as #1, found on a different board → same dedup_hash
    RawJob(title="Senior AI Engineer", company="Acme AI", location="Remote",
           description="(cross-posted) Build LangGraph agents...",
           url="https://himalayas.app/jobs/99", source=JobSource.HIMALAYAS),
]
BOARD = ["RemoteOK", "The Muse", "Arbeitnow", "Himalayas"]

# What the Groq LLM *would* return as JSON for each job (Module 6 replaces this)
FAKE_LLM = {
    "Acme AI": dict(
        enrichment={"required_skills": ["Python", "LangGraph", "AWS", "RAG"],
                    "remote_type": "Remote", "seniority": "Senior",
                    "salary_min": 160000, "salary_max": 190000},
        scoring={"match_score": 92, "match_reasoning": "Strong overlap on agents + AWS.",
                 "skills_match": ["Python", "LangGraph", "AWS"], "skills_gap": ["RAG evals"]},
        tailor={"bullet_suggestions": ["Shipped LangGraph agent cutting triage time 40%"],
                "keywords_to_add": ["pgvector", "RAG"]},
    ),
    "DataCo": dict(
        enrichment={"required_skills": ["Python", "FastAPI", "Databricks", "Spark"],
                    "remote_type": "HYBRID", "seniority": "Lead"},  # "Lead" isn't allowed
        scoring={"match_score": 78, "match_reasoning": "Good Python/Databricks fit.",
                 "skills_match": ["Python", "FastAPI", "Databricks"], "skills_gap": ["Spark"]},
        tailor={"bullet_suggestions": ["Built Databricks pipelines processing 2TB/day"],
                "keywords_to_add": ["Spark"]},
    ),
    "PixelWorks": dict(
        enrichment={"required_skills": ["React", "TypeScript"],
                    "remote_type": "onsite", "seniority": "mid"},
        scoring={"match_score": 31, "match_reasoning": "Frontend role; little overlap.",
                 "skills_match": [], "skills_gap": ["React", "TypeScript"]},
        tailor={"bullet_suggestions": ["N/A — low match"], "keywords_to_add": []},
    ),
}


def step(n: int, title: str, module: str, simulated: bool = False) -> None:
    tag = " [yellow](simulated)[/]" if simulated else " [green](real code)[/]"
    console.print()
    console.rule(f"[bold]Step {n} — {title}[/]  ·  {module}{tag}")


# ═════════════════════════════════════════════════════════════════════════════
# The flow
# ═════════════════════════════════════════════════════════════════════════════

async def main(live: bool, keep: bool = False) -> None:
    settings = get_settings()
    jobs_repo, apps_repo = JobsRepo(), ApplicationsRepo()

    if live:
        console.print(Panel("Using your REAL Supabase project from .env.\n"
                            + ("Demo rows are KEPT so you can view them in Supabase."
                           if keep else "Demo rows are deleted again at the end."),
                            style="red"))
    else:
        fake = InMemorySupabase()
        jobs_repo._supabase = fake   # inject the stand-in — same trick as test_db.py
        apps_repo._supabase = fake

    # ── Step 0: config ──────────────────────────────────────────────────────
    step(0, "Load settings", "Module 1 · backend/config.py")
    console.print(f"app_env={settings.app_env}  model={settings.groq_model}  "
                  f"email threshold={settings.match_score_threshold}  "
                  f"embeddings={settings.embedding_model} ({settings.embedding_dimensions} dims)")
    console.print("[dim]get_settings() reads .env once (lru_cache); every module imports it.[/]")

    # ── Step 1: scrape ──────────────────────────────────────────────────────
    step(1, "Scrape job boards → RawJob", "Module 3 · backend/scrapers/", simulated=True)
    t = Table("#", "board", "title (after validator)", "company", "dedup_hash")
    for i, (job, board) in enumerate(zip(SCRAPED, BOARD), 1):
        t.add_row(str(i), board, repr(job.title), job.company, job.dedup_hash[:12] + "…")
    console.print(t)
    console.print("[dim]Note the title whitespace was stripped by RawJob's validator, and "
                  "#1 and #4 share a hash — same job on two boards.[/]")

    # ── Step 2: dedup + store ───────────────────────────────────────────────
    step(2, "Dedup + store", "Module 2 · JobsRepo.upsert_raw")
    t = Table("#", "action", "job id")
    ids: dict[str, str] = {}
    for i, job in enumerate(SCRAPED, 1):
        res = await jobs_repo.upsert_raw(job)
        ids.setdefault(job.company, res["data"]["id"])
        style = "green" if res["action"] == "inserted" else "yellow"
        t.add_row(str(i), f"[{style}]{res['action']}[/]", res["data"]["id"][:8] + "…")
    console.print(t)

    # ── Step 3: enrichment queue ────────────────────────────────────────────
    step(3, "Find work for the agents", "Module 2 · JobsRepo.get_unenriched")
    queue = await jobs_repo.get_unenriched()
    console.print(f"{len(queue)} jobs have enriched_at = NULL → queued for the agents: "
                  + ", ".join(j["company"] for j in queue))

    # ── Step 4: enrich + score + tailor ─────────────────────────────────────
    step(4, "Enrich → score → tailor", "Module 6 · LangGraph + Groq", simulated=True)
    console.print("[dim]The LLM's JSON is validated by Pydantic before we trust it:[/]")
    try:
        ScoringOutput(match_score=110, match_reasoning="hallucinated")
    except ValidationError as e:
        console.print(f"  [red]✗ rejected[/] match_score=110 → {e.errors()[0]['msg']}")

    for row in queue:
        llm = FAKE_LLM[row["company"]]
        enr = EnrichmentOutput(**llm["enrichment"])
        sco = ScoringOutput(**llm["scoring"])
        tai = ResumeTailorOutput(**llm["tailor"])
        console.print(f"  {row['company']}: remote_type "
                      f"{llm['enrichment']['remote_type']!r}→{enr.remote_type!r}, seniority "
                      f"{llm['enrichment']['seniority']!r}→{enr.seniority!r}, score {sco.match_score}")

        enriched = EnrichedJob(
            dedup_hash="", title=row["title"], company=row["company"],
            description=row["description"], url=row["url"], source=JobSource(row["source"]),
            remote_type=RemoteType(enr.remote_type), seniority=Seniority(enr.seniority),
            salary_min=enr.salary_min, salary_max=enr.salary_max,
            required_skills=enr.required_skills,
            match_score=sco.match_score, skills_match=sco.skills_match,
            skills_gap=sco.skills_gap,
            resume_bullet_suggestions=tai.bullet_suggestions,
            keywords_to_add=tai.keywords_to_add,
            enrichment_model=settings.groq_model, prompt_version="demo-v0",
        )
        await jobs_repo.save_enrichment(row["id"], enriched)
    console.print("[dim]JobsRepo.save_enrichment wrote results + enriched_at "
                  "(real code). Queue is now: "
                  f"{len(await jobs_repo.get_unenriched())} jobs.[/]")

    # ── Step 5: embeddings ──────────────────────────────────────────────────
    step(5, "Embed for semantic search", "Module 4 · sentence-transformers", simulated=True)
    console.print("Not built yet. Module 4 turns each job into a 384-number vector, "
                  "stores it with JobsRepo.save_embedding, and pgvector's HNSW index "
                  "finds similar jobs. Skipped here.")

    # ── Step 6: top matches ─────────────────────────────────────────────────
    step(6, "Show best matches", "Module 2 · JobsRepo.get_top_matches → Module 12 dashboard")
    t = Table("score", "title", "company", "remote", "salary", "gap", "email digest?")
    for j in await jobs_repo.get_top_matches(min_score=0):
        salary = (f"${j['salary_min'] // 1000}k–${j['salary_max'] // 1000}k"
                  if j.get("salary_min") and j.get("salary_max") else "—")
        digest = "[green]yes[/]" if j["match_score"] >= settings.match_score_threshold else "no"
        t.add_row(f"{j['match_score']:.0f}", j["title"], j["company"], j["remote_type"],
                  salary, ", ".join(j["skills_gap"] or []), digest)
    console.print(t)

    # ── Step 7: application pipeline ────────────────────────────────────────
    step(7, "Track your application", "Module 2 · ApplicationsRepo")
    saved = await apps_repo.save_job(ids["Acme AI"])
    app_id = saved["data"]["id"]
    console.print(f"save_job → {saved['action']} (status=saved)")
    for status in (ApplicationStatus.APPLIED, ApplicationStatus.INTERVIEW):
        row = await apps_repo.update_status(app_id, status)
        extra = f", applied_at auto-set" if status is ApplicationStatus.APPLIED else ""
        console.print(f"update_status → {row['status']}{extra}")
    again = await apps_repo.save_job(ids["Acme AI"])
    console.print(f"save_job again → {again['action']} (no duplicate application)")
    stats = await apps_repo.get_pipeline_stats()
    console.print("pipeline: " + ", ".join(f"{k}={v}" for k, v in stats.items() if v))

    # ── Step 8: stats ───────────────────────────────────────────────────────
    step(8, "Dashboard stats", "Module 2 · JobsRepo.get_stats")
    console.print(await jobs_repo.get_stats())

    if live and keep:
        console.print("[dim]--keep: demo rows left in Supabase. Open Table Editor → jobs / "
                      "applications to see them. Re-running skips them as duplicates.[/]")
    elif live:
        client = await jobs_repo._client()
        hashes = list({j.dedup_hash for j in SCRAPED})
        await client.table("jobs").delete().in_("dedup_hash", hashes).execute()
        console.print("[dim]Live demo rows deleted (applications cascade).[/]")

    console.print()
    console.rule("[bold green]Done[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk through the job lifecycle.")
    parser.add_argument("--live", action="store_true",
                        help="use your real Supabase project instead of in-memory")
    parser.add_argument("--keep", action="store_true",
                        help="with --live: leave the demo rows in Supabase to inspect")
    args = parser.parse_args()
    asyncio.run(main(args.live, args.keep))
