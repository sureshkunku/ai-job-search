"""
test_scrapers.py — Module 3: Scrapers

No network: every HTTP call is intercepted by `respx`, which plugs into
httpx and returns the responses we script. The sample payloads below are
trimmed copies of REAL responses captured from each API (Nov 2026), so the
tests check our parsing against the shapes the APIs actually send.

What's covered:
    normalize   — mojibake, (double-escaped) HTML, dates, salary, keyword match
    base        — retry on 429/5xx/timeouts, no retry on 4xx, bad items skipped
    each source — fetch + parse + paging + filtering
    orchestrator— parallel run, per-source isolation, DB counts, scrape_runs
    repos       — concurrent-insert race, scrape_runs start/finish
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from postgrest.exceptions import APIError

from backend.config import Settings
from backend.db.jobs_repo import JobsRepo
from backend.db.scrape_runs_repo import ScrapeRunsRepo
from backend.models.job import JobSource, RawJob
from backend.scrapers import (
    ArbeitnowScraper,
    HimalayasScraper,
    RemoteOKScraper,
    ScrapeOrchestrator,
    TheMuseScraper,
    build_scrapers,
)
from backend.scrapers.base import BaseScraper, ScrapeResult
from backend.scrapers.normalize import (
    clean_text,
    fix_mojibake,
    from_epoch,
    from_iso,
    html_to_text,
    matches_keywords,
    salary_text,
)

KEYWORDS = ["python", "ai", "ml", "machine learning", "llm", "data engineer"]
# Tests must never sleep: no politeness gap, no backoff wait.
FAST = {"min_interval": 0.0, "backoff_base": 0.0}


# ===========================================================================
# Sample payloads (trimmed from real API responses)
# ===========================================================================

REMOTEOK_PAYLOAD = [
    {"last_updated": 1794355215, "legal": "API Terms of Service: Please link back..."},
    {
        "slug": "remote-senior-ai-engineer-acme-1", "id": "1",
        "epoch": 1794261602, "date": "2026-11-09T22:00:02+00:00",
        "company": "Acme AI",
        # real RemoteOK mojibake: "—" arrives as "â\u0080\u0094"
        "position": "Senior AI Engineer â\u0080\u0094 Agents",
        "tags": ["python", "llm", "senior"],
        "description": "<p>Build <b>LangGraph</b> agents.</p><ul><li>Python</li><li>AWS</li></ul>",
        "location": "", "salary_min": 150000, "salary_max": 190000,
        "apply_url": "https://remoteOK.com/remote-jobs/1",
        "url": "https://remoteOK.com/remote-jobs/remote-senior-ai-engineer-acme-1",
    },
    {
        "slug": "kubernetes-go-2", "id": "2", "epoch": 1794261000,
        "date": "2026-11-09T21:50:00+00:00", "company": "Buoyant",
        "position": "Senior Software Engineer", "tags": ["golang", "dev"],
        "description": "Go and Kubernetes.", "location": "", "salary_min": 0,
        "salary_max": 0, "apply_url": "https://remoteOK.com/2", "url": "https://remoteOK.com/2",
    },
]


def arbeitnow_page(jobs: list[dict], has_next: bool) -> dict:
    return {
        "data": jobs,
        "links": {"first": "...", "next": "https://www.arbeitnow.com/api/job-board-api?page=2" if has_next else None},
        "meta": {"current_page": 1},
    }


ARBEITNOW_JOB = {
    "slug": "ml-engineer-berlin-1", "company_name": "DataCo GmbH",
    "title": "Machine Learning Engineer (w/m/d)",
    # real Arbeitnow quirk: double-escaped HTML
    "description": "&lt;p&gt;Train &amp;amp; deploy models&lt;/p&gt;",
    "remote": True, "url": "https://www.arbeitnow.com/jobs/companies/dataco/ml-engineer-berlin-1",
    "tags": ["Remote", "Development"], "job_types": ["berufserfahren"],
    "location": "Berlin", "created_at": 1794358831,
}
ARBEITNOW_SALES = {
    **ARBEITNOW_JOB, "slug": "sales-2", "title": "Account Executive EMEA",
    "tags": ["Sales"], "url": "https://www.arbeitnow.com/jobs/sales-2", "remote": False,
}


def himalayas_job(title: str, company: str, **extra: object) -> dict:
    return {
        "title": title, "excerpt": "Short excerpt.", "companyName": company,
        "companySlug": company.lower(), "employmentType": "Full Time",
        "minSalary": 120000, "maxSalary": 160000, "salaryPeriod": "annual",
        "seniority": ["Senior"], "currency": "USD", "locationRestrictions": ["United States"],
        "categories": ["Machine-Learning-Engineer"], "parentCategories": ["Data Science"],
        "description": "<p>Ship LLM features.</p>", "pubDate": 1794315610,
        "applicationLink": f"https://himalayas.app/companies/{company.lower()}/jobs/x?utm_source=api",
        "guid": f"https://himalayas.app/companies/{company.lower()}/jobs/x",
        **extra,
    }


# ===========================================================================
# normalize.py
# ===========================================================================

class TestNormalize:
    def test_fix_mojibake_repairs_remoteok_text(self):
        assert fix_mojibake("Engineer â\u0080\u0094 Remote") == "Engineer — Remote"

    def test_fix_mojibake_leaves_good_text_alone(self):
        assert fix_mojibake("Café — Zürich") == "Café — Zürich"

    def test_html_to_text_keeps_structure(self):
        text = html_to_text("<p>Hello <b>world</b></p><ul><li>One</li><li>Two</li></ul>")
        assert "Hello world" in text
        assert "• One" in text and "• Two" in text
        assert "<" not in text

    def test_html_to_text_handles_double_escaped_html(self):
        assert html_to_text("&lt;p&gt;Train &amp;amp; deploy&lt;/p&gt;") == "Train & deploy"

    def test_clean_text_none_is_empty(self):
        assert clean_text(None) == ""

    def test_from_epoch(self):
        assert from_epoch(1794358831) == datetime.fromtimestamp(1794358831, tz=UTC)
        assert from_epoch(0) is None and from_epoch(None) is None

    def test_from_iso_with_z(self):
        assert from_iso("2026-11-09T22:00:02Z") == datetime(2026, 11, 9, 22, 0, 2, tzinfo=UTC)
        assert from_iso("not a date") is None

    @pytest.mark.parametrize(("lo", "hi", "cur", "per", "expected"), [
        (120000, 150000, "USD", "annual", "USD 120,000–150,000 /year"),
        (20000, 27000, "PHP", "monthly", "PHP 20,000–27,000 /month"),
        (0, 0, "USD", None, None),            # 0 means "not given"
        (None, None, "USD", None, None),
        (90000, None, "EUR", None, "EUR 90,000+"),
    ])
    def test_salary_text(self, lo, hi, cur, per, expected):
        assert salary_text(lo, hi, cur, per) == expected

    def test_keywords_match_whole_words_only(self):
        assert matches_keywords("Senior ML Engineer", KEYWORDS)
        assert matches_keywords("AI/ML Platform Lead", KEYWORDS)
        assert not matches_keywords("HTML Email Developer", KEYWORDS)  # 'ml' inside 'html'
        assert not matches_keywords("Maintenance Technician", KEYWORDS)  # 'ai' inside a word

    def test_empty_keyword_list_keeps_everything(self):
        assert matches_keywords("Anything at all", [])


# ===========================================================================
# BaseScraper: retry / backoff / bad items
# ===========================================================================

class _EchoScraper(BaseScraper):
    """Minimal scraper used to test the shared machinery."""
    source = JobSource.UNKNOWN
    URL = "https://api.example.com/jobs"

    async def fetch(self):
        return await self._get_json(self.URL)

    def parse(self, item):
        return RawJob(title=item["title"], company=item["company"],
                      description="d", url="https://x.com/" + item["title"])


class TestBaseScraper:
    @respx.mock
    async def test_retries_429_then_succeeds(self):
        route = respx.get(_EchoScraper.URL).mock(side_effect=[
            httpx.Response(429),
            httpx.Response(503),
            httpx.Response(200, json=[{"title": "Python Dev", "company": "A"}]),
        ])
        async with httpx.AsyncClient() as client:
            result = await _EchoScraper(client, **FAST).scrape()
        assert route.call_count == 3
        assert len(result.jobs) == 1

    @respx.mock
    async def test_retries_timeouts(self):
        route = respx.get(_EchoScraper.URL).mock(side_effect=[
            httpx.ConnectTimeout("slow"),
            httpx.Response(200, json=[]),
        ])
        async with httpx.AsyncClient() as client:
            await _EchoScraper(client, **FAST).scrape()
        assert route.call_count == 2

    @respx.mock
    async def test_does_not_retry_403(self):
        """A block won't go away by retrying — fail fast."""
        route = respx.get(_EchoScraper.URL).mock(return_value=httpx.Response(403))
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await _EchoScraper(client, **FAST).scrape()
        assert route.call_count == 1

    @respx.mock
    async def test_gives_up_after_max_attempts(self):
        route = respx.get(_EchoScraper.URL).mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await _EchoScraper(client, **FAST).scrape()
        assert route.call_count == _EchoScraper.max_attempts

    @respx.mock
    async def test_bad_items_are_counted_not_fatal(self):
        respx.get(_EchoScraper.URL).mock(return_value=httpx.Response(200, json=[
            {"title": "Python Dev", "company": "A"},
            {"title": "No company field"},          # KeyError → unparseable
            {"title": "Python Dev", "company": "A"},  # same job twice → in-batch dup
        ]))
        async with httpx.AsyncClient() as client:
            result = await _EchoScraper(client, **FAST).scrape()
        assert (len(result.jobs), result.unparseable, result.duplicates_in_batch) == (1, 1, 1)


# ===========================================================================
# Individual sources
# ===========================================================================

class TestRemoteOK:
    @respx.mock
    async def test_parses_jobs_skips_legal_notice_and_filters(self):
        respx.get(RemoteOKScraper.URL).mock(return_value=httpx.Response(200, json=REMOTEOK_PAYLOAD))
        async with httpx.AsyncClient() as client:
            result = await RemoteOKScraper(client, keywords=KEYWORDS, **FAST).scrape()

        assert result.fetched == 2          # legal notice isn't a job
        assert result.filtered_out == 1     # the Go/Kubernetes role
        job = result.jobs[0]
        assert job.title == "Senior AI Engineer — Agents"   # mojibake fixed
        assert job.source is JobSource.REMOTEOK
        assert job.location == "Remote"
        assert job.salary_raw == "USD 150,000–190,000 /year"
        assert "• Python" in job.description and "<" not in job.description
        assert job.posted_at == datetime.fromtimestamp(1794261602, tz=UTC)

    @respx.mock
    async def test_tags_can_match_when_title_is_generic(self):
        payload = [{**REMOTEOK_PAYLOAD[2], "tags": ["python", "backend"]}]
        respx.get(RemoteOKScraper.URL).mock(return_value=httpx.Response(200, json=payload))
        async with httpx.AsyncClient() as client:
            result = await RemoteOKScraper(client, keywords=KEYWORDS, **FAST).scrape()
        assert len(result.jobs) == 1


class TestArbeitnow:
    @respx.mock
    async def test_pages_until_no_next_link(self):
        route = respx.get(ArbeitnowScraper.URL).mock(side_effect=[
            httpx.Response(200, json=arbeitnow_page([ARBEITNOW_JOB], has_next=True)),
            httpx.Response(200, json=arbeitnow_page([ARBEITNOW_SALES], has_next=False)),
        ])
        async with httpx.AsyncClient() as client:
            result = await ArbeitnowScraper(client, keywords=KEYWORDS, max_pages=5, **FAST).scrape()

        assert route.call_count == 2            # stopped early, not 5 pages
        assert result.fetched == 2 and result.filtered_out == 1
        job = result.jobs[0]
        assert job.location == "Berlin (Remote)"
        assert job.description == "Train & deploy models"   # double-escaped HTML fixed
        assert job.source is JobSource.ARBEITNOW

    @respx.mock
    async def test_respects_max_pages(self):
        route = respx.get(ArbeitnowScraper.URL).mock(
            return_value=httpx.Response(200, json=arbeitnow_page([ARBEITNOW_JOB], has_next=True)))
        async with httpx.AsyncClient() as client:
            await ArbeitnowScraper(client, max_pages=2, **FAST).scrape()
        assert route.call_count == 2


class TestHimalayas:
    @respx.mock
    async def test_searches_each_term_and_dedups_across_terms(self):
        shared = himalayas_job("LLM Engineer", "Acme")
        route = respx.get(HimalayasScraper.URL).mock(side_effect=[
            httpx.Response(200, json={"jobs": [shared, himalayas_job("Data Engineer", "DataCo")]}),
            httpx.Response(200, json={"jobs": [shared]}),   # same job found by 2nd term
        ])
        async with httpx.AsyncClient() as client:
            result = await HimalayasScraper(
                client, search_terms=["python", "llm"], keywords=KEYWORDS, **FAST).scrape()

        assert route.call_count == 2                     # one short page per term
        assert [c.request.url.params["q"] for c in route.calls] == ["python", "llm"]
        assert len(result.jobs) == 2 and result.duplicates_in_batch == 1
        job = result.jobs[0]
        assert job.location == "United States (Remote)"
        assert job.url == "https://himalayas.app/companies/acme/jobs/x"   # canonical, no utm
        assert job.salary_raw == "USD 120,000–160,000 /year"

    def test_worldwide_when_no_location_restrictions(self):
        scraper = HimalayasScraper(MagicMock(), search_terms=[], **FAST)
        job = scraper.parse(himalayas_job("AI Engineer", "Z", locationRestrictions=[],
                                          minSalary=None, maxSalary=None))
        assert job.location == "Remote (worldwide)" and job.salary_raw is None


class TestTheMuse:
    def test_parses_documented_shape(self):
        item = {
            "name": "Senior Data Engineer", "contents": "<p>Pipelines</p>",
            "publication_date": "2026-11-01T10:00:00Z",
            "locations": [{"name": "New York, NY"}, {"name": "Flexible / Remote"}],
            "categories": [{"name": "Data and Analytics"}], "levels": [{"name": "Senior Level"}],
            "refs": {"landing_page": "https://www.themuse.com/jobs/acme/senior-data-engineer"},
            "company": {"name": "Acme"},
        }
        job = TheMuseScraper(MagicMock(), api_key="k", **FAST).parse(item)
        assert job.title == "Senior Data Engineer"
        assert job.location == "New York, NY, Flexible / Remote"
        assert job.posted_at == datetime(2026, 11, 1, 10, tzinfo=UTC)

    def test_disabled_without_api_key(self):
        names = {s.source.value for s in build_scrapers(MagicMock(), Settings(_env_file=None))}
        assert names == {"remoteok", "arbeitnow", "himalayas"}
        with_key = Settings(_env_file=None, themuse_api_key="k")
        assert "themuse" in {s.source.value for s in build_scrapers(MagicMock(), with_key)}

    def test_only_filter(self):
        scrapers = build_scrapers(MagicMock(), Settings(_env_file=None), only=["remoteok"])
        assert [s.source.value for s in scrapers] == ["remoteok"]


# ===========================================================================
# Orchestrator
# ===========================================================================

def _job(title: str) -> RawJob:
    return RawJob(title=title, company="Co", description="d", url=f"https://x.com/{title}")


class _FakeScraper(BaseScraper):
    def __init__(self, source: JobSource, jobs=None, error: Exception | None = None):
        super().__init__(MagicMock(), **FAST)
        self.source = source  # type: ignore[misc]
        self._jobs, self._error = jobs or [], error

    async def fetch(self):
        return []

    def parse(self, item):
        return None

    async def scrape(self):
        if self._error:
            raise self._error
        return ScrapeResult(source=self.source, jobs=self._jobs, fetched=len(self._jobs))


@pytest.fixture
def repos():
    jobs_repo = MagicMock(spec=JobsRepo)
    # first job new, second already in DB
    jobs_repo.upsert_raw = AsyncMock(side_effect=lambda job: {
        "action": "skipped" if job.title.endswith("old") else "inserted", "data": {}})
    runs_repo = MagicMock(spec=ScrapeRunsRepo)
    runs_repo.start_run = AsyncMock(side_effect=lambda source: f"run-{source}")
    runs_repo.finish_run = AsyncMock(return_value={})
    return jobs_repo, runs_repo


class TestOrchestrator:
    async def test_one_failing_source_does_not_stop_others(self, repos):
        jobs_repo, runs_repo = repos
        orchestrator = ScrapeOrchestrator(
            [_FakeScraper(JobSource.REMOTEOK, [_job("AI new"), _job("ML old")]),
             _FakeScraper(JobSource.ARBEITNOW, error=httpx.ConnectError("down"))],
            jobs_repo, runs_repo,
        )
        ok, failed = await orchestrator.run()

        assert (ok.status, ok.new, ok.duplicates) == ("success", 1, 1)
        assert failed.status == "failed" and "ConnectError" in failed.error
        # Both runs were recorded in scrape_runs, including the failure
        finished = {c.args[0]: c.kwargs for c in runs_repo.finish_run.call_args_list}
        assert finished["run-remoteok"]["jobs_new"] == 1
        assert finished["run-arbeitnow"]["status"] == "failed"

    async def test_caps_jobs_per_source(self, repos):
        jobs_repo, runs_repo = repos
        scraper = _FakeScraper(JobSource.HIMALAYAS, [_job(f"AI {i}") for i in range(5)])
        (report,) = await ScrapeOrchestrator([scraper], jobs_repo, runs_repo,
                                             max_jobs_per_source=2).run()
        assert report.matched == 2 and jobs_repo.upsert_raw.await_count == 2

    async def test_save_error_is_counted_not_fatal(self, repos):
        jobs_repo, runs_repo = repos
        jobs_repo.upsert_raw = AsyncMock(side_effect=[RuntimeError("db"), {"action": "inserted"}])
        scraper = _FakeScraper(JobSource.REMOTEOK, [_job("AI 1"), _job("AI 2")])
        (report,) = await ScrapeOrchestrator([scraper], jobs_repo, runs_repo).run()
        assert (report.status, report.new, report.save_errors) == ("success", 1, 1)

    async def test_dry_run_touches_no_database(self):
        scraper = _FakeScraper(JobSource.REMOTEOK, [_job("AI 1")])
        (report,) = await ScrapeOrchestrator([scraper], dry_run=True).run()
        assert report.matched == 1 and report.new == 0

    def test_requires_repos_when_saving(self):
        with pytest.raises(ValueError):
            ScrapeOrchestrator([], dry_run=False)


# ===========================================================================
# Repositories used by the scrapers
# ===========================================================================

def _mock_client():
    """Supabase query-builder mock, same pattern as test_db.py."""
    client, qb = MagicMock(), MagicMock()
    qb.execute = AsyncMock()
    for method in ("select", "eq", "insert", "update", "order", "limit"):
        getattr(qb, method).return_value = qb
    client.table.return_value = qb
    return client, qb


class TestScrapeRepos:
    async def test_upsert_treats_concurrent_unique_violation_as_duplicate(self):
        client, qb = _mock_client()
        qb.execute.side_effect = [
            MagicMock(data=[]),                                              # not there yet
            APIError({"message": "duplicate key", "code": "23505", "hint": None, "details": None}),
            MagicMock(data=[{"id": "winner"}]),                              # re-select
        ]
        repo = JobsRepo()
        repo._supabase = client
        result = await repo.upsert_raw(_job("AI Engineer"))
        assert result == {"action": "skipped", "data": {"id": "winner"}}

    async def test_upsert_reraises_other_db_errors(self):
        client, qb = _mock_client()
        qb.execute.side_effect = [
            MagicMock(data=[]),
            APIError({"message": "column missing", "code": "PGRST204", "hint": None, "details": None}),
        ]
        repo = JobsRepo()
        repo._supabase = client
        with pytest.raises(APIError):
            await repo.upsert_raw(_job("AI Engineer"))

    async def test_upsert_saves_salary_raw(self):
        client, qb = _mock_client()
        qb.execute.side_effect = [MagicMock(data=[]), MagicMock(data=[{"id": "1"}])]
        repo = JobsRepo()
        repo._supabase = client
        job = RawJob(title="AI", company="Co", description="d", url="https://x.com/1",
                     source=JobSource.HIMALAYAS, salary_raw="USD 100,000–120,000 /year")
        await repo.upsert_raw(job)
        row = qb.insert.call_args.args[0]
        assert row["salary_raw"] == "USD 100,000–120,000 /year"
        assert row["source"] == "himalayas"

    async def test_scrape_run_start_and_finish(self):
        client, qb = _mock_client()
        qb.execute.side_effect = [MagicMock(data=[{"id": "run-1"}]),
                                  MagicMock(data=[{"id": "run-1", "status": "success"}])]
        repo = ScrapeRunsRepo()
        repo._supabase = client
        run_id = await repo.start_run("remoteok")
        await repo.finish_run(run_id, status="success", jobs_found=3, jobs_new=2, jobs_duplicate=1)

        assert run_id == "run-1"
        assert qb.insert.call_args.args[0] == {"source": "remoteok", "status": "running"}
        update = qb.update.call_args.args[0]
        assert update["jobs_new"] == 2 and update["status"] == "success" and update["finished_at"]
